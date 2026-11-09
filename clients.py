"""
clients.py — Live API clients: ClinicalTrials.gov v2, NCBI E-utilities, cBioPortal.

Fixes relative to v6–v8
  * Cache keys are SHA-1 hashes of the full request (v6 truncated keys to 120 chars and
    silently collided). Cached responses never expire, so re-runs reproduce the same
    inputs. Transient failures are never cached; 404s are cached as "absent".
  * Retries with back-off on 429/5xx; the NCBI rate limit is enforced.
  * ClinicalTrials.gov: no page cap (v6 stopped at 5,000 trials in API order, a
    non-random subset), Phase 2 and Phase 3, results section included, design fields
    parsed. Actual enrollment is deliberately NOT parsed (it shrinks when a trial stops
    early, i.e. it leaks the outcome).
  * PubMed counts are time-bounded: only records published before a given year. (Issue 3)
  * PubMed efetch is parsed as XML (efetch has no JSON mode; v8's parser never ran).
  * cBioPortal frequencies come from real per-study mutation / CNA calls.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (CACHE_DIR, CBIOPORTAL_API, CTGOV_API, NCBI_API_KEY, NCBI_EMAIL,
                    NCBI_EUTILS, PHASES, RAW_DIR, log)


# ── HTTP plumbing ────────────────────────────────────────────────────────────
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET", "POST"}), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = f"MechanisticTrialEngine/9.0 (academic research; {NCBI_EMAIL or 'no-contact'})"
    return s


SESSION = _session()


class RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


NCBI_LIMITER = RateLimiter(9.0 if NCBI_API_KEY else 2.8)
_ABSENT = {"__absent__": True}
_KEY_EXCLUDE = {"api_key", "email", "tool"}


def cached_request(method: str, url: str, *, params: dict | None = None, json_body=None,
                   namespace: str = "misc", timeout: int = 60, parse: str = "json",
                   limiter: RateLimiter | None = None):
    key_params = {k: v for k, v in (params or {}).items() if k not in _KEY_EXCLUDE}
    key = json.dumps({"m": method, "u": url, "p": key_params, "j": json_body}, sort_keys=True, default=str)
    digest = hashlib.sha1(key.encode()).hexdigest()
    path = CACHE_DIR / namespace / digest[:2] / f"{digest}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        return None if payload == _ABSENT else payload["body"]
    if limiter:
        limiter.wait()
    try:
        resp = SESSION.request(method, url, params=params, json=json_body, timeout=timeout)
    except requests.RequestException as exc:
        log.warning(f"{method} {url} failed: {exc}")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if resp.status_code == 404:
        path.write_text(json.dumps(_ABSENT))
        return None
    if not resp.ok:
        log.warning(f"{method} {url} -> HTTP {resp.status_code}")
        return None
    try:
        body = resp.json() if parse == "json" else resp.text
    except ValueError:
        log.warning(f"Non-JSON response from {url}")
        return None
    path.write_text(json.dumps({"body": body, "fetched_at": datetime.now(timezone.utc).isoformat()}))
    return body


# ── ClinicalTrials.gov ───────────────────────────────────────────────────────
ONCOLOGY_QUERY = ("cancer OR neoplasm OR tumor OR carcinoma OR lymphoma OR leukemia OR sarcoma "
                  "OR melanoma OR myeloma OR glioma OR glioblastoma OR mesothelioma")
CTGOV_FIELDS = ["protocolSection", "hasResults", "resultsSection.outcomeMeasuresModule",
                "derivedSection.conditionBrowseModule"]


def download_trials(phases=PHASES, force: bool = False) -> Path:
    """Stream every matching study to raw/ctgov_studies.jsonl.gz (no page cap)."""
    from data_sources import record_manifest

    out = RAW_DIR / "ctgov_studies.jsonl.gz"
    if out.exists() and not force:
        log.info(f"Using frozen trial snapshot {out}")
        return out
    params = {
        "query.cond": ONCOLOGY_QUERY,
        "filter.advanced": f"AREA[Phase]({' OR '.join(phases)}) AND AREA[StudyType]INTERVENTIONAL",
        "fields": ",".join(CTGOV_FIELDS),
        "pageSize": 1000,
        "format": "json",
    }
    tmp = out.with_suffix(".part")
    token, n = None, 0
    with gzip.open(tmp, "wt") as fh:
        while True:
            p = dict(params, **({"pageToken": token} if token else {}))
            r = SESSION.get(CTGOV_API, params=p, timeout=180)
            r.raise_for_status()
            data = r.json()
            for study in data.get("studies", []):
                fh.write(json.dumps(study) + "\n")
                n += 1
            token = data.get("nextPageToken")
            log.info(f"CT.gov: {n} studies downloaded")
            if not token:
                break
            time.sleep(0.25)
    tmp.rename(out)
    record_manifest("ctgov_snapshot", CTGOV_API, out,
                    f"snapshot {datetime.now(timezone.utc).date().isoformat()} ({n} studies)")
    return out


def iter_studies(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as fh:
        for line in fh:
            yield json.loads(line)


def _d(struct) -> str | None:
    return (struct or {}).get("date")


def parse_study(s: dict) -> dict:
    ps = s.get("protocolSection", {}) or {}
    idm = ps.get("identificationModule", {}) or {}
    st = ps.get("statusModule", {}) or {}
    dm = ps.get("designModule", {}) or {}
    cm = ps.get("conditionsModule", {}) or {}
    am = ps.get("armsInterventionsModule", {}) or {}
    om = ps.get("outcomesModule", {}) or {}
    em = ps.get("eligibilityModule", {}) or {}
    spm = ps.get("sponsorCollaboratorsModule", {}) or {}
    info = dm.get("designInfo", {}) or {}
    arms = am.get("armGroups", []) or []
    oms = ((s.get("resultsSection") or {}).get("outcomeMeasuresModule") or {}).get("outcomeMeasures", []) or []
    meshes = ((s.get("derivedSection") or {}).get("conditionBrowseModule") or {}).get("meshes", []) or []
    return {
        "nct_id": idm.get("nctId", ""),
        "title": idm.get("officialTitle") or idm.get("briefTitle", ""),
        "phases": dm.get("phases", []) or [],
        "study_type": dm.get("studyType", ""),
        "primary_purpose": info.get("primaryPurpose", ""),
        "status": st.get("overallStatus", ""),
        "start_date": _d(st.get("startDateStruct")),
        "primary_completion_date": _d(st.get("primaryCompletionDateStruct")),
        "completion_date": _d(st.get("completionDateStruct")),
        "results_first_posted": _d(st.get("resultsFirstPostDateStruct")),
        "why_stopped": st.get("whyStopped", "") or "",
        "conditions": cm.get("conditions", []) or [],
        "keywords": cm.get("keywords", []) or [],
        "mesh_conditions": [m.get("term", "") for m in meshes],
        "allocation": info.get("allocation", ""),
        "masking": (info.get("maskingInfo") or {}).get("masking", ""),
        "arm_types": {a.get("label", ""): a.get("type", "") for a in arms},
        "n_arms": len(arms),
        "interventions": [{"name": iv.get("name", ""), "type": iv.get("type", ""),
                           "other_names": iv.get("otherNames", []) or [],
                           "arms": iv.get("armGroupLabels", []) or []}
                          for iv in am.get("interventions", []) or []],
        "primary_outcomes": [o.get("measure", "") for o in om.get("primaryOutcomes", []) or []],
        "eligibility": em.get("eligibilityCriteria", "") or "",
        "sponsor_class": (spm.get("leadSponsor") or {}).get("class", ""),
        "has_results": bool(s.get("hasResults", False)),
        "primary_results": [o for o in oms if o.get("type") == "PRIMARY"],
    }


# ── PubMed ───────────────────────────────────────────────────────────────────
def _ncbi(params: dict) -> dict:
    p = dict(params, tool="mechanistic_trial_engine")
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


def pubmed_count(term: str, before_year: int, since_year: int | None = None) -> int | None:
    """Records matching `term` published in [since_year, before_year-1] (look-ahead free)."""
    params = _ncbi({"db": "pubmed", "term": term, "retmax": 0, "retmode": "json", "datetype": "pdat",
                    "mindate": f"{since_year or 1800}/01/01", "maxdate": f"{before_year - 1}/12/31"})
    data = cached_request("GET", f"{NCBI_EUTILS}/esearch.fcgi", params=params,
                          namespace="pubmed_count", limiter=NCBI_LIMITER)
    try:
        return int(data["esearchresult"]["count"])
    except (TypeError, KeyError, ValueError):
        return None


def pubmed_trial_publications(nct_id: str, max_date: str) -> list[dict]:
    """Publications that cite the NCT number (secondary identifier or text), parsed from XML."""
    params = _ncbi({"db": "pubmed", "term": f"{nct_id}[si] OR {nct_id}[tiab]", "retmax": 20,
                    "retmode": "json", "datetype": "pdat", "mindate": "1990/01/01",
                    "maxdate": max_date.replace("-", "/")})
    search = cached_request("GET", f"{NCBI_EUTILS}/esearch.fcgi", params=params,
                            namespace="pubmed_nct", limiter=NCBI_LIMITER)
    ids = (search or {}).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    xml = cached_request("GET", f"{NCBI_EUTILS}/efetch.fcgi",
                         params=_ncbi({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}),
                         namespace="pubmed_efetch", parse="text", limiter=NCBI_LIMITER)
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    pubs = []
    for art in root.findall(".//PubmedArticle"):
        title_el = art.find(".//ArticleTitle")
        year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        pubs.append({
            "pmid": art.findtext(".//MedlineCitation/PMID"),
            "title": "".join(title_el.itertext()) if title_el is not None else "",
            "year": int(year) if year.isdigit() else None,
            "sections": [(el.get("Label") or el.get("NlmCategory") or "", "".join(el.itertext()))
                         for el in art.findall(".//Abstract/AbstractText")],
            "pub_types": [pt.text or "" for pt in art.findall(".//PublicationTypeList/PublicationType")],
        })
    return pubs


# ── cBioPortal ───────────────────────────────────────────────────────────────
def cbio_entrez(symbols: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in range(0, len(symbols), 500):
        batch = sorted(symbols)[i:i + 500]
        data = cached_request("POST", f"{CBIOPORTAL_API}/genes/fetch",
                              params={"geneIdType": "HUGO_GENE_SYMBOL", "projection": "SUMMARY"},
                              json_body=batch, namespace="cbio_genes")
        for g in data or []:
            out[g["hugoGeneSymbol"]] = int(g["entrezGeneId"])
    return out


def _sample_count(sample_list_id: str) -> int:
    data = cached_request("GET", f"{CBIOPORTAL_API}/sample-lists/{sample_list_id}",
                          params={"projection": "DETAILED"}, namespace="cbio_samplelists")
    if not data:
        return 0
    return int(data.get("sampleCount") or len(data.get("sampleIds", [])))


def cbio_study_alterations(study: str, entrez_ids: list[int]) -> dict | None:
    """Per-gene counts of mutated / amplified / deeply deleted samples in one study."""
    n_seq = _sample_count(f"{study}_sequenced")
    n_cna = _sample_count(f"{study}_cna")
    if n_seq == 0 and n_cna == 0:
        return None
    counts: dict[int, list[int]] = {e: [0, 0, 0] for e in entrez_ids}
    if n_seq:
        muts = cached_request("POST", f"{CBIOPORTAL_API}/molecular-profiles/{study}_mutations/mutations/fetch",
                              params={"projection": "SUMMARY"},
                              json_body={"sampleListId": f"{study}_sequenced", "entrezGeneIds": entrez_ids},
                              namespace="cbio_mut", timeout=180) or []
        seen = {(m["entrezGeneId"], m["sampleId"]) for m in muts}
        for e, _ in seen:
            if e in counts:
                counts[e][0] += 1
    if n_cna:
        cna = cached_request("POST", f"{CBIOPORTAL_API}/molecular-profiles/{study}_gistic/discrete-copy-number/fetch",
                             params={"discreteCopyNumberEventType": "HOMDEL_AND_AMP", "projection": "SUMMARY"},
                             json_body={"sampleListId": f"{study}_cna", "entrezGeneIds": entrez_ids},
                             namespace="cbio_cna", timeout=180) or []
        for c in cna:
            e = c["entrezGeneId"]
            if e in counts:
                if c.get("alteration") == 2:
                    counts[e][1] += 1
                elif c.get("alteration") == -2:
                    counts[e][2] += 1
    return {"n_seq": n_seq, "n_cna": n_cna, "counts": counts}


def cbio_clinical_values(study: str, attribute: str) -> list[float]:
    data = cached_request("GET", f"{CBIOPORTAL_API}/studies/{study}/clinical-data",
                          params={"clinicalDataType": "SAMPLE", "attributeId": attribute,
                                  "projection": "SUMMARY", "pageSize": 100000},
                          namespace="cbio_clinical") or []
    vals = []
    for row in data:
        try:
            vals.append(float(row["value"]))
        except (KeyError, TypeError, ValueError):
            continue
    return vals
