"""
curation_tools.py — Biology-side curation helpers for the Mechanistic Engine (Arabella).

These tools never read model predictions or automated labels, so outcome review stays
blind to the model (research plan: "judge outcomes from evidence rather than model scores").

Commands (run from the project folder):

  1. Outcome review workbook (Week 1-4)
     python curation_tools.py review --ids NCT00000001,NCT00000002
     python curation_tools.py review --ids-file curation/review_ids.txt
       -> curation/outcome_review.xlsx  (evidence pre-filled; you fill the yellow columns)
       Re-running with new IDs ADDS rows and keeps everything you already filled in.

  2. Extracted-value spot-check sheet (Week 2)
     python curation_tools.py spotcheck --dataset mpe_output/data/analysis_dataset.csv --n 30
       -> curation/spotcheck.xlsx  (one row per trial x extracted field, with the source passage)
     python curation_tools.py spotcheck --dataset ... --ids-file curation/spotcheck_ids.txt
       (use this when Tianze has preselected the sample)

  3. Feature-quality table from a completed spot-check (Week 2 deliverable)
     python curation_tools.py quality
       -> curation/feature_quality_table.csv

  5. Case-study notes for Week 3 (from trials already in the review workbook)
     python curation_tools.py cases --ids NCT00866047,NCT02551432,NCT03878472
       -> curation/cases/<NCT>.md  (pre-filled; complete the TODO sections)

  4. Gold labels from completed reviews (one or two reviewers)
     python curation_tools.py gold --review curation/outcome_review.xlsx
     python curation_tools.py gold --review curation/outcome_review.xlsx --review2 curation/outcome_review_tianze.xlsx
       -> curation/gold_labels_candidate.csv (nct_id,label,source) + agreement report

Needs: pandas, requests, openpyxl (all in Anaconda). Internet access to clinicaltrials.gov
and NCBI. Optional: export NCBI_API_KEY=... for faster PubMed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("curation")
CTGOV = "https://clinicaltrials.gov/api/v2/studies/{}"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_KEY = os.environ.get("NCBI_API_KEY", "")
SEED = 42
NCT_RE = re.compile(r"NCT\d{8}", re.I)

# Columns the reviewer fills in (everything else is pre-filled evidence)
REVIEW_COLS = ["outcome_label", "evidence_source", "evidence_text", "evidence_date",
               "confidence", "reason_if_unknown", "notes", "reviewer", "review_date"]
SPOT_COLS = ["correct", "correct_value", "note"]

LABEL_RULES = [
    ("outcome_label", "1 = met its own pre-specified primary objective; 0 = did not; U = unknown / indeterminate."),
    ("Judge against", "The trial's OWN primary endpoint, analysis and alpha (Phase 2 often uses one-sided 0.10). "
                      "Not secondary endpoints, not later approvals, not what the drug did in other trials."),
    ("Single-arm trials", "1 if the observed primary result met the pre-specified threshold (e.g. ORR >= X% or the "
                          "Simon two-stage rule). If no threshold is stated anywhere, U."),
    ("Randomised trials", "1 if the primary comparison was significant in favour of the experimental arm at the "
                          "trial's alpha. Check arm order: a 'significant' HR > 1 may favour control."),
    ("Terminated / withdrawn", "Futility or lack of efficacy -> 0. Stopped early for efficacy -> 1. Accrual, funding, "
                               "business or sponsor decision -> U. Safety/toxicity stop -> U (note it)."),
    ("Completed, no results, no paper", "U. Absence of a positive statement is not evidence of failure."),
    ("evidence_source", "ctgov_results | publication:PMID | termination_text | abstract (conference) | other:<what>"),
    ("evidence_text", "A short passage (one sentence) that supports the label, copied from the source."),
    ("evidence_date", "Date the evidence first became public (results posted date or publication year). "
                      "Needed for the historical-availability check."),
    ("confidence", "high | medium | low"),
    ("Blinding", "Do not look at model predictions or automated labels while reviewing."),
]

# Extracted fields that can be checked against the CT.gov record -> which passage to show
SPOT_FIELDS = {
    "disease": "conditions_title",
    "investigational_drug": "interventions",
    "drug_targets": "interventions",
    "modality": "interventions",
    "stratum": "interventions",
    "is_combination": "interventions",
    "n_investigational_agents": "interventions",
    "biomarker_selected": "title_inclusion",
    "biomarker_matches_target": "title_inclusion",
    "biomarker_names": "title_inclusion",
    "endpoint_os": "primary_outcomes",
    "endpoint_pfs": "primary_outcomes",
    "endpoint_response": "primary_outcomes",
    "is_randomized": "design",
    "placebo_controlled": "arms",
    "n_arms": "arms",
    "start_year": "dates",
}
BLIND_PREFIXES = ("label", "pred", "prob", "score", "oof", "y_", "target_outcome")
BIOMARKER_HINT = re.compile(
    r"\b(egfr|alk|ros1|ret|ntrk|braf|kras|her-?2|erbb2|c-?met|met\b|pik3ca|fgfr|idh|flt3|brca|hrd|msi|dmmr|"
    r"mismatch repair|tmb|mutational burden|pd-?l1|estrogen|progesterone|hormone receptor|er[+/ -]|androgen|"
    r"cd20|cd19|cd30|cd38|cd33|claudin|cldn18|trop-?2|psma|dll3|triple[\s-]*negative|mutat\w*|fusion|"
    r"rearrang\w*|amplif\w*|overexpress\w*|positive)\b", re.I)


# ── HTTP ─────────────────────────────────────────────────────────────────────
def _get(url: str, params: dict | None = None, as_json: bool = True, tries: int = 4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json() if as_json else r.text
        except requests.RequestException as exc:
            if i == tries - 1:
                print(f"  ! request failed: {url} ({exc})", file=sys.stderr)
                return None
            time.sleep(2 ** i)
    return None


def fetch_study(nct_id: str) -> dict | None:
    return _get(CTGOV.format(nct_id))


def _ncbi(params: dict) -> dict:
    if NCBI_KEY:
        params["api_key"] = NCBI_KEY
    time.sleep(0.12 if NCBI_KEY else 0.35)
    return params


def fetch_pubmed(nct_id: str) -> list[dict]:
    """PubMed records that cite the NCT number. All are returned; the reviewer decides relevance."""
    s = _get(f"{EUTILS}/esearch.fcgi", _ncbi({"db": "pubmed", "term": f"{nct_id}[si] OR {nct_id}[tiab]",
                                              "retmax": 20, "retmode": "json"}))
    ids = ((s or {}).get("esearchresult") or {}).get("idlist", [])
    if not ids:
        return []
    xml = _get(f"{EUTILS}/efetch.fcgi", _ncbi({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}),
               as_json=False)
    return parse_pubmed_xml(xml or "")


def parse_pubmed_xml(xml: str) -> list[dict]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out = []
    for art in root.findall(".//PubmedArticle"):
        t = art.find(".//ArticleTitle")
        year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        secs = [((el.get("Label") or el.get("NlmCategory") or "").upper(), "".join(el.itertext()))
                for el in art.findall(".//Abstract/AbstractText")]
        concl = [txt for lab, txt in secs if "CONCLU" in lab or "INTERPRETATION" in lab]
        if not concl and secs:
            concl = [secs[-1][1]]
        out.append({
            "pmid": art.findtext(".//MedlineCitation/PMID"),
            "year": year,
            "title": "".join(t.itertext()) if t is not None else "",
            "pub_types": "; ".join(pt.text or "" for pt in art.findall(".//PublicationTypeList/PublicationType")),
            "conclusion": " ".join(concl),
        })
    return out


# ── CT.gov record -> readable evidence ───────────────────────────────────────
def _date(struct) -> str:
    return (struct or {}).get("date", "") or ""


def _inclusion(elig: str) -> str:
    m = re.search(r"exclusion criteria", elig or "", re.I)
    return (elig or "")[: m.start()] if m else (elig or "")


def _fmt_measurement(m: dict) -> str:
    v = m.get("value", "")
    lo, hi = m.get("lowerLimit"), m.get("upperLimit")
    return f"{v} ({lo}–{hi})" if lo is not None and hi is not None else str(v)


def summarise_primary_results(oms: list[dict]) -> str:
    """Primary outcome results as text: per-group values and any inferential analyses."""
    lines = []
    for om in oms:
        if om.get("type") != "PRIMARY":
            continue
        groups = {g.get("id"): g.get("title", "") for g in om.get("groups", []) or []}
        head = f"[{om.get('title', '')}] {om.get('paramType', '')} {om.get('unitOfMeasure', '')}".strip()
        lines.append(head)
        for cls in om.get("classes", []) or []:
            for cat in cls.get("categories", []) or []:
                label = " / ".join(x for x in (cls.get("title"), cat.get("title")) if x)
                vals = "; ".join(f"{groups.get(m.get('groupId'), m.get('groupId'))}: {_fmt_measurement(m)}"
                                 for m in cat.get("measurements", []) or [])
                lines.append(f"   {label + ': ' if label else ''}{vals}")
        for a in om.get("analyses", []) or []:
            gnames = " vs ".join(groups.get(g, g) for g in a.get("groupIds", []) or [])
            bits = [f"groups: {gnames}" if gnames else "",
                    f"{a.get('paramType', '')} {a.get('paramValue', '')}".strip(),
                    f"{a.get('ciPctValue', '')}% CI {a.get('ciLowerLimit', '')}–{a.get('ciUpperLimit', '')}"
                    if a.get("ciLowerLimit") is not None else "",
                    f"p {a.get('pValue')}" if a.get("pValue") else "",
                    a.get("statisticalMethod", ""),
                    f"NI: {a.get('nonInferiorityType')}" if a.get("nonInferiorityType") else "",
                    f"comment: {a.get('pValueComment') or a.get('statisticalComment') or ''}".strip()
                    if (a.get("pValueComment") or a.get("statisticalComment")) else ""]
            lines.append("   ANALYSIS: " + " | ".join(b for b in bits if b))
    return "\n".join(lines)


def summarise_study(s: dict) -> dict:
    ps = s.get("protocolSection", {}) or {}
    idm, st = ps.get("identificationModule", {}) or {}, ps.get("statusModule", {}) or {}
    dm, cm = ps.get("designModule", {}) or {}, ps.get("conditionsModule", {}) or {}
    am, om = ps.get("armsInterventionsModule", {}) or {}, ps.get("outcomesModule", {}) or {}
    em, rm = ps.get("eligibilityModule", {}) or {}, ps.get("referencesModule", {}) or {}
    info = dm.get("designInfo", {}) or {}
    oms = ((s.get("resultsSection") or {}).get("outcomeMeasuresModule") or {}).get("outcomeMeasures", []) or []
    nct = idm.get("nctId", "")
    arms = am.get("armGroups", []) or []
    return {
        "nct_id": nct,
        "title": idm.get("officialTitle") or idm.get("briefTitle", ""),
        "phase": ", ".join(dm.get("phases", []) or []),
        "status": st.get("overallStatus", ""),
        "why_stopped": st.get("whyStopped", "") or "",
        "start_date": _date(st.get("startDateStruct")),
        "primary_completion_date": _date(st.get("primaryCompletionDateStruct")),
        "results_first_posted": _date(st.get("resultsFirstPostDateStruct")),
        "conditions": "; ".join(cm.get("conditions", []) or []),
        "allocation": info.get("allocation", ""),
        "arms": "\n".join(f"{a.get('label', '')} [{a.get('type', '')}]" for a in arms),
        "n_arms_record": len(arms),
        "interventions": "\n".join(
            f"{iv.get('name', '')} [{iv.get('type', '')}]"
            + (f" (aka {', '.join(iv.get('otherNames') or [])})" if iv.get("otherNames") else "")
            + (f" -> {', '.join(iv.get('armGroupLabels') or [])}" if iv.get("armGroupLabels") else "")
            for iv in am.get("interventions", []) or []),
        "primary_outcomes": "\n".join(
            f"{o.get('measure', '')} (time frame: {o.get('timeFrame', '')})"
            + (f" — {o.get('description', '')}" if o.get("description") else "")
            for o in om.get("primaryOutcomes", []) or []),
        "has_results": bool(s.get("hasResults", False)),
        "primary_results": summarise_primary_results(oms),
        "result_refs": "; ".join(r.get("pmid", "") for r in rm.get("references", []) or []
                                 if r.get("type") == "RESULT" and r.get("pmid")),
        "inclusion_criteria": _inclusion(em.get("eligibilityCriteria", "") or "").strip(),
        "ctgov_url": f"https://clinicaltrials.gov/study/{nct}",
        "history_url": f"https://clinicaltrials.gov/study/{nct}?tab=history",
        "scope_flag": scope_flag(dm.get("phases", []) or [], st.get("overallStatus", ""),
                                 _date(st.get("startDateStruct")), om.get("primaryOutcomes", []) or []),
        "threshold_hints": threshold_hints(
            " ".join([(ps.get("descriptionModule") or {}).get("detailedDescription", "") or "",
                      (ps.get("descriptionModule") or {}).get("briefSummary", "") or ""]
                     + [o.get("description", "") or "" for o in om.get("primaryOutcomes", []) or []]
                     + [a.get("statisticalComment", "") or "" for m in oms for a in m.get("analyses", []) or []])),
        "protocol_sap_docs": doc_links(nct, s),
    }


# ── Scope, threshold hints, documents ───────────────────────────────────────
# Mirrors v9 config: PHASES = PHASE2/PHASE3, EXCLUDE_PHASE1_2, TRAIN_START_YEAR = 2005.
# Keep in sync with config.py if Tianze changes the scope.
MIN_START_YEAR = 2005
_SAFETY_RE = re.compile(r"adverse event|toxicit|safety|tolerab|maximum tolerated|\bmtd\b|dose[\s-]*limiting|"
                        r"\bdlt\b|pharmacokinetic|recommended phase (?:2|ii) dose|\brp2d\b", re.I)
_EFFICACY_RE = re.compile(r"respon|remission|surviv|progression|\bpfs\b|\bos\b|\borr\b|recurrence|relapse|"
                          r"\bpcr\b|pathologic|disease control|time to|event[\s-]free|biochemical|failure", re.I)
_THRESH_RE = re.compile(r"null hypothes|alternative hypothes|\bh[01]\s*[:=]|simon|two[\s-]*stage|historical|"
                        r"power|alpha|type i error|target|promising|uninteresting|"
                        r"\d+(?:\.\d+)?\s*%\s*(?:to|vs\.?|versus)\s*\d+(?:\.\d+)?\s*%|"
                        r"(?:at least|≥|>=)\s*\d+\s*(?:of|/)\s*\d+\s*(?:patients|responses|responders)", re.I)


def scope_flag(phases: list[str], status: str, start: str, primary_outcomes: list[dict]) -> str:
    """Why a trial is outside the v9 label_primary scope; '' if in scope."""
    reasons = []
    if "PHASE1" in phases:
        reasons.append("Phase 1/2 (excluded in v9)")
    if phases and not any(p in ("PHASE2", "PHASE3") for p in phases):
        reasons.append(f"phase {', '.join(phases)}")
    if status == "WITHDRAWN":
        reasons.append("withdrawn (never enrolled)")
    year = int(start[:4]) if start[:4].isdigit() else None
    if year is not None and year < MIN_START_YEAR:
        reasons.append(f"start {year} < {MIN_START_YEAR}")
    text = " ".join(f"{o.get('measure', '')} {o.get('description', '')}" for o in primary_outcomes)
    measures = " ".join(o.get("measure", "") for o in primary_outcomes)
    if primary_outcomes and _SAFETY_RE.search(measures) and not _EFFICACY_RE.search(measures):
        reasons.append("primary endpoint is safety/dose-finding only")
    elif not text.strip():
        reasons.append("no primary outcome registered")
    return "EXCLUDE? " + "; ".join(reasons) if reasons else ""


def threshold_hints(text: str, k: int = 4) -> str:
    """Sentences that may state the pre-specified success threshold. Hints only: verify in the source."""
    sents = re.split(r"(?<=[.;])\s+|\n+", text or "")
    hits = [s.strip() for s in sents if _THRESH_RE.search(s) and len(s.strip()) > 15]
    return "\n".join(dict.fromkeys(h[:400] for h in hits[:k]))


def doc_links(nct: str, s: dict) -> str:
    """Links to protocol / SAP PDFs posted on CT.gov — often where the null/target rate is stated."""
    docs = (((s.get("documentSection") or {}).get("largeDocumentModule") or {}).get("largeDocs") or [])
    out = []
    for d in docs:
        kind = "+".join(k for k, f in (("protocol", "hasProtocol"), ("SAP", "hasSap"), ("ICF", "hasIcf"))
                        if d.get(f)) or d.get("typeAbbrev", "doc")
        if d.get("filename"):
            out.append(f"{kind}: https://cdn.clinicaltrials.gov/large-docs/{nct[-2:]}/{nct}/{d['filename']}")
    return "\n".join(out)


# ── Excel helpers ────────────────────────────────────────────────────────────
def _clip(v, n: int = 32000):
    return v[:n] if isinstance(v, str) else v


def _write_xlsx(path: Path, sheets: dict[str, pd.DataFrame], highlight: dict[str, list[str]]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    yellow = PatternFill("solid", fgColor="FFF2CC")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.apply(lambda col: col.map(_clip)).to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "B2"
            for j, col in enumerate(df.columns, start=1):
                letter = ws.cell(row=1, column=j).column_letter
                long = col in ("primary_results", "inclusion_criteria", "source_passage", "conclusion",
                               "interventions", "primary_outcomes", "evidence_text", "rule", "arms", "title")
                ws.column_dimensions[letter].width = 60 if long else 18
                ws.cell(row=1, column=j).font = Font(bold=True)
                if col in highlight.get(name, []):
                    for i in range(1, len(df) + 2):
                        ws.cell(row=i, column=j).fill = yellow
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")


def _read_ids(ids: str | None, ids_file: str | None) -> list[str]:
    text = (ids or "") + "\n" + (Path(ids_file).read_text() if ids_file else "")
    seen, out = set(), []
    for m in NCT_RE.findall(text):
        m = m.upper()
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ── 1. Outcome review ────────────────────────────────────────────────────────
def cmd_review(args) -> None:
    out = Path(args.out)
    ids = _read_ids(args.ids, args.ids_file)
    if not ids:
        sys.exit("No NCT IDs given. Use --ids NCT...,NCT... or --ids-file file.txt")
    old_rev = old_pub = None
    if out.exists():
        old_rev = pd.read_excel(out, sheet_name="review", dtype=str).fillna("")
        try:
            old_pub = pd.read_excel(out, sheet_name="publications", dtype=str).fillna("")
        except ValueError:
            old_pub = None
        done = set(old_rev["nct_id"])
        ids = [i for i in ids if i not in done]
        print(f"{out} exists: keeping {len(done)} reviewed rows, adding {len(ids)} new trials.")
    rows, pubs = [], []
    for k, nct in enumerate(ids, 1):
        print(f"[{k}/{len(ids)}] {nct}")
        s = fetch_study(nct)
        if s is None:
            rows.append({"nct_id": nct, "title": "NOT FOUND ON CT.GOV"})
            continue
        r = summarise_study(s)
        papers = [] if args.no_pubmed else fetch_pubmed(nct)
        for p in papers:
            pubs.append({"nct_id": nct, **p, "is_result_ref": p["pmid"] in r["result_refs"].split("; "),
                         "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/"})
        r["n_pubmed_citing"] = len(papers)
        rows.append(r)
    new = pd.DataFrame(rows)
    for c in REVIEW_COLS:
        new[c] = ""
    rev = pd.concat([old_rev, new], ignore_index=True) if old_rev is not None else new
    lead = ["nct_id", "scope_flag"] + REVIEW_COLS + ["threshold_hints", "protocol_sap_docs"]
    for c in lead:
        if c not in rev.columns:
            rev[c] = ""
    rev = rev[lead + [c for c in rev.columns if c not in lead]]
    n_flag = (rev["scope_flag"].fillna("") != "").sum()
    if n_flag:
        print(f"  {n_flag} trial(s) flagged as possibly out of scope (see scope_flag). Confirm and replace them.")
    pub = pd.DataFrame(pubs)
    if old_pub is not None and len(old_pub):
        pub = pd.concat([old_pub, pub], ignore_index=True)
    if pub.empty:
        pub = pd.DataFrame(columns=["nct_id", "pmid", "year", "title", "pub_types", "conclusion",
                                    "is_result_ref", "pubmed_url"])
    rules = pd.DataFrame(LABEL_RULES, columns=["item", "rule"])
    _write_xlsx(out, {"review": rev, "publications": pub, "rules": rules}, {"review": REVIEW_COLS})
    print(f"Wrote {out}: {len(rev)} trials, {len(pub)} linked publications. Fill the yellow columns.")


# ── 2. Spot-check of extracted values ────────────────────────────────────────
def _load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    blind = [c for c in df.columns if c.lower().startswith(BLIND_PREFIXES)]
    return df.drop(columns=blind)


def _passage(kind: str, r: dict) -> str:
    if kind == "conditions_title":
        return f"Conditions: {r['conditions']}\nTitle: {r['title']}"
    if kind == "interventions":
        return f"Interventions:\n{r['interventions']}\nArms:\n{r['arms']}"
    if kind == "title_inclusion":
        lines = [ln.strip() for ln in r["inclusion_criteria"].splitlines() if BIOMARKER_HINT.search(ln)]
        body = "\n".join(lines) if lines else "(no biomarker terms found in inclusion criteria)"
        return f"Title: {r['title']}\nInclusion lines with biomarker terms:\n{body}"
    if kind == "primary_outcomes":
        return r["primary_outcomes"]
    if kind == "design":
        return f"Allocation: {r['allocation']}\nArms:\n{r['arms']}"
    if kind == "arms":
        return f"{r['n_arms_record']} arms:\n{r['arms']}"
    if kind == "dates":
        return f"Start date: {r['start_date']}"
    return ""


def cmd_spotcheck(args) -> None:
    df = _load_dataset(args.dataset)
    if "nct_id" not in df.columns:
        sys.exit("Dataset has no nct_id column.")
    fields = [f for f in SPOT_FIELDS if f in df.columns] + \
             [c for c in df.columns if c.startswith("biomarker_") and c not in SPOT_FIELDS]
    if not fields:
        sys.exit("None of the checkable fields are in this dataset. Columns: " + ", ".join(df.columns[:40]))
    ids = _read_ids(None, args.ids_file) if args.ids_file else None
    if ids:
        sample = df[df["nct_id"].isin(ids)]
        missing = set(ids) - set(sample["nct_id"])
        if missing:
            print(f"  ! not in dataset: {', '.join(sorted(missing))}")
    else:  # stratified by stratum so every therapy class is represented
        g = df.groupby("stratum", group_keys=False) if "stratum" in df.columns else None
        if g is not None and df["stratum"].nunique() > 1:
            per = max(1, -(-args.n // df["stratum"].nunique()))  # ceiling
            sample = g.apply(lambda x: x.sample(min(len(x), per), random_state=SEED))
        else:
            sample = df.sample(min(len(df), args.n), random_state=SEED)
    sample = sample.drop_duplicates("nct_id")
    rows = []
    for k, (_, row) in enumerate(sample.iterrows(), 1):
        nct = row["nct_id"]
        print(f"[{k}/{len(sample)}] {nct}")
        s = fetch_study(nct)
        rec = summarise_study(s) if s else None
        for f in fields:
            kind = SPOT_FIELDS.get(f, "title_inclusion")
            rows.append({"nct_id": nct, "field": f, "extracted_value": row.get(f, ""),
                         **{c: "" for c in SPOT_COLS},
                         "source_passage": _passage(kind, rec) if rec else "NOT FOUND ON CT.GOV",
                         "ctgov_url": f"https://clinicaltrials.gov/study/{nct}"})
    out = Path(args.out)
    sheet = pd.DataFrame(rows)
    sheet = sheet[["nct_id", "field", "extracted_value"] + SPOT_COLS + ["source_passage", "ctgov_url"]]
    guide = pd.DataFrame([
        ("correct", "Y = extracted value matches the source passage; N = wrong; ? = source is ambiguous."),
        ("correct_value", "If N, what the value should be."),
        ("note", "Why it is wrong (e.g. negation missed, biomarker in exclusion criteria, backbone counted as "
                 "investigational, wrong cancer type mapped)."),
        ("Scope", "Check only against the passage and the CT.gov record. Do not look at labels or predictions."),
    ], columns=["item", "rule"])
    sample_ids = pd.DataFrame({"nct_id": sample["nct_id"].tolist()})
    _write_xlsx(out, {"spotcheck": sheet, "guide": guide, "sample_ids": sample_ids}, {"spotcheck": SPOT_COLS})
    print(f"Wrote {out}: {len(sample)} trials x {len(fields)} fields = {len(sheet)} checks.")


# ── 3. Feature-quality table ─────────────────────────────────────────────────
def cmd_quality(args) -> None:
    sc = pd.read_excel(args.spotcheck, sheet_name="spotcheck", dtype=str).fillna("")
    sc["correct"] = sc["correct"].str.strip().str.upper().str[:1]
    rows = []
    for f, g in sc.groupby("field", sort=False):
        y, n, q = (g["correct"] == "Y").sum(), (g["correct"] == "N").sum(), (g["correct"] == "?").sum()
        checked = y + n
        errs = g[g["correct"] == "N"]
        rows.append({
            "field": f, "n_trials": len(g), "n_checked": int(checked), "n_correct": int(y), "n_wrong": int(n),
            "n_ambiguous": int(q), "n_not_reviewed": int(len(g) - y - n - q),
            "accuracy": round(y / checked, 3) if checked else None,
            "error_examples": " | ".join(f"{r.nct_id}: {r.extracted_value} -> {r.correct_value} ({r.note})"
                                         for r in errs.head(3).itertuples()),
        })
    out = OUT_DIR / "feature_quality_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows)[["field", "n_checked", "n_wrong", "n_ambiguous", "accuracy"]].to_string(index=False))
    print(f"\nWrote {out}")


# ── 4. Gold labels ───────────────────────────────────────────────────────────
def _labels(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="review", dtype=str).fillna("")
    df["outcome_label"] = df["outcome_label"].str.strip().str.upper()
    return df[["nct_id", "outcome_label", "evidence_source", "confidence"]]


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def cmd_gold(args) -> None:
    a = _labels(args.review)
    a = a[a["outcome_label"].isin(["0", "1", "U"])]
    if args.review2:
        b = _labels(args.review2)
        m = a.merge(b, on="nct_id", suffixes=("_1", "_2"))
        m = m[m["outcome_label_2"].isin(["0", "1", "U"])]
        agree = m["outcome_label_1"] == m["outcome_label_2"]
        k = cohen_kappa(m["outcome_label_1"].tolist(), m["outcome_label_2"].tolist())
        print(f"Double-reviewed: {len(m)} | agreement {agree.mean():.2f} | Cohen's kappa {k}")
        m[~agree].to_csv(OUT_DIR / "label_disagreements_to_adjudicate.csv", index=False)
        print(f"Disagreements -> {OUT_DIR / 'label_disagreements_to_adjudicate.csv'} ({(~agree).sum()})")
        gold = m[agree & m["outcome_label_1"].isin(["0", "1"])]
        gold = pd.DataFrame({"nct_id": gold["nct_id"], "label": gold["outcome_label_1"].astype(int),
                             "source": "dual_review:" + gold["evidence_source_1"]})
        pd.DataFrame([{"n_double_reviewed": len(m), "percent_agreement": round(agree.mean(), 3),
                       "cohen_kappa": k, "n_disagreements": int((~agree).sum()),
                       "date": date.today().isoformat()}]).to_csv(OUT_DIR / "label_agreement_manual.csv", index=False)
    else:
        det = a[a["outcome_label"].isin(["0", "1"])]
        gold = pd.DataFrame({"nct_id": det["nct_id"], "label": det["outcome_label"].astype(int),
                             "source": "single_review:" + det["evidence_source"]})
        print("Single reviewer: these are candidate labels, not gold, until a second review agrees.")
    print(f"Unknown (U) labels: {(a['outcome_label'] == 'U').sum()}  — report these; do not force them.")
    out = OUT_DIR / "gold_labels_candidate.csv"
    gold.to_csv(out, index=False)
    print(f"Wrote {out} ({len(gold)} labels). After adjudication, copy to <MPE_OUTPUT>/data/gold_labels.csv.")


# ── 5. Case-study notes (Week 3) ─────────────────────────────────────────────
CASE_TEMPLATE = """# Case: {nct_id}

**Title:** {title}
**Phase / design:** {phase} · {allocation} · {n_arms_record} arm(s) · start {start_date} · status {status}
**Record:** {ctgov_url}  ·  history: {history_url}

## 1. Trial and biology
- Cancer type: {conditions}
- Investigational agent(s):
{interventions_md}
- Target / mechanism: _TODO: target gene(s), mechanism class, modality_
- Biomarker-defined enrollment: _TODO: yes/no — quote the inclusion line_

## 2. Outcome (checked against source)
- Primary endpoint: {primary_outcomes}
- Pre-specified threshold: _TODO: quote it, or state that none exists_
- Label: **{outcome_label}** ({confidence} confidence) — source: {evidence_source} ({evidence_date})
- Evidence: "{evidence_text}"
- Uncertainty: {reason_if_unknown}

## 3. What the biology features say
_TODO after Tianze shares feature values: 3–4 feature values for this trial, compared with the cohort median.
Describe what each measures (see week2_biology_notes.md §1) — not what it "causes"._

## 4. Why this case is informative
_TODO: e.g. strong biology but failed; weak biology but succeeded; uncertain label._

## 5. Alternative explanations
- Trial design (dose, comparator, endpoint choice, sample size): _TODO_
- Patient selection / biomarker: _TODO_
- Missingness or mapping errors in the features: _TODO_
- Disease or drug-class effect rather than target biology: _TODO_

## 6. Notes
{notes}
"""


def cmd_cases(args) -> None:
    rev = pd.read_excel(args.review, sheet_name="review", dtype=str).fillna("")
    ids = _read_ids(args.ids, None)
    missing = [i for i in ids if i not in set(rev["nct_id"])]
    if missing:
        sys.exit(f"Not in {args.review}: {', '.join(missing)} — run `review` for them first.")
    out_dir = OUT_DIR / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, r in rev[rev["nct_id"].isin(ids)].iterrows():
        d = {k: (v or "—") for k, v in r.to_dict().items()}
        d["interventions_md"] = "\n".join(f"  - {ln}" for ln in str(r.get("interventions", "")).splitlines()) or "  - —"
        for k in ("title", "phase", "allocation", "n_arms_record", "start_date", "status", "conditions",
                  "primary_outcomes", "ctgov_url", "history_url", "outcome_label", "confidence", "evidence_source",
                  "evidence_date", "evidence_text", "reason_if_unknown", "notes"):
            d.setdefault(k, "—")
        path = out_dir / f"{r['nct_id']}.md"
        if path.exists() and not args.overwrite:
            print(f"  skip {path} (exists; use --overwrite)")
            continue
        path.write_text(CASE_TEMPLATE.format(**d))
        print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review", help="Build/extend the blinded outcome-review workbook")
    r.add_argument("--ids")
    r.add_argument("--ids-file")
    r.add_argument("--out", default=str(OUT_DIR / "outcome_review.xlsx"))
    r.add_argument("--no-pubmed", action="store_true")
    s = sub.add_parser("spotcheck", help="Build the extracted-value spot-check sheet")
    s.add_argument("--dataset", required=True)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--ids-file")
    s.add_argument("--out", default=str(OUT_DIR / "spotcheck.xlsx"))
    q = sub.add_parser("quality", help="Summarise a completed spot-check into the feature-quality table")
    q.add_argument("--spotcheck", default=str(OUT_DIR / "spotcheck.xlsx"))
    g = sub.add_parser("gold", help="Export candidate gold labels (+ kappa if two reviewers)")
    g.add_argument("--review", default=str(OUT_DIR / "outcome_review.xlsx"))
    g.add_argument("--review2")
    c = sub.add_parser("cases", help="Write Week 3 case-study notes from reviewed trials")
    c.add_argument("--ids", required=True)
    c.add_argument("--review", default=str(OUT_DIR / "outcome_review.xlsx"))
    c.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    {"review": cmd_review, "spotcheck": cmd_spotcheck, "quality": cmd_quality, "gold": cmd_gold,
     "cases": cmd_cases}[args.cmd](args)


if __name__ == "__main__":
    main()
    