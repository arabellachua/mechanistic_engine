"""Validated, rate-limited PubMed counts with query-specific caches."""

import hashlib
import json
import threading
import time
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from config import CACHE_DIR, CACHE_TTL_DAYS, USE_CACHE, log


URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MechanisticEngine/7.0 (research)"})
_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def _throttle():
    global _LAST_REQUEST
    with _LOCK:
        wait = 0.4 - (time.monotonic() - _LAST_REQUEST)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST = time.monotonic()


def _parse_count(data):
    """Reject error payloads and absent/invalid counts, but accept true zero."""
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError("PubMed error response")
    result = data.get("esearchresult")
    if not isinstance(result, dict) or result.get("ERROR") or result.get("errorlist"):
        raise ValueError("Missing or invalid esearchresult")
    count = result.get("count")
    if isinstance(count, bool) or not str(count).isdigit():
        raise ValueError("Missing or invalid PubMed count")
    return int(count)


def _retry_delay(response, attempt):
    value = response.headers.get("Retry-After") if response is not None else None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            delay = 2 ** attempt
    return max(delay, 0.4)


def _pubmed_count(params, cache_dir, retries=4):
    # A new namespace avoids old caches that may contain a 200/error payload.
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"pubmed_v2_{digest}.json"
    if USE_CACHE and path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL_DAYS * 86400:
        try:
            return _parse_count(json.loads(path.read_text()))
        except (ValueError, OSError):
            pass  # Invalid cache is not evidence of zero publications.
    for attempt in range(retries):
        response = None
        try:
            _throttle()  # Applies to every network attempt, not cache hits.
            response = SESSION.get(URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            count = _parse_count(data)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            return count
        except requests.HTTPError as exc:
            status = exc.response.status_code
            if status != 429 and not 500 <= status < 600:
                log.warning("PubMed HTTP %s; not retrying", status)
                return None
            log.warning("PubMed HTTP %s attempt %s/%s", status, attempt + 1, retries)
        except (requests.RequestException, ValueError, OSError) as exc:
            log.warning("PubMed request/response failure attempt %s/%s: %s", attempt + 1, retries, exc)
        if attempt + 1 < retries:
            time.sleep(_retry_delay(response, attempt))
    return None


def query_pubmed_pair(symbol: str, disease: str, *, as_of=None, cache_dir=None) -> dict:
    """Return all three counts or explicit partial/failed status.

    as_of caps publication date, NOT historical indexing availability. The
    acceleration feature retains its old scaling and is a recent-share proxy.
    """
    end = date.fromisoformat(str(as_of)) if as_of is not None else date.today()
    start = date(end.year - 3, 1, 1)
    query = f"({symbol}[Title/Abstract]) AND ({disease}[Title/Abstract])"
    base = {"db": "pubmed", "term": query, "rettype": "count", "retmode": "json",
            "retmax": 0, "tool": "mech_engine", "datetype": "pdat",
            "mindate": "1800/01/01", "maxdate": end.strftime("%Y/%m/%d")}
    queries = {
        "total": base,
        "clinical_trial": dict(base, term=query + " AND clinical trial[pt]"),
        "recent": dict(base, mindate=start.strftime("%Y/%m/%d")),
    }
    counts = {name: _pubmed_count(params, cache_dir if cache_dir is not None else CACHE_DIR)
              for name, params in queries.items()}
    total, clinical, recent = counts["total"], counts["clinical_trial"], counts["recent"]
    complete = all(value is not None for value in counts.values())
    status = "complete" if complete else ("partial" if any(v is not None for v in counts.values()) else "failed")
    # Cross-query inconsistency must not silently produce a valid feature vector.
    if total is not None and any(v is not None and v > total for v in [clinical, recent]):
        status, complete = "inconsistent", False
    return {
        "pubmed_pair_count": min(total / 1000, 1.0) if total is not None else None,
        "clinical_trial_pub_count": min(clinical / 100, 1.0) if clinical is not None else None,
        "pair_pub_acceleration": (min(recent / max(total * 0.4, 1), 1.0)
                                  if total is not None and recent is not None and status != "inconsistent" else None),
        "has_real_pubmed_data": complete, "pubmed_data_missing": int(not complete),
        "pubmed_status": status, "pubmed_total_raw": total,
        "pubmed_clinical_trial_raw": clinical, "pubmed_recent_raw": recent,
        "pubmed_as_of": end.isoformat(), "pubmed_recent_start": start.isoformat(),
        "pubmed_query": query,
    }
