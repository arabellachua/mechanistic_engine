"""
labels.py — Trial-specific outcome labels with provenance and confidence. (Issue 1)
Merges v7 labels.py and v8 labels_v8.py.

What changed and why
  * REMOVED: "approved drug in approved indication => positive". Approval is a
    drug-level fact that is also reflected in knowledge-type features, so the model
    learnt "famous drug => success". Approval status is now only a reporting variable.
  * REMOVED: text triggers applied to brief_summary / primary_outcome. Those fields are
    written at registration and describe the PLAN ("objective response rate will be
    assessed"), so "objective response" fired as a positive outcome signal.
  * REMOVED: "completed + results posted + no positive text => negative". Absence of an
    outcome sentence in a registry record is not evidence of failure; it is indeterminate.
  * Evidence tiers, each evaluated independently so tier agreement can be measured:
        gold (dual manual curation)  >  structured_results  >  publication  >  termination_text
  * Structured results: first primary outcome with an inferential analysis; the alpha
    is parsed from the analysis comment (Phase 2 designs often use one-sided 0.10), and
    significant hazard ratios > 1 are flagged indeterminate because CT.gov does not fix
    the arm order.
  * Publications: only the CONCLUSIONS section, negation-aware patterns, protocol and
    design papers excluded, efetch XML actually parsed.
  * Termination: futility/efficacy phrases only; accrual, business and (by default)
    safety stops are indeterminate; "stopped early for efficacy" is positive.
  * label_transition: Phase 3 of the same drug in the same cancer type started within N
    years of Phase 2 primary completion; censored when follow-up is too short.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (COUNT_SAFETY_STOP_AS_FAILURE, DATA_FREEZE_DATE, DEFAULT_ALPHA, GOLD_LABELS_CSV,
                    LABEL_CONFIDENCE, PUBLICATION_FETCH, TRANSITION_WINDOW_YEARS, log)

# ── Structured results ───────────────────────────────────────────────────────
_P_RE = re.compile(r"^\s*(<=|>=|<|>|≤|≥|=)?\s*(\d*\.?\d+(?:\s*[eE]\s*-?\s*\d+)?)")
_ALPHA_RE = re.compile(r"(?:alpha|α|significance level|type[\s-]*i error(?: rate)?|level of significance|"
                       r"significant at)\D{0,25}?(0?\.\d+|\d+(?:\.\d+)?\s*%)", re.I)


def parse_p_value(raw) -> tuple[float | None, str | None]:
    if raw is None:
        return None, None
    s = str(raw).strip().replace("×10^", "e").replace("x10^", "e").replace("x 10^", "e")
    m = _P_RE.match(s)
    if not m:
        return None, None
    rel = (m.group(1) or "=").replace("≤", "<=").replace("≥", ">=")
    try:
        val = float(m.group(2).replace(" ", ""))
    except ValueError:
        return None, None
    return (val, rel) if 0.0 <= val <= 1.0 else (None, None)


def parse_alpha(text: str) -> float | None:
    m = _ALPHA_RE.search(text or "")
    if not m:
        return None
    v = m.group(1)
    val = float(v.replace("%", "").strip()) / 100 if "%" in v else float(v)
    return val if 0 < val <= 0.3 else None


def is_significant(p: float, rel: str, alpha: float) -> bool | None:
    if rel in ("<", "<="):
        return True if p <= alpha else None     # "<0.10" with alpha 0.05 is undeterminable
    if rel in (">", ">="):
        return False if p >= alpha else None
    return p < alpha


def label_from_structured(primary_results: list[dict], default_alpha: float = DEFAULT_ALPHA) -> tuple[int, str]:
    for om in primary_results or []:
        analyses = [a for a in (om.get("analyses") or []) if a.get("pValue")]
        if not analyses:
            continue
        an = analyses[0]                       # pre-specified rule: first analysis of first primary outcome
        p, rel = parse_p_value(an.get("pValue"))
        if p is None:
            continue
        comment = f"{an.get('pValueComment', '')} {an.get('statisticalComment', '')} {an.get('estimateComment', '')}"
        alpha = parse_alpha(comment) or default_alpha
        sig = is_significant(p, rel, alpha)
        if sig is None:
            return -1, f"p{rel}{p} vs alpha {alpha}: undeterminable"
        ptype = (an.get("paramType") or "").lower()
        try:
            est = float(an.get("paramValue"))
        except (TypeError, ValueError):
            est = None
        if sig and "hazard ratio" in ptype and est is not None and est > 1:
            return -1, f"significant HR={est}>1 (arm order ambiguous)"
        n_prim = sum(1 for o in primary_results if o.get("analyses"))
        return int(sig), f"p{rel}{p} alpha={alpha} n_primary_with_analyses={n_prim}"
    return -1, "no primary inferential analysis"


# ── Publications ─────────────────────────────────────────────────────────────
NEG_PUB = [re.compile(p) for p in (
    r"did not (meet|reach|achieve) (its |the )?(pre-?specified |primary )*(end ?point|objective|outcome)",
    r"primary (end ?point|objective|outcome) (was|were) not (met|reached|achieved)",
    r"failed to (meet|reach|show|demonstrate|improve|achieve|prolong)",
    r"\bno (statistically )?significant (difference|improvement|benefit|prolongation|increase)",
    r"\bnot (statistically )?significant(ly)?( different| improv| prolong| better)",
    r"did not (significantly )?(improve|prolong|increase|show|demonstrate|result in)",
    r"(does|did) not (support|warrant) further", r"not warrant(ed)? further",
    r"(insufficient|minimal|limited|no meaningful|negligible) (clinical |anti-?tumou?r )?(activity|efficacy|benefit)",
    r"\bfutility\b", r"no (clinical|survival|additional) benefit", r"lack of (efficacy|activity|benefit)",
    r"(was|is) (not effective|ineffective)",
)]
POS_PUB = [re.compile(p) for p in (
    r"(met|achieved|reached) (its |the )(pre-?specified )?primary (end ?point|objective)",
    r"primary (end ?point|objective) (was|were) (met|achieved|reached)",
    r"significant(ly)? (improv|prolong|increas|longer|better|superior|higher)",
    r"warrants? further (clinical )?(investigation|evaluation|study|studies|development|testing)",
    r"phase (iii|3) (trial|study|evaluation|investigation)s? (is |are )?(warranted|planned|ongoing|underway)",
    r"(promising|encouraging|meaningful|substantial|durable) (anti-?tumou?r |clinical )?(activity|efficacy|responses?|benefit)",
    r"\b(superior|non-?inferior) to\b",
)]
NON_RESULT_TITLE = re.compile(r"design|rationale|protocol|trial in progress|study in progress|methodolog", re.I)


def classify_abstract(sections: list[tuple[str, str]]) -> tuple[int, str]:
    concl = " ".join(t for lab, t in sections if re.search(r"conclu|interpret", lab or "", re.I))
    if not concl:
        full = " ".join(t for _, t in sections)
        sentences = re.split(r"(?<=[.!?])\s+", full)
        concl = " ".join(sentences[-2:])
    text = concl.lower()
    neg = [p.pattern for p in NEG_PUB if p.search(text)]
    if neg:
        return 0, f"neg:{neg[0][:40]}"
    pos = [p.pattern for p in POS_PUB if p.search(text)]
    if pos:
        return 1, f"pos:{pos[0][:40]}"
    return -1, "no decisive conclusion"


def label_from_publications(pubs: list[dict], start_year: float) -> tuple[int, str, str | None]:
    candidates = [p for p in pubs
                  if p.get("year") and (np.isnan(start_year) or p["year"] >= start_year + 1)
                  and not NON_RESULT_TITLE.search(p.get("title", ""))
                  and not any("protocol" in (t or "").lower() for t in p.get("pub_types", []))
                  and p.get("sections")]
    for pub in sorted(candidates, key=lambda p: p["year"]):
        lab, ev = classify_abstract(pub["sections"])
        if lab != -1:
            return lab, ev, pub.get("pmid")
    return -1, "no results publication", None


# ── Termination text ─────────────────────────────────────────────────────────
TERM_NEG = re.compile(r"futil|lack of (efficacy|activity|response|benefit|effect)|(insufficient|inadequate|poor|low|"
                      r"limited|no|lower than expected|disappointing) (efficacy|activity|response|benefit|clinical benefit)|"
                      r"did not (meet|reach|show|demonstrate)|failed to (meet|show|demonstrate|reach)|"
                      r"unlikely to (meet|achieve|demonstrate|show)|interim analysis[^.]{0,60}(futil|no benefit|not met|"
                      r"insufficient|unlikely)|negative (results|outcome)", re.I)
TERM_POS = re.compile(r"(early|stopped|terminated)[^.]{0,30}(for|due to) (efficacy|benefit|positive)|"
                      r"overwhelming efficacy|primary endpoint (was )?(met|achieved)", re.I)
TERM_SAFETY = re.compile(r"toxicit|safety|adverse|side effect", re.I)


def label_from_termination(status: str, why: str) -> tuple[int, str]:
    if status != "TERMINATED" or not why:
        return -1, "not terminated / no reason"
    if TERM_POS.search(why):
        return 1, "stopped early for efficacy"
    if TERM_NEG.search(why):
        return 0, "terminated for futility/lack of efficacy"
    if COUNT_SAFETY_STOP_AS_FAILURE and TERM_SAFETY.search(why):
        return 0, "terminated for safety"
    return -1, "terminated for non-efficacy reason"


# ── Assembly ─────────────────────────────────────────────────────────────────
def load_gold() -> dict[str, dict]:
    if not GOLD_LABELS_CSV.exists():
        return {}
    g = pd.read_csv(GOLD_LABELS_CSV)
    g = g[g["label"].isin([0, 1])]
    log.info(f"Gold labels loaded: {len(g)}")
    return {r.nct_id: {"label": int(r.label), "source": getattr(r, "source", "gold")} for r in g.itertuples()}


def assign_label(row, pubs: list[dict] | None, gold: dict) -> dict:
    sl, sev = label_from_structured(row.get("primary_results") or [])
    pl, pev, pmid = label_from_publications(pubs or [], row.get("start_year", np.nan))
    tl, tev = label_from_termination(row.get("status", ""), row.get("why_stopped", ""))
    out = {"label_structured": sl, "label_publication": pl, "label_termination": tl,
           "evidence_structured": sev, "evidence_publication": pev, "evidence_termination": tev,
           "publication_pmid": pmid}
    g = gold.get(row["nct_id"])
    if g is not None:
        label, tier, evidence = g["label"], "gold", f"gold:{g['source']}"
    elif sl != -1:
        label, tier, evidence = sl, "structured_results", sev
    elif pl != -1:
        label, tier, evidence = pl, "publication", pev
    elif tl != -1:
        label, tier, evidence = tl, "termination_text", tev
    else:
        label, tier, evidence = -1, "indeterminate", "no trial-specific outcome evidence"
    out.update({"label_primary": label, "label_tier": tier, "label_evidence": evidence,
                "label_confidence": LABEL_CONFIDENCE.get(tier, 0.0)})
    return out


LABELABLE_STATUSES = {"COMPLETED", "TERMINATED", "ACTIVE_NOT_RECRUITING", "WITHDRAWN"}


def build_labels(df: pd.DataFrame, fetch_publications: bool = True) -> pd.DataFrame:
    gold = load_gold()
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labels"):
        pubs = None
        if fetch_publications and row["status"] in LABELABLE_STATUSES:
            need = PUBLICATION_FETCH == "all" or label_from_structured(row.get("primary_results") or [])[0] == -1
            if need:
                from clients import pubmed_trial_publications
                pubs = pubmed_trial_publications(row["nct_id"], DATA_FREEZE_DATE)
        records.append(assign_label(row, pubs, gold))
    out = pd.concat([df, pd.DataFrame(records, index=df.index)], axis=1)
    counts = out.groupby("label_tier")["label_primary"].value_counts().unstack(fill_value=0)
    log.info(f"Label tiers:\n{counts}")
    return out


def add_transition_labels(df: pd.DataFrame, window: float = TRANSITION_WINDOW_YEARS,
                          freeze: str = DATA_FREEZE_DATE) -> pd.DataFrame:
    """Phase 2 -> Phase 3 transition of the same investigational drug in the same cancer type."""
    df = df.copy()
    freeze_ts = pd.Timestamp(freeze)
    p3 = df[(df["phase_group"] == "PHASE3") & df["investigational_drug"].notna()]
    first_p3 = p3.groupby(["investigational_drug", "disease"])["start_date_dt"].min().to_dict()
    labels = []
    for r in df.itertuples():
        if r.phase_group != "PHASE2" or r.investigational_drug is None or pd.isna(r.investigational_drug):
            labels.append(-1)
            continue
        pc = r.primary_completion_date_dt if pd.notna(r.primary_completion_date_dt) else r.completion_date_dt
        if pd.isna(pc) or r.disease in ("cancer_nos", "solid_tumors", "basket"):
            labels.append(-1)
            continue
        deadline = pc + pd.DateOffset(years=int(window))
        p3_start = first_p3.get((r.investigational_drug, r.disease))
        if p3_start is not None and pd.notna(r.start_date_dt) and r.start_date_dt <= p3_start <= deadline:
            labels.append(1)
        elif deadline <= freeze_ts:
            labels.append(0)
        else:
            labels.append(-1)      # censored: not enough follow-up
    df["label_transition"] = labels
    df["label_transition_confidence"] = np.where(np.array(labels) >= 0, 0.8, 0.0)
    return df
