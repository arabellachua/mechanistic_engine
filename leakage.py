"""
leakage.py — Automated audits that turn the paper's validity claims into checks. (Issues 1-5)

  assert_allowed()        every model feature is registered and of an allowed class;
                          outcome proxies raise an error instead of silently training.
  audit_mapping_bias()    is drug-dictionary coverage associated with the outcome? (issue 5)
  audit_univariate()      any single feature with AUC > 0.80 is flagged as suspicious.
  audit_time_bounds()     knowledge cut-off precedes trial start for every row.
  audit_split_overlap()   share of test trials whose drug was seen in training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from config import (FEATURE_REGISTRY, FORBIDDEN_NAME_PATTERNS, OUTCOME_PROXY, REPORT_DIR,
                    STAGE1_ALLOWED, STAGE1_SENSITIVITY_ALLOWED, STAGE2_ALLOWED, log)

_ALLOWED = {"stage1": STAGE1_ALLOWED, "stage1_sensitivity": STAGE1_SENSITIVITY_ALLOWED, "stage2": STAGE2_ALLOWED}


class LeakageError(RuntimeError):
    pass


def assert_allowed(features: list[str], stage: str = "stage1") -> None:
    allowed = _ALLOWED[stage]
    problems = []
    for f in features:
        meta = FEATURE_REGISTRY.get(f)
        if meta is None:
            problems.append(f"{f}: not in FEATURE_REGISTRY")
        elif meta["leakage_class"] == OUTCOME_PROXY:
            problems.append(f"{f}: outcome proxy")
        elif meta["leakage_class"] not in allowed:
            problems.append(f"{f}: class {meta['leakage_class']} not allowed in {stage}")
        if any(p in f.lower() for p in FORBIDDEN_NAME_PATTERNS):
            problems.append(f"{f}: matches forbidden name pattern")
    if problems:
        raise LeakageError("Leakage audit failed:\n  " + "\n  ".join(problems))


def audit_mapping_bias(df: pd.DataFrame, label_col: str = "label_primary") -> pd.DataFrame:
    det = df[df[label_col].isin([0, 1])].copy()
    det["mapped"] = det["investigational_drug"].notna().astype(int)
    rows = []
    tab = pd.crosstab(det["mapped"], det[label_col])
    if tab.shape == (2, 2):
        chi2, p, _, _ = chi2_contingency(tab)
        rate_m = det.loc[det.mapped == 1, label_col].mean()
        rate_u = det.loc[det.mapped == 0, label_col].mean()
        rows.append({"check": "success_rate_mapped_vs_unmapped", "mapped": round(rate_m, 3),
                     "unmapped": round(rate_u, 3), "chi2_p": p})
        msg = (f"Mapping bias: success {rate_m:.1%} (mapped) vs {rate_u:.1%} (unmapped), p={p:.3g}. ")
        msg += ("Mapped population is outcome-enriched: restrict claims to it and report this."
                if p < 0.05 else "No evidence of outcome-related mapping.")
        log.info(msg)
    by_year = det.groupby("start_year")["mapped"].mean().rename("mapping_rate").reset_index()
    by_year.to_csv(REPORT_DIR / "mapping_rate_by_year.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "audit_mapping_bias.csv", index=False)
    return out


def audit_univariate(X: pd.DataFrame, y: pd.Series, threshold: float = 0.80) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        x = X[c]
        ok = x.notna()
        if ok.sum() < 20 or y[ok].nunique() < 2 or x[ok].nunique() < 2:
            continue
        auc = roc_auc_score(y[ok], x[ok])
        rows.append({"feature": c, "auc": auc, "abs_dev": abs(auc - 0.5), "n": int(ok.sum())})
    out = pd.DataFrame(rows).sort_values("abs_dev", ascending=False) if rows else pd.DataFrame()
    flagged = out[(out["auc"] > threshold) | (out["auc"] < 1 - threshold)] if len(out) else out
    for r in flagged.itertuples():
        log.warning(f"Suspiciously predictive single feature: {r.feature} AUC={r.auc:.3f} — check for leakage")
    return out


def audit_time_bounds(df: pd.DataFrame) -> None:
    if "knowledge_cutoff_year" not in df:
        return
    bad = (df["knowledge_cutoff_year"] >= df["start_year"]).sum()
    if bad:
        raise LeakageError(f"{bad} trials have knowledge cut-off >= start year")
    log.info("Time-bound audit passed: all knowledge features predate trial start")


def audit_split_overlap(train: pd.DataFrame, test: pd.DataFrame, col: str = "investigational_drug") -> dict:
    seen = set(train[col].dropna())
    frac = test[col].isin(seen).mean() if len(test) else np.nan
    log.info(f"{frac:.1%} of test trials use a drug seen in training (reported separately as 'seen' vs 'novel')")
    return {"test_frac_seen_drug": frac}
