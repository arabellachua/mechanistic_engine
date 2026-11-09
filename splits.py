"""
splits.py — Temporal design, locked prospective holdout, grouped CV. (Issue 4)

  * Periods by start year: train (incl. calibration years) | embargo | test | holdout.
    The embargo removes trials whose outcomes resolve during the test era.
  * The holdout (start >= HOLDOUT_START_YEAR, completed at least N years before the
    data freeze) is frozen the first time the pipeline runs: its NCT IDs AND labels are
    written to locks/holdout.json and reused verbatim afterwards, so new publications
    cannot silently change it. Evaluating it requires --unlock-holdout and every access
    is appended to locks/holdout_access.log with a hash of the analysis specification.
  * CV is grouped by investigational drug (StratifiedGroupKFold). v6 used plain
    stratified folds, so trials of the same drug sat on both sides of a fold, which is
    one reason CV and test AUCs disagreed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from config import (CALIB_YEARS, CV_FOLDS, DATA_FREEZE_DATE, EMBARGO_YEARS, HOLDOUT_MIN_YEARS_SINCE_COMPLETION,
                    HOLDOUT_START_YEAR, LOCK_DIR, RANDOM_SEED, TEST_END_YEAR, TEST_START_YEAR, TRAIN_END_YEAR,
                    TRAIN_START_YEAR, log)

LOCK_FILE = LOCK_DIR / "holdout.json"
ACCESS_LOG = LOCK_DIR / "holdout_access.log"
LABEL_COLS = ("label_primary", "label_transition")


def assign_period(df: pd.DataFrame) -> pd.Series:
    y = df["start_year"]
    freeze = pd.Timestamp(DATA_FREEZE_DATE)
    comp = df["primary_completion_date_dt"].fillna(df.get("completion_date_dt"))
    mature = comp.notna() & (comp <= freeze - pd.DateOffset(months=int(12 * HOLDOUT_MIN_YEARS_SINCE_COMPLETION)))
    period = pd.Series("out_of_range", index=df.index)
    period[(y >= TRAIN_START_YEAR) & (y <= TRAIN_END_YEAR)] = "train"
    period[y.isin(CALIB_YEARS)] = "calib"
    period[y.isin(EMBARGO_YEARS)] = "embargo"
    period[(y >= TEST_START_YEAR) & (y <= TEST_END_YEAR)] = "test"
    period[(y >= HOLDOUT_START_YEAR) & mature] = "holdout"
    period[(y >= HOLDOUT_START_YEAR) & ~mature] = "immature"
    return period


def freeze_holdout(df: pd.DataFrame) -> dict:
    """Freeze holdout IDs and labels on first run; verify afterwards."""
    period = assign_period(df)
    hold = df[period == "holdout"]
    current = {c: {r.nct_id: int(getattr(r, c)) for r in hold.itertuples() if getattr(r, c, -1) in (0, 1)}
               for c in LABEL_COLS if c in hold}
    if not LOCK_FILE.exists():
        payload = {"frozen_at": datetime.now(timezone.utc).isoformat(), "labels": current,
                   "sha256": hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()}
        LOCK_FILE.write_text(json.dumps(payload, indent=1))
        log.info(f"Holdout frozen: {({c: len(v) for c, v in current.items()})} trials. Do not look at it until "
                 f"the primary analysis is fully specified.")
        return payload
    frozen = json.loads(LOCK_FILE.read_text())
    for c, labels in frozen["labels"].items():
        now = current.get(c, {})
        changed = sum(1 for k, v in labels.items() if k in now and now[k] != v)
        if changed:
            log.warning(f"{changed} holdout labels ({c}) differ from the frozen version; the FROZEN labels are used.")
    return frozen


def spec_hash(spec: dict, features: list[str], model_name: str, params: dict) -> str:
    blob = json.dumps({"spec": spec, "features": sorted(features), "model": model_name, "params": params},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def log_holdout_access(analysis: str, h: str) -> bool:
    """Returns True if this is the first access for this analysis (confirmatory)."""
    prior = ACCESS_LOG.read_text().splitlines() if ACCESS_LOG.exists() else []
    earlier = [ln for ln in prior if f"\t{analysis}\t" in ln]
    confirmatory = not earlier
    if earlier and not any(ln.endswith(h) for ln in earlier):
        log.warning(f"Holdout re-accessed for '{analysis}' with a DIFFERENT specification. "
                    f"Report this result as post hoc, not confirmatory.")
    with open(ACCESS_LOG, "a") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{analysis}\t{h}\n")
    return confirmatory


def make_splits(pop: pd.DataFrame, label_col: str, include_holdout: bool = False) -> dict[str, pd.DataFrame]:
    pop = pop.copy()
    pop["period"] = assign_period(pop)
    det = pop[pop[label_col].isin([0, 1])]
    out = {
        "train_core": det[det["period"] == "train"],
        "calib": det[det["period"] == "calib"],
        "test": det[det["period"] == "test"],
    }
    if include_holdout and LOCK_FILE.exists():
        frozen = json.loads(LOCK_FILE.read_text())["labels"].get(label_col, {})
        hold = pop[pop["nct_id"].isin(frozen.keys())].copy()
        hold[label_col] = hold["nct_id"].map(frozen).astype(int)
        out["holdout"] = hold
    for k, v in out.items():
        log.info(f"  {k:<10} n={len(v):>5}  positives={int(v[label_col].sum()):>4}  "
                 f"rate={v[label_col].mean() if len(v) else float('nan'):.3f}")
    return out


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = CV_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = len(np.unique(groups))
    k = max(2, min(n_splits, n_groups, int(np.sum(y == 1)), int(np.sum(y == 0))))
    try:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        folds = list(cv.split(np.zeros(len(y)), y, groups))
    except ValueError:
        folds = list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))
    return [(tr, va) for tr, va in folds if len(np.unique(y[va])) == 2 and len(np.unique(y[tr])) == 2]
