"""
modeling.py — Model fitting for one pre-specified analysis. (Issues 4, 6, 7, 8)
Merges v7 modeling.py and v8 modeling_v8.py.

Key corrections
  * Model selection by grouped CV on the TRAINING data. v6 picked the "best" model by
    TEST AUC across 45 configurations, which is optimistic selection on the test set.
  * Class imbalance handled with weights computed from each training set (v6 hard-coded
    scale_pos_weight=4.0), multiplied by label-confidence weights.
  * Probability calibration on a temporally later calibration slice (Platt scaling
    fitted on held-out years), never on the data used to fit the model.
  * Linear/RF models impute inside a pipeline with missingness indicators; LightGBM
    uses native NaN handling. No fillna(-1).
  * Strata (targeted / IO / cytotoxic / hormonal) and phases are separate analyses
    declared in config, not an afterthought.
  * Two-stage model: stage-2 = logit(stage-1 biology probability, out-of-fold on train)
    + trial-design features, so the added value of design is measured, not assumed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (FEATURE_REGISTRY, MIN_TEST_POSITIVES, MIN_TRAIN_POSITIVES, RANDOM_SEED, STAGE1_ALLOWED,
                    STAGE1_SENSITIVITY_ALLOWED, TRIAL_DESIGN, log)
import leakage
import splits

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False

GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0]},
    "lightgbm": {"num_leaves": [7, 15], "min_child_samples": [10, 30]},
    "random_forest": {"min_samples_leaf": [5, 20]},
}


def make_estimator(name: str, params: dict):
    if name == "logreg":
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_SEED, **params))])
    if name == "lightgbm":
        if not HAS_LGB:
            raise ImportError("lightgbm not installed")
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_SEED,
                                  verbose=-1, **params)
    if name == "random_forest":
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("clf", RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1,
                                                        random_state=RANDOM_SEED, **params))])
    raise ValueError(name)


def fit_weighted(est, X: pd.DataFrame, y: np.ndarray, w: np.ndarray):
    if isinstance(est, Pipeline):
        est.fit(X, y, clf__sample_weight=w)
    else:
        est.fit(X, y, sample_weight=w)
    return est


def sample_weights(y: np.ndarray, confidence: np.ndarray | None, balance: bool = True) -> np.ndarray:
    w = np.ones(len(y)) if confidence is None else np.asarray(confidence, float).copy()
    if balance:
        pos, neg = (y == 1).sum(), (y == 0).sum()
        if pos and neg:
            w = w * np.where(y == 1, neg / pos, 1.0)
    return w / w.mean()


class PlattCalibrator:
    """Logistic recalibration of logit(p), fitted on a temporally held-out slice."""

    def __init__(self):
        self.lr = None

    def fit(self, p: np.ndarray, y: np.ndarray):
        if len(np.unique(y)) == 2 and len(y) >= 20:
            self.lr = LogisticRegression(C=1e4, max_iter=1000).fit(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
        else:
            log.warning("Calibration slice too small/one-class: probabilities left uncalibrated")
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.lr is None:
            return p
        return self.lr.predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]


def feature_columns(df: pd.DataFrame, allowed: set[str]) -> list[str]:
    cols = [c for c, m in FEATURE_REGISTRY.items() if m["leakage_class"] in allowed and c in df.columns]
    return [c for c in cols if df[c].notna().any() and df[c].nunique(dropna=True) > 1]


def _groups(df: pd.DataFrame) -> np.ndarray:
    return df["investigational_drug"].fillna(df["nct_id"]).astype(str).values


def cv_evaluate(name: str, params: dict, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
                groups: np.ndarray) -> tuple[float, float, np.ndarray]:
    oof = np.full(len(y), np.nan)
    aucs = []
    for tr, va in splits.grouped_folds(y, groups):
        est = fit_weighted(make_estimator(name, params), X.iloc[tr], y[tr], w[tr])
        p = est.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y[va], p))
    return (float(np.mean(aucs)) if aucs else np.nan, float(np.std(aucs)) if aucs else np.nan, oof)


@dataclass
class AnalysisResult:
    spec: dict
    label: str
    features: list[str]
    cv_table: pd.DataFrame
    model_name: str
    params: dict
    estimator: object
    calibrator: PlattCalibrator
    predictions: pd.DataFrame            # test (+ holdout) rows with p_raw, p_cal, p_stage2
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    holdout_df: pd.DataFrame | None = None
    holdout_confirmatory: bool | None = None
    stage2_features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def select_population(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    pop = df[df["phase_group"] == spec["phase"]]
    if spec.get("stratum", "all") != "all":
        pop = pop[pop["stratum"] == spec["stratum"]]
    return pop[pop["primary_target"].notna()]


def run_analysis(df: pd.DataFrame, spec: dict, unlock_holdout: bool = False,
                 include_undated: bool = False, models=("logreg", "lightgbm", "random_forest")) -> AnalysisResult | None:
    label = spec["label"]
    conf_col = "label_confidence" if label == "label_primary" else "label_transition_confidence"
    log.info(f"=== Analysis {spec['name']}{' [+undated knowledge]' if include_undated else ''} ===")
    pop = select_population(df, spec)
    sp = splits.make_splits(pop, label, include_holdout=unlock_holdout)
    tr, ca, te = sp["train_core"], sp["calib"], sp["test"]
    if tr[label].sum() < MIN_TRAIN_POSITIVES or te[label].sum() < MIN_TEST_POSITIVES:
        log.warning(f"{spec['name']}: insufficient positives (train={int(tr[label].sum())}, "
                    f"test={int(te[label].sum())}); analysis skipped and reported as such")
        return None

    allowed = STAGE1_SENSITIVITY_ALLOWED if include_undated else STAGE1_ALLOWED
    feats = feature_columns(tr, allowed)
    if not feats:
        log.warning(f"{spec['name']}: no non-constant biology features in this population; "
                    f"analysis skipped and reported as such")
        return None
    leakage.assert_allowed(feats, "stage1_sensitivity" if include_undated else "stage1")
    leakage.audit_univariate(tr[feats], tr[label])
    leakage.audit_split_overlap(tr, te)

    X, y = tr[feats], tr[label].values.astype(int)
    w = sample_weights(y, tr[conf_col].values)
    g = _groups(tr)

    rows, best = [], None
    for name in models:
        if name == "lightgbm" and not HAS_LGB:
            continue
        for params in ParameterGrid(GRIDS[name]):
            m, s, oof = cv_evaluate(name, params, X, y, w, g)
            rows.append({"model": name, "params": params, "cv_auc_mean": m, "cv_auc_sd": s})
            if not np.isnan(m) and (best is None or m > best[2]):
                best = (name, params, m, oof)
    cv_table = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    if best is None:
        log.warning(f"{spec['name']}: CV failed for all models")
        return None
    name, params, cv_auc, oof = best
    log.info(f"Selected by grouped CV: {name} {params} (CV AUC {cv_auc:.3f})")

    est = fit_weighted(make_estimator(name, params), X, y, w)
    cal = PlattCalibrator()
    if len(ca):
        cal.fit(est.predict_proba(ca[feats])[:, 1], ca[label].values.astype(int))

    eval_sets = [("test", te)] + ([("holdout", sp["holdout"])] if "holdout" in sp else [])
    preds = []
    for pname, part in eval_sets:
        if not len(part):
            continue
        p_raw = est.predict_proba(part[feats])[:, 1]
        preds.append(pd.DataFrame({
            "nct_id": part["nct_id"].values, "period": pname, "y": part[label].values.astype(int),
            "p_raw": p_raw, "p_cal": cal.transform(p_raw), "group": _groups(part),
            "stratum": part["stratum"].values, "disease": part["disease"].values,
            "modality": part["modality"].values, "primary_target": part["primary_target"].values,
            "investigational_drug": part["investigational_drug"].values, "start_year": part["start_year"].values,
        }, index=part.index))
    pred_df = pd.concat(preds)
    seen = set(tr["investigational_drug"].dropna())
    pred_df["drug_seen_in_train"] = pred_df["investigational_drug"].isin(seen)

    # ── Stage 2: biology probability + trial design ──
    design = [c for c in feature_columns(tr, {TRIAL_DESIGN})]
    stage2_feats = []
    if design and not np.isnan(oof).all():
        leakage.assert_allowed(design, "stage2")
        ok = ~np.isnan(oof)
        s2_train = tr.loc[ok, design].copy()
        s2_train.insert(0, "stage1_logit", logit(np.clip(oof[ok], 1e-6, 1 - 1e-6)))
        s2 = fit_weighted(make_estimator("logreg", {"C": 0.1}), s2_train, y[ok], w[ok])
        stage2_feats = list(s2_train.columns)
        s2_cal = PlattCalibrator()
        if len(ca):
            s2_ca = ca[design].copy()
            s2_ca.insert(0, "stage1_logit", logit(np.clip(est.predict_proba(ca[feats])[:, 1], 1e-6, 1 - 1e-6)))
            s2_cal.fit(s2.predict_proba(s2_ca)[:, 1], ca[label].values.astype(int))
        parts = {"test": te, **({"holdout": sp["holdout"]} if "holdout" in sp else {})}
        for pname, part in parts.items():
            idx = pred_df.index[pred_df["period"] == pname]
            if not len(idx):
                continue
            s2_x = part.loc[idx, design].copy()
            s2_x.insert(0, "stage1_logit", logit(np.clip(pred_df.loc[idx, "p_raw"].values, 1e-6, 1 - 1e-6)))
            pred_df.loc[idx, "p_stage2"] = s2_cal.transform(s2.predict_proba(s2_x)[:, 1])

    confirmatory = None
    if "holdout" in sp and len(sp["holdout"]):
        confirmatory = splits.log_holdout_access(spec["name"],
                                                 splits.spec_hash(spec, feats, name, params))
    return AnalysisResult(spec=spec, label=label, features=feats, cv_table=cv_table, model_name=name,
                          params=params, estimator=est, calibrator=cal, predictions=pred_df,
                          train_df=tr, test_df=te, holdout_df=sp.get("holdout"), holdout_confirmatory=confirmatory,
                          stage2_features=stage2_feats)
