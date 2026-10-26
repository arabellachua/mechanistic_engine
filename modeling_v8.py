"""
modeling_v8.py — Two-stage model, endpoint-specific models, continuous outcomes.

v8 CHANGES:
  1. Two-stage model: stage 1 = biology score, stage 2 = trial outcome prediction
  2. Endpoint-specific models: separate models for survival/response/PFS/biomarker
  3. Continuous outcome regression where effect size data is available
  4. Separate IO and non-IO models
  5. Split TDS (small molecule vs biologic)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from config import RANDOM_SEED, CV_FOLDS, FIG_DIR, REPORT_DIR, log
from modeling import get_models, evaluate, _safe_f1


# ── Stage 1: Biology scoring model (existing) ─────────────────────────────────
# This is the current model — predicts P(success) from biology features alone.
# Output of stage 1 becomes an input feature for stage 2.

def train_stage1(X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Train stage 1 biology model and return predicted probabilities
    for both train and test sets (train probs via cross-val to avoid leakage).
    """
    mask_tr = y_train.isin([0, 1])
    X_tr = X_train[mask_tr]
    y_tr = y_train[mask_tr]

    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=15,
            max_depth=5, min_child_samples=15, reg_alpha=0.3, reg_lambda=0.3,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=4.0,
            random_state=RANDOM_SEED, verbose=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_SEED,
        )

    model.fit(X_tr, y_tr)

    # Train probabilities via cross-val to avoid leakage into stage 2
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_probs = np.zeros(len(X_tr))
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        fold_model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=15,
            random_state=RANDOM_SEED, verbose=-1,
        ) if HAS_LGB else GradientBoostingClassifier(random_state=RANDOM_SEED)
        fold_model.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        train_probs[val_idx] = fold_model.predict_proba(X_tr.iloc[val_idx])[:, 1]

    mask_te = y_test.isin([0, 1])
    test_probs = model.predict_proba(X_test[mask_te])[:, 1] if mask_te.sum() > 0 else np.array([])

    return model, train_probs, test_probs


# ── Stage 2: Trial outcome model ──────────────────────────────────────────────

TRIAL_SPECIFIC_FEATURES = [
    # These come from ClinicalTrials.gov and are trial-specific, not biology-only
    "drug_is_mapped",
    "modality_score",
    "has_biomarker",
    "biomarker_missing",
    "is_io_trial",
    "drug_class",           # biologic vs small molecule (from v8)
    "endpoint_type_num",    # encoded endpoint type
]

ENDPOINT_TYPE_MAP = {
    "survival": 0,
    "pfs": 1,
    "response": 2,
    "biomarker": 3,
    "other": 4,
    None: -1,
}

def build_stage2_features(df: pd.DataFrame, stage1_probs: np.ndarray,
                           feature_cols: list) -> pd.DataFrame:
    """
    Build stage 2 features: stage 1 biology score + trial-specific features.
    """
    stage2_df = df.copy()
    stage2_df["biology_score_s1"] = stage1_probs

    # Encode endpoint type numerically
    if "endpoint_type" in stage2_df.columns:
        stage2_df["endpoint_type_num"] = stage2_df["endpoint_type"].map(ENDPOINT_TYPE_MAP).fillna(-1)
    else:
        stage2_df["endpoint_type_num"] = -1

    # Encode is_io_trial if present
    if "is_io_trial" not in stage2_df.columns:
        stage2_df["is_io_trial"] = 0

    # Select stage 2 features
    s2_cols = list(dict.fromkeys(
        ["biology_score_s1"] + [c for c in TRIAL_SPECIFIC_FEATURES if c in stage2_df.columns]
    ))
    return stage2_df[s2_cols].fillna(-1)


def train_stage2(X2_train: pd.DataFrame, y_train: pd.Series,
                 X2_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Train stage 2 model combining biology score with trial-specific features."""
    mask_tr = y_train.isin([0, 1])
    mask_te = y_test.isin([0, 1])

    X_tr = X2_train[mask_tr]
    y_tr = y_train[mask_tr]
    X_te = X2_test[mask_te]
    y_te = y_test[mask_te]

    if len(X_tr) < 20 or len(np.unique(y_tr)) < 2:
        return {"error": "insufficient data"}

    models = {}
    results = []

    for name, model in [
        ("LogReg", LogisticRegression(C=0.1, class_weight="balanced",
                                       max_iter=1000, random_state=RANDOM_SEED)),
        ("RF", RandomForestClassifier(n_estimators=100, max_depth=4,
                                       class_weight="balanced", random_state=RANDOM_SEED)),
    ]:
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5
        results.append({"model": name, "auc": auc})
        models[name] = model
        print(f"    Stage 2 {name}: AUC={auc:.3f}")

    best = max(results, key=lambda x: x["auc"])
    return {"models": models, "results": results, "best": best}


# ── Endpoint-specific models ──────────────────────────────────────────────────

def train_endpoint_models(df: pd.DataFrame, feature_cols: list,
                           label_col: str = "label_permissive") -> dict:
    """
    Train separate models for each endpoint type.
    Returns dict of {endpoint_type: model_results}.
    """
    from features import build_feature_sets
    from config import TEST_YEAR_CUTOFF

    endpoint_results = {}
    endpoint_types = ["survival", "response", "pfs", "other"]

    print("\n  === ENDPOINT-SPECIFIC MODELS ===")

    for ep_type in endpoint_types:
        if "endpoint_type" not in df.columns:
            print(f"  {ep_type}: endpoint_type column missing, skipping")
            continue

        # Filter to this endpoint type (or all if endpoint unknown)
        if ep_type == "other":
            ep_df = df[df["endpoint_type"].isna() | (df["endpoint_type"] == "other")]
        else:
            ep_df = df[df["endpoint_type"] == ep_type]

        ep_det = ep_df[ep_df[label_col].isin([0, 1])]

        if len(ep_det) < 30:
            print(f"  {ep_type}: only {len(ep_det)} trials, skipping")
            continue

        train_df = ep_det[ep_det["start_year"] <= TEST_YEAR_CUTOFF]
        test_df = ep_det[ep_det["start_year"] > TEST_YEAR_CUTOFF]

        if len(test_df) < 10 or len(np.unique(test_df[label_col])) < 2:
            print(f"  {ep_type}: insufficient test data ({len(test_df)} trials), skipping")
            continue

        feat_sets = build_feature_sets(train_df, feature_cols)
        X_tr = feat_sets["composite"]
        feat_sets_te = build_feature_sets(test_df, feature_cols)
        X_te = feat_sets_te["composite"].reindex(columns=X_tr.columns, fill_value=-1)

        y_tr = train_df[label_col]
        y_te = test_df[label_col]

        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, num_leaves=10,
                scale_pos_weight=3.0, random_state=RANDOM_SEED, verbose=-1,
            )
        else:
            model = GradientBoostingClassifier(random_state=RANDOM_SEED)

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5

        pos_rate = y_te.mean()
        print(f"  {ep_type:<12}: n={len(ep_det)}, test={len(test_df)}, "
              f"pos_rate={pos_rate:.1%}, AUC={auc:.3f}")

        endpoint_results[ep_type] = {
            "model": model, "auc": auc,
            "n_train": len(train_df), "n_test": len(test_df),
            "pos_rate": pos_rate,
        }

    return endpoint_results


# ── IO vs non-IO separate models ──────────────────────────────────────────────

def train_io_split_models(df: pd.DataFrame, feature_cols: list,
                           label_col: str = "label_permissive") -> dict:
    """
    Train separate models for IO and non-IO trials.
    IO trials: targets PDCD1, CD274, CTLA4
    Non-IO: everything else
    """
    from features import build_feature_sets
    from config import TEST_YEAR_CUTOFF

    print("\n  === IO vs NON-IO MODELS ===")
    results = {}

    for group_name, mask in [
        ("IO", df.get("is_io_trial", pd.Series([0]*len(df), index=df.index)) == 1),
        ("Non-IO", df.get("is_io_trial", pd.Series([0]*len(df), index=df.index)) == 0),
    ]:
        group_df = df[mask]
        det = group_df[group_df[label_col].isin([0, 1])]

        if len(det) < 30:
            print(f"  {group_name}: only {len(det)} trials, skipping")
            continue

        train_df = det[det["start_year"] <= TEST_YEAR_CUTOFF]
        test_df = det[det["start_year"] > TEST_YEAR_CUTOFF]

        if len(test_df) < 10 or len(np.unique(test_df[label_col])) < 2:
            print(f"  {group_name}: insufficient test data, skipping")
            continue

        feat_tr = build_feature_sets(train_df, feature_cols)["composite"]
        feat_te = build_feature_sets(test_df, feature_cols)["composite"]
        feat_te = feat_te.reindex(columns=feat_tr.columns, fill_value=-1)

        y_tr = train_df[label_col]
        y_te = test_df[label_col]

        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.04, num_leaves=12,
                scale_pos_weight=4.0, random_state=RANDOM_SEED, verbose=-1,
            )
        else:
            model = GradientBoostingClassifier(random_state=RANDOM_SEED)

        model.fit(feat_tr, y_tr)
        probs = model.predict_proba(feat_te)[:, 1]
        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5

        print(f"  {group_name:<10}: n={len(det)}, test={len(test_df)}, AUC={auc:.3f}")
        results[group_name] = {"model": model, "auc": auc, "n": len(det)}

    return results


# ── Continuous outcome regression ─────────────────────────────────────────────

def train_continuous_model(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train regression model on continuous outcome scores (effect sizes).
    Only uses trials with structured results providing actual effect size data.
    """
    from features import build_feature_sets
    from config import TEST_YEAR_CUTOFF

    print("\n  === CONTINUOUS OUTCOME REGRESSION ===")

    if "continuous_outcome" not in df.columns:
        print("  No continuous_outcome column found, skipping")
        return {}

    cont_df = df[df["continuous_outcome"].notna()]
    print(f"  Trials with continuous outcomes: {len(cont_df)}")

    if len(cont_df) < 20:
        print("  Insufficient data for continuous model, skipping")
        return {}

    train_df = cont_df[cont_df["start_year"] <= TEST_YEAR_CUTOFF]
    test_df = cont_df[cont_df["start_year"] > TEST_YEAR_CUTOFF]

    if len(test_df) < 10:
        print(f"  Insufficient test data ({len(test_df)}), skipping")
        return {}

    feat_tr = build_feature_sets(train_df, feature_cols)["composite"]
    feat_te = build_feature_sets(test_df, feature_cols)["composite"]
    feat_te = feat_te.reindex(columns=feat_tr.columns, fill_value=-1)

    y_tr = train_df["continuous_outcome"]
    y_te = test_df["continuous_outcome"]

    model = Ridge(alpha=1.0)
    model.fit(feat_tr, y_tr)
    preds = model.predict(feat_te)

    mse = mean_squared_error(y_te, preds)
    r2 = r2_score(y_te, preds)

    print(f"  Regression MSE: {mse:.3f}, R²: {r2:.3f}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    return {"model": model, "mse": mse, "r2": r2,
            "n_train": len(train_df), "n_test": len(test_df)}
