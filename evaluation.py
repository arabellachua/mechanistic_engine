"""
evaluation.py — Evaluation fit for a clinical-prediction paper. (Issue 9)

  * Cluster (drug-level) bootstrap 95% CIs: trials of the same drug are correlated, so
    trial-level resampling understates uncertainty.
  * Discrimination (ROC-AUC, PR-AUC), overall accuracy (Brier, scaled Brier) and
    calibration (intercept / slope / ECE), as TRIPOD+AI expects.
  * Decision-curve analysis: net benefit vs "advance all" and "advance none".
  * Baselines the model must beat, with paired bootstrap differences:
        prevalence, disease prior, target-history ("familiarity"), biomarker-only,
        disease + biomarker.
    Beating the target-history baseline is what supports a *biology* claim rather
    than "the model recognises drugs/targets that have worked before".
  * Subgroups (stratum, disease, modality) and seen-vs-novel drugs.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from config import FIG_DIR, N_BOOTSTRAP, RANDOM_SEED, REPORT_DIR, log


# ── Metrics ──────────────────────────────────────────────────────────────────
def scaled_brier(y, p):
    ref = np.mean(y) * (1 - np.mean(y))
    return 1 - brier_score_loss(y, p) / ref if ref > 0 else np.nan


def calibration_stats(y, p, n_bins: int = 10) -> dict:
    y = np.asarray(y, float)
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    a = 0.0
    for _ in range(100):                     # calibration-in-the-large (offset logistic model)
        mu = expit(a + lp)
        h = np.sum(mu * (1 - mu))
        if h <= 0:
            break
        step = np.sum(y - mu) / h
        a += step
        if abs(step) < 1e-10:
            break
    bins = pd.qcut(p, q=min(n_bins, len(np.unique(p))), duplicates="drop")
    grp = pd.DataFrame({"y": y, "p": p, "b": bins}).groupby("b", observed=True)
    ece = float((grp.size() / len(y) * (grp["y"].mean() - grp["p"].mean()).abs()).sum())
    return {"calibration_intercept": a, "calibration_slope": slope, "ece": ece}


METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score,
           "brier": brier_score_loss, "scaled_brier": scaled_brier}


def _group_index(groups):
    uniq, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    return [np.where(inv == k)[0] for k in range(len(uniq))]


def cluster_bootstrap(y, p, groups, fn, n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED):
    y, p = np.asarray(y), np.asarray(p)
    idx_by_group = _group_index(groups)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_group[k] for k in rng.integers(0, len(idx_by_group), len(idx_by_group))])
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(fn(y[idx], p[idx]))
    point = fn(y, p) if len(np.unique(y)) == 2 else np.nan
    lo, hi = (np.percentile(stats, [2.5, 97.5]) if stats else (np.nan, np.nan))
    return point, lo, hi


def paired_bootstrap_diff(y, p1, p2, groups, fn=roc_auc_score, n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED):
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    idx_by_group = _group_index(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_group[k] for k in rng.integers(0, len(idx_by_group), len(idx_by_group))])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(fn(y[idx], p1[idx]) - fn(y[idx], p2[idx]))
    diffs = np.array(diffs)
    point = fn(y, p1) - fn(y, p2)
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if len(diffs) else np.nan
    return point, *np.percentile(diffs, [2.5, 97.5]), min(p_two, 1.0)


def summarize(y, p, groups, name: str) -> dict:
    row = {"predictor": name, "n": len(y), "n_pos": int(np.sum(y)), "n_drugs": len(np.unique(groups))}
    for m, fn in METRICS.items():
        pt, lo, hi = cluster_bootstrap(y, p, groups, fn)
        row.update({m: pt, f"{m}_lo": lo, f"{m}_hi": hi})
    if len(np.unique(y)) == 2:
        row.update(calibration_stats(y, p))
    return row


# ── Baselines ────────────────────────────────────────────────────────────────
def _smoothed_rate(train: pd.DataFrame, test: pd.DataFrame, col: str, label: str, m: float = 10.0) -> np.ndarray:
    prior = train[label].mean()
    stats = train.groupby(col)[label].agg(["sum", "count"])
    rate = (stats["sum"] + m * prior) / (stats["count"] + m)
    return test[col].map(rate).fillna(prior).values


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, label: str) -> dict[str, np.ndarray]:
    out = {"baseline_prevalence": np.full(len(test), train[label].mean()),
           "baseline_disease_prior": _smoothed_rate(train, test, "disease", label),
           "baseline_target_history": _smoothed_rate(train, test, "primary_target", label)}
    ytr = train[label].values.astype(int)
    for name, cols in {"baseline_biomarker_only": ["biomarker_selected", "biomarker_matches_target"]}.items():
        cols = [c for c in cols if c in train]
        if cols and train[cols].nunique().max() > 1:
            lr = Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                           ("m", LogisticRegression(max_iter=1000))]).fit(train[cols], ytr)
            out[name] = lr.predict_proba(test[cols])[:, 1]
    if "biomarker_selected" in train:
        d_tr = pd.get_dummies(train["disease"], prefix="d").astype(float)
        d_te = pd.get_dummies(test["disease"], prefix="d").astype(float).reindex(columns=d_tr.columns, fill_value=0)
        Xtr = pd.concat([d_tr, train[["biomarker_selected"]].astype(float)], axis=1)
        Xte = pd.concat([d_te, test[["biomarker_selected"]].astype(float)], axis=1)
        out["baseline_disease_plus_biomarker"] = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return out


# ── Decision curves & plots ──────────────────────────────────────────────────
def decision_curve(y, p, thresholds=np.arange(0.05, 0.81, 0.01)) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    n, prev = len(y), np.mean(y)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        odds = t / (1 - t)
        rows.append({"threshold": t, "net_benefit": tp / n - fp / n * odds,
                     "treat_all": prev - (1 - prev) * odds, "treat_none": 0.0})
    return pd.DataFrame(rows)


def _plots(name: str, y, preds: dict[str, np.ndarray]) -> None:
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for k, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, lw=2 if k == "model" else 1, label=f"{k} ({roc_auc_score(y, p):.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    axes[0].legend(fontsize=7)
    p = preds["model"]
    bins = pd.qcut(p, q=min(10, len(np.unique(p))), duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "b": bins}).groupby("b", observed=True).mean()
    axes[1].plot(g["p"], g["y"], "o-", label="model (calibrated)")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set(title="Calibration (deciles)", xlabel="Predicted", ylabel="Observed")
    dca = decision_curve(y, p)
    axes[2].plot(dca["threshold"], dca["net_benefit"], label="model")
    axes[2].plot(dca["threshold"], dca["treat_all"], "--", label="advance all")
    axes[2].axhline(0, color="k", lw=0.8, label="advance none")
    axes[2].set(title="Decision curve", xlabel="Threshold probability", ylabel="Net benefit",
                ylim=(min(-0.05, dca["net_benefit"].min()), max(0.05, np.mean(y)) * 1.2))
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}_evaluation.png", dpi=150)
    plt.close()


def subgroup_table(pred: pd.DataFrame, p_col: str, by: list[str], min_n: int = 30) -> pd.DataFrame:
    rows = []
    for col in by:
        for val, grp in pred.groupby(col):
            if len(grp) < min_n or grp["y"].nunique() < 2:
                continue
            pt, lo, hi = cluster_bootstrap(grp["y"].values, grp[p_col].values, grp["group"].values,
                                           roc_auc_score, n_boot=500)
            rows.append({"subgroup": col, "value": val, "n": len(grp), "n_pos": int(grp["y"].sum()),
                         "roc_auc": pt, "roc_auc_lo": lo, "roc_auc_hi": hi})
    return pd.DataFrame(rows)


def evaluate_analysis(res, period: str = "test") -> dict:
    name = res.spec["name"] + ("" if period == "test" else f"_{period}")
    pred = res.predictions[res.predictions["period"] == period]
    if pred.empty or pred["y"].nunique() < 2:
        log.warning(f"{name}: no evaluable {period} predictions")
        return {}
    frame = res.test_df if period == "test" else res.holdout_df
    y, g = pred["y"].values, pred["group"].values
    preds = {"model": pred["p_cal"].values, "model_uncalibrated": pred["p_raw"].values}
    if "p_stage2" in pred and pred["p_stage2"].notna().all():
        preds["stage2_biology_plus_design"] = pred["p_stage2"].values
    if frame is not None:
        preds.update(baseline_predictions(res.train_df, frame.loc[pred.index], res.label))

    summary = pd.DataFrame([summarize(y, p, g, k) for k, p in preds.items()])
    comps = []
    for k, p in preds.items():
        if k.startswith("baseline"):
            d, lo, hi, pv = paired_bootstrap_diff(y, preds["model"], p, g)
            comps.append({"comparison": f"model - {k}", "delta_auc": d, "lo": lo, "hi": hi, "p": pv})
    if "stage2_biology_plus_design" in preds:
        d, lo, hi, pv = paired_bootstrap_diff(y, preds["stage2_biology_plus_design"], preds["model"], g)
        comps.append({"comparison": "stage2 - stage1", "delta_auc": d, "lo": lo, "hi": hi, "p": pv})
    comps = pd.DataFrame(comps)
    subgroups = subgroup_table(pred, "p_cal", ["stratum", "disease", "modality", "drug_seen_in_train"])
    dca = decision_curve(y, preds["model"])

    summary.to_csv(REPORT_DIR / f"{name}_metrics.csv", index=False)
    comps.to_csv(REPORT_DIR / f"{name}_comparisons.csv", index=False)
    subgroups.to_csv(REPORT_DIR / f"{name}_subgroups.csv", index=False)
    dca.to_csv(REPORT_DIR / f"{name}_decision_curve.csv", index=False)
    res.cv_table.to_csv(REPORT_DIR / f"{res.spec['name']}_cv_model_selection.csv", index=False)
    _plots(name, y, {k: v for k, v in preds.items() if k != "model_uncalibrated"})

    m = summary.set_index("predictor").loc["model"]
    log.info(f"{name}: AUC {m.roc_auc:.3f} [{m.roc_auc_lo:.3f}, {m.roc_auc_hi:.3f}], "
             f"Brier {m.brier:.3f}, slope {m.get('calibration_slope', np.nan):.2f} (n={int(m.n)}, pos={int(m.n_pos)})")
    for r in comps.itertuples():
        log.info(f"   {r.comparison:<45} ΔAUC {r.delta_auc:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] p={r.p:.3f}")
    return {"summary": summary, "comparisons": comps, "subgroups": subgroups, "dca": dca}
