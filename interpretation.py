"""
interpretation.py — Data-driven domain attribution. Replaces counterfactual.py and the
hand-weighted composite scores. (Issue 6)

Why counterfactual.py was removed
  It perturbed biology features ("what if DepMap dependency were stronger?") on a model
  fitted to observational data. A target's dependency is not a manipulable property of a
  trial, and the model's response to a synthetic perturbation is not a causal effect, so
  those outputs could not support the claims made from them.

What replaces it
  * SHAP values of the fitted biology model on test trials, summed within each
    evidence domain -> per-trial domain attribution (a data-derived "CDS/TDS/BFS").
  * Global domain importance (mean |domain SHAP|).
  * Leave-one-domain-out ablation: refit without a domain and report ΔAUC with a paired
    drug-level bootstrap CI. This answers "does this domain add predictive information?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from config import DOMAIN_LABELS, FEATURE_REGISTRY, REPORT_DIR, log
from evaluation import paired_bootstrap_diff
from modeling import fit_weighted, make_estimator, sample_weights


def _domain(feature: str) -> str:
    base = feature.replace("missingindicator_", "")
    return FEATURE_REGISTRY.get(base, {}).get("domain", "other")


def shap_frame(res, X: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import shap
    except ImportError:
        log.warning("shap not installed; skipping attribution")
        return None
    est = res.estimator
    try:
        if isinstance(est, Pipeline):
            pre, clf = est[:-1], est[-1]
            Xt = pre.transform(X)
            names = list(pre.get_feature_names_out())
            if res.model_name == "logreg":
                sv = shap.LinearExplainer(clf, shap.maskers.Independent(Xt, max_samples=500)).shap_values(Xt)
            else:
                sv = shap.TreeExplainer(clf).shap_values(Xt)
        else:
            names = list(X.columns)
            sv = shap.TreeExplainer(est).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return pd.DataFrame(sv, columns=names, index=X.index)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"SHAP failed for {res.model_name}: {exc}")
        return None


def domain_attribution(shap_df: pd.DataFrame) -> pd.DataFrame:
    dom = {c: _domain(c) for c in shap_df.columns}
    out = shap_df.T.groupby(pd.Series(dom)).sum().T
    return out.rename(columns=lambda d: f"attr_{d}")


def domain_ablation(res) -> pd.DataFrame:
    tr, te, label = res.train_df, res.test_df, res.label
    conf_col = "label_confidence" if label == "label_primary" else "label_transition_confidence"
    y_tr = tr[label].values.astype(int)
    w = sample_weights(y_tr, tr[conf_col].values)
    y_te = te[label].values.astype(int)
    groups = te["investigational_drug"].fillna(te["nct_id"]).astype(str).values
    full = res.estimator.predict_proba(te[res.features])[:, 1]
    rows = []
    for domain in sorted({_domain(f) for f in res.features}):
        keep = [f for f in res.features if _domain(f) != domain]
        if not keep:
            continue
        est = fit_weighted(make_estimator(res.model_name, res.params), tr[keep], y_tr, w)
        p = est.predict_proba(te[keep])[:, 1]
        d, lo, hi, pv = paired_bootstrap_diff(y_te, full, p, groups, roc_auc_score, n_boot=1000)
        rows.append({"domain": domain, "label": DOMAIN_LABELS.get(domain, domain),
                     "n_features_removed": len(res.features) - len(keep),
                     "auc_without": roc_auc_score(y_te, p), "delta_auc_full_minus_without": d,
                     "lo": lo, "hi": hi, "p": pv})
    out = pd.DataFrame(rows)
    return out.sort_values("delta_auc_full_minus_without", ascending=False) if len(out) else out


def interpret(res) -> dict:
    name = res.spec["name"]
    X = res.test_df[res.features]
    out = {}
    sdf = shap_frame(res, X)
    if sdf is not None:
        attr = domain_attribution(sdf)
        attr.insert(0, "nct_id", res.test_df.loc[attr.index, "nct_id"].values)
        attr.to_csv(REPORT_DIR / f"{name}_domain_attribution_test.csv", index=False)
        glob = attr.drop(columns="nct_id").abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
        glob.to_csv(REPORT_DIR / f"{name}_domain_importance.csv")
        feat_imp = sdf.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
        feat_imp.to_csv(REPORT_DIR / f"{name}_feature_importance.csv")
        out.update({"domain_attribution": attr, "domain_importance": glob, "feature_importance": feat_imp})
        log.info(f"{name} domain importance:\n{glob.round(4).to_string()}")
    abl = domain_ablation(res)
    abl.to_csv(REPORT_DIR / f"{name}_domain_ablation.csv", index=False)
    out["ablation"] = abl
    return out
