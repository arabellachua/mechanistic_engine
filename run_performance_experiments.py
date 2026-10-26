"""Small, predeclared temporal model search on one frozen trial snapshot.

No external APIs or replacement of existing outputs. The post-2015 holdout was
previously inspected in this project: this is retrospective, not fresh validation.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_sets, compute_composite_scores


SEED = 42
CORE_COLUMNS = [
    "CDS", "BFS", "TWS", "ot_genetic_association", "ot_somatic_mutation",
    "ot_rna_expression", "lineage_specificity", "depmap_essential", "depmap_chronos",
    "cosmic_census_member", "gwas_association_count", "disgenet_score", "tractability",
    "binding_evidence", "potency_proxy", "tumor_expression", "tumor_specificity",
    "normal_tissue_burden", "pathway_evidence", "modality_score", "ot_data_missing",
    "biomarker_missing",
]


def refreshed_features(frame, pubmed):
    """Join without changing cohort/labels; keep unmatched PubMed rows missing."""
    frame = frame.copy().reset_index(drop=True)
    pubmed = pubmed.copy()
    keys = ["primary_target", "disease"]
    for key in keys:
        frame["_" + key] = frame[key].fillna("").str.strip().str.casefold()
        pubmed["_" + key] = pubmed[key].fillna("").str.strip().str.casefold()
    join_keys = ["_" + key for key in keys]
    raw = ["pubmed_total_raw", "pubmed_clinical_trial_raw", "pubmed_recent_raw"]
    joined = frame.merge(pubmed[join_keys + raw + ["pubmed_status"]],
                         on=join_keys, how="left", validate="many_to_one", sort=False)
    valid = joined.pubmed_status.eq("complete") & joined[raw].notna().all(axis=1)
    joined.loc[~valid, raw] = np.nan
    total, clinical, recent = (joined[c] for c in raw)
    joined["pubmed_pair_count"] = (total / 1000).clip(upper=1)
    joined["clinical_trial_pub_count"] = (clinical / 100).clip(upper=1)
    joined["pair_pub_acceleration"] = (recent / (total * 0.4).clip(lower=1)).clip(upper=1)
    joined["pubmed_data_missing"] = (~valid).astype(int)
    # Raw counts retain information lost by clipping large publication counts.
    for col in raw:
        joined["log_" + col] = np.log1p(joined[col])
    fraction = clinical / total.clip(lower=1)
    joined["clinical_publication_fraction"] = fraction
    # Existing names are proxies, not measurements of all human studies.
    joined["human_study_fraction"] = fraction
    joined["translational_study_fraction"] = fraction * 0.7
    joined["pub_acceleration"] = joined.pair_pub_acceleration
    joined["literature_diversity"] = (joined.pubmed_pair_count + joined.clinical_trial_pub_count) * 0.5
    scores = pd.DataFrame([compute_composite_scores(row) for row in joined.to_dict("records")])
    # Only update publication-dependent composite scores; other old scores stay fixed.
    joined[["MCS", "EMS"]] = scores[["MCS", "EMS"]]
    joined["BIOLOGY_SCORE"] = (0.25 * joined.CDS + 0.2 * joined.TDS + 0.2 * joined.MCS
                                + 0.15 * joined.TWS + 0.1 * joined.BFS + 0.1 * joined.EMS).round(4)
    return joined, valid


def feature_views(frame, refreshed):
    legacy_cols = list(build_feature_sets(frame, [])["composite"].columns)
    updated_cols = list(build_feature_sets(refreshed, [])["composite"].columns)
    updated_cols += ["log_pubmed_total_raw", "log_pubmed_clinical_trial_raw",
                     "log_pubmed_recent_raw", "clinical_publication_fraction"]
    return {"legacy": frame[legacy_cols].fillna(-1),
            "refreshed": refreshed[updated_cols],
            "biology_core": frame[CORE_COLUMNS]}


def temporal_folds(frame):
    """Predeclared expanding windows entirely inside the <=2015 training pool."""
    folds = []
    for start, end in [(2007, 2009), (2010, 2012), (2013, 2015)]:
        train = np.flatnonzero(frame.start_year.to_numpy() < start)
        val = np.flatnonzero(frame.start_year.between(start, end).to_numpy())
        if len(train) < 30 or len(val) < 20:
            raise ValueError("Insufficient trials for the declared temporal folds")
        if any(frame.iloc[idx].label_strict.nunique() < 2 for idx in [train, val]):
            raise ValueError("A fold contains fewer than two label classes")
        folds.append((train, val))
    return folds


def candidates():
    estimators = {
        "scaled_lr_c01": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                        StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, random_state=SEED)),
        "scaled_lr_c1": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                       StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)),
        "gb_depth1": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                    GradientBoostingClassifier(n_estimators=100, learning_rate=0.03,
                                                               max_depth=1, min_samples_leaf=15, random_state=SEED)),
        "gb_depth2": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                    GradientBoostingClassifier(n_estimators=100, learning_rate=0.03,
                                                               max_depth=2, min_samples_leaf=15, random_state=SEED)),
        "rf_depth4": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                    RandomForestClassifier(n_estimators=200, max_depth=4,
                                                           min_samples_leaf=10, max_features="sqrt",
                                                           random_state=SEED, n_jobs=1)),
    }
    result = {f"{view}__{name}": (view, model) for view in ["legacy", "refreshed", "biology_core"]
              for name, model in estimators.items()}
    # Exact existing GradBoost configuration, chosen as reference BEFORE running.
    result["reference_legacy_gradboost"] = ("legacy", GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, min_samples_leaf=10,
        subsample=0.8, random_state=SEED))
    return result


def metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if np.sum(y) else None,
            "brier": float(brier_score_loss(y, p))}


def paired_auc_interval(y, baseline, selected, groups=None, repeats=2000):
    y, baseline, selected = map(np.asarray, [y, baseline, selected])
    clusters = ([np.array([i]) for i in range(len(y))] if groups is None else
                [np.flatnonzero(np.asarray(groups) == key) for key in pd.unique(np.asarray(groups))])
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repeats):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        if np.unique(y[idx]).size == 2:
            a, b = roc_auc_score(y[idx], baseline[idx]), roc_auc_score(y[idx], selected[idx])
            values.append([a, b, b-a])
    bounds = np.quantile(values, [0.025, 0.975], axis=0)
    return {name: {"low": float(bounds[0, i]), "high": float(bounds[1, i])}
            for i, name in enumerate(["reference_auc", "selected_auc", "auc_difference"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path("mechanistic_engine_output/data/classification_matrix.csv")
    pm_source = Path("mechanistic_engine_output/pubmed_refresh_2026-09-14/pubmed_pair_features.csv")
    original = pd.read_csv(source)
    if original.nct_id.isna().any() or original.nct_id.duplicated().any():
        raise ValueError("Trial IDs must be present and unique")
    eligible = original.label_strict.isin([0, 1]) & original.start_year.notna()
    frame = original.loc[eligible].reset_index(drop=True)
    refreshed, matched = refreshed_features(frame, pd.read_csv(pm_source))
    views = feature_views(frame, refreshed)
    train_idx = np.flatnonzero(frame.start_year.le(2015).to_numpy())
    test_idx = np.flatnonzero(frame.start_year.gt(2015).to_numpy())
    folds = temporal_folds(frame.iloc[train_idx].reset_index(drop=True))
    specs = candidates()
    plan = {"label": "label_strict (unchanged heuristic labels)", "seed": SEED,
            "selection_rule": "maximum mean rolling-validation AUC, then minimum mean Brier, then candidate name",
            "candidate_names": list(specs), "n_input": len(original), "n_used": len(frame),
            "excluded_invalid_label": int((~original.label_strict.isin([0, 1])).sum()),
            "excluded_missing_year_valid_label": int((original.label_strict.isin([0, 1]) & original.start_year.isna()).sum()),
            "n_train": len(train_idx), "n_test": len(test_idx),
            "pubmed_matched_rows": int(matched.sum()), "pubmed_unmatched_rows": int((~matched).sum()),
            "duplicate_mapping_columns_excluded_from_features": [c for c in frame if c.startswith("drug_is_mapped.")],
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, pm_source, Path(__file__), Path('features.py')]},
            "feature_columns": {k: list(v.columns) for k, v in views.items()},
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "limitations": ["Previously inspected temporal holdout, not fresh external validation.",
                            "Labels are heuristic and not adjudicated; this freezes one old snapshot, not a unified new cohort.",
                            "Current/publication-date features can contain post-trial knowledge; retrospective only.",
                            "Core ablation reduces literature/approval features but does not establish absence of leakage.",
                            "Bootstrap intervals condition on fitted predictions, not model selection/training uncertainty."]}
    (args.output_dir / "experiment_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    split = frame[["nct_id", "start_year", "label_strict"]].copy()
    split["split"] = np.where(frame.start_year <= 2015, "train", "test")
    split.to_csv(args.output_dir / "split_ids.csv", index=False)
    refreshed.to_csv(args.output_dir / "experiment_snapshot.csv", index=False)
    fold_rows, scores = [], []
    for number, (tr, va) in enumerate(folds):
        for role, indices in [("train", tr), ("validation", va)]:
            fold_rows.extend({"fold": number, "role": role, "nct_id": frame.iloc[train_idx[i]].nct_id}
                             for i in indices)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "validation_split_ids.csv", index=False)
    for name, (view, estimator) in specs.items():
        for number, (tr, va) in enumerate(folds):
            tr, va = train_idx[tr], train_idx[va]
            model = clone(estimator).fit(views[view].iloc[tr], frame.label_strict.iloc[tr])
            p = model.predict_proba(views[view].iloc[va])[:, 1]
            scores.append(dict(candidate=name, fold=number, n_train=len(tr), n_val=len(va),
                               **metrics(frame.label_strict.iloc[va], p)))
        print(f"Validated {name}", flush=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(args.output_dir / "validation_folds.csv", index=False)
    ranking = scores.groupby("candidate", as_index=False).agg(
        mean_auc=("auc", "mean"), sd_auc=("auc", "std"), mean_brier=("brier", "mean"),
        mean_average_precision=("average_precision", "mean"))
    ranking = ranking.sort_values(["mean_auc", "mean_brier", "candidate"], ascending=[False, True, True])
    ranking.to_csv(args.output_dir / "validation_comparison.csv", index=False)
    chosen = ranking.iloc[0].candidate
    # Freeze selection on disk before computing ANY final holdout predictions.
    (args.output_dir / "selected_model.json").write_text(json.dumps({"candidate": chosen,
        "validation_auc": float(ranking.iloc[0].mean_auc)}, indent=2) + "\n")
    predictions = frame.iloc[test_idx][["nct_id", "start_year", "canonical_drug", "primary_target", "disease", "label_strict"]].copy()
    final = []
    for role, name in [("reference", "reference_legacy_gradboost"), ("selected", chosen)]:
        view, estimator = specs[name]
        model = clone(estimator).fit(views[view].iloc[train_idx], frame.label_strict.iloc[train_idx])
        p = model.predict_proba(views[view].iloc[test_idx])[:, 1]
        predictions[role + "_prob"] = p
        final.append(dict(role=role, candidate=name, n_test=len(test_idx), **metrics(predictions.label_strict, p)))
    prevalence = float(frame.label_strict.iloc[train_idx].mean())
    final.append(dict(role="prevalence", candidate="training_prevalence", n_test=len(test_idx),
                      **metrics(predictions.label_strict, np.full(len(test_idx), prevalence))))
    pd.DataFrame(final).to_csv(args.output_dir / "holdout_comparison.csv", index=False)
    predictions.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    intervals = {}
    for name, groups in [("trial", None), ("drug_cluster", predictions.canonical_drug.fillna("unknown").str.casefold().to_numpy())]:
        intervals[name] = paired_auc_interval(predictions.label_strict, predictions.reference_prob,
                                              predictions.selected_prob, groups)
    (args.output_dir / "paired_bootstrap.json").write_text(json.dumps(intervals, indent=2) + "\n")
    print(json.dumps({"selected": chosen, "holdout": final, "intervals": intervals}, indent=2), flush=True)


if __name__ == "__main__":
    main()
