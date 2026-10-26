"""Read-only source audit; writes new evidence to a previously unused directory.

Run: .venv/bin/python audit_research_snapshot.py --output-dir <new-directory>
No API calls, model fitting, label changes, or existing output replacement.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


LABELS = ["label_strict", "label_balanced", "label_permissive"]
UNKNOWN = {"", "unknown", "unmapped", "nan", "none", "n/a"}


def entity_keys(series):
    values = series.fillna("").astype(str).str.strip().str.casefold()
    return values.mask(values.isin(UNKNOWN))


def overlap_rows(df, label, cutoff=2015):
    """Count test-trial overlap, keeping unknown entities separate from unseen."""
    eligible = df[label].isin([0, 1]) & df.start_year.notna()
    train = df.loc[eligible & (df.start_year <= cutoff)]
    test = df.loc[eligible & (df.start_year > cutoff)]
    rows = []
    for column in ["canonical_drug", "primary_target", "drug_disease_pair"]:
        if column == "drug_disease_pair":
            drug, disease = entity_keys(df.canonical_drug), entity_keys(df.disease)
            keys = drug + " | " + disease
        else:
            keys = entity_keys(df[column])
        known_train = set(keys.loc[train.index].dropna())
        test_keys = keys.loc[test.index]
        seen = test_keys.isin(known_train) & test_keys.notna()
        unseen = test_keys.notna() & ~seen
        rows.append({
            "label": label, "entity": column, "n_train": len(train),
            "n_test": len(test), "train_positive": int(train[label].sum()),
            "test_positive": int(test[label].sum()),
            "test_seen": int(seen.sum()), "test_unseen": int(unseen.sum()),
            "test_unknown": int(test_keys.isna().sum()),
            "unseen_positive": int(test.loc[unseen, label].sum()),
            "unseen_negative": int((test.loc[unseen, label] == 0).sum()),
            "seen_fraction_all_test": float(seen.mean()) if len(test) else None,
            "excluded_invalid_label": int((~df[label].isin([0, 1])).sum()),
            "excluded_missing_year_with_valid_label": int(
                (df[label].isin([0, 1]) & df.start_year.isna()).sum()),
        })
    return rows


def bootstrap_auc(y, p, groups=None, repeats=2000, seed=42):
    """Percentile CI conditional on fixed predictions; skip one-class replicates.

    For groups, resample whole clusters with replacement, preserving multiplicity.
    This does not cover training, model-selection, or label uncertainty.
    """
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid prediction vector")
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and probabilities in [0, 1]")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if len(np.unique(y)) < 2:
        return {"auc": None, "ci_low": None, "ci_high": None, "valid_replicates": 0}
    if groups is not None and len(groups) != len(y):
        raise ValueError("Group vector length mismatch")
    if groups is not None:
        groups = np.asarray(groups)
        if pd.isna(groups).any():
            raise ValueError("Missing cluster identity must be resolved explicitly")
    clusters = ([np.array([i]) for i in range(len(y))] if groups is None else
                [np.flatnonzero(groups == g) for g in pd.unique(groups)])
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        indices = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        if len(np.unique(y[indices])) == 2:
            scores.append(roc_auc_score(y[indices], p[indices]))
    bounds = np.quantile(scores, [0.025, 0.975]) if scores else [None, None]
    return {"auc": float(roc_auc_score(y, p)), "ci_low": bounds[0],
            "ci_high": bounds[1], "valid_replicates": len(scores)}


def profile(path):
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    df = pd.read_csv(path)
    return df, {
        "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "rows": len(df), "columns": len(header),
        "duplicate_header_names": {k: v for k, v in Counter(header).items() if v > 1},
        "duplicate_nct_ids": int(df.nct_id.duplicated().sum()) if "nct_id" in df else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("mechanistic_engine_output/data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("mechanistic_engine_output/reports"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=2015)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    # Refuse existing destinations, including partially completed prior audits.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    datasets, profiles, overlaps, label_counts, coverage = {}, [], [], [], []
    for name in ["classification_matrix", "full_feature_matrix"]:
        df, info = profile(args.data_dir / f"{name}.csv")
        if df.nct_id.isna().any() or df.nct_id.duplicated().any():
            raise ValueError(f"{name}: nct_id must be unique and present")
        info.update({
            "year_min": float(df.start_year.min()), "year_max": float(df.start_year.max()),
            "strict_balanced_disagreements": int((df.label_strict != df.label_balanced).sum()),
            "balanced_permissive_disagreements": int((df.label_balanced != df.label_permissive).sum()),
        })
        datasets[name] = df
        profiles.append(info)
        for label in LABELS:
            overlaps.extend(dict(dataset=name, **row) for row in overlap_rows(df, label, args.cutoff))
            for (value, provenance), group in df.groupby([label, "label_provenance"], dropna=False):
                label_counts.append(dict(dataset=name, label=label, value=int(value),
                                         provenance=provenance, n=len(group)))
        for column in df:
            if column.endswith("_missing") or column.startswith("has_real_"):
                coverage.append(dict(dataset=name, column=column, n=len(df),
                                     n_null=int(df[column].isna().sum()),
                                     n_zero=int((df[column] == 0).sum()),
                                     n_one=int((df[column] == 1).sum())))

    left, right = datasets.values()
    common = left.merge(right, on="nct_id", suffixes=("_classification", "_full"), validate="one_to_one")
    comparisons = []
    for column in LABELS + ["canonical_drug", "primary_target", "disease", "start_year"]:
        a, b = common[column + "_classification"], common[column + "_full"]
        mismatch = ~(a.eq(b) | (a.isna() & b.isna()))
        comparisons.append(dict(field=column, common_trials=len(common), disagreements=int(mismatch.sum())))
    conflicts = common.loc[
        common.label_strict_classification != common.label_strict_full,
        ["nct_id", "label_strict_classification", "label_strict_full",
         "label_provenance_classification", "label_provenance_full"],
    ]
    conflicts.to_csv(args.output_dir / "cross_snapshot_label_conflicts.csv", index=False)

    # Stratified review queue, not completed adjudication or a prevalence sample.
    queue_parts = []
    for _, group in left.groupby(["label_strict", "label_provenance"], sort=True):
        queue_parts.append(group.sample(n=min(20, len(group)), random_state=42))
    queue = pd.concat(queue_parts)
    remaining = left.loc[~left.nct_id.isin(queue.nct_id)]
    queue = pd.concat([queue, remaining.sample(n=min(max(150-len(queue), 0), len(remaining)), random_state=42)])
    queue = queue[["nct_id", "title", "status", "canonical_drug", "primary_target",
                   "disease", "start_year"]].sort_values("nct_id")
    # Keep heuristic labels in a separate key to support blinded first-pass review.
    left.loc[left.nct_id.isin(queue.nct_id), ["nct_id"] + LABELS + ["label_provenance"]].to_csv(
        args.output_dir / "label_review_key.csv", index=False)
    queue["trial_url"] = "https://clinicaltrials.gov/study/" + queue.nct_id
    for column in ["reviewer", "primary_endpoint", "endpoint_met_yes_no_unclear",
                   "effect_direction", "evidence_url", "evidence_date", "notes"]:
        queue[column] = ""
    queue.to_csv(args.output_dir / "label_review_queue.csv", index=False)

    # Existing deep predictions can be assessed only against an identity-matched snapshot.
    predictions, pred_info = profile(args.reports_dir / "deep_predictions.csv")
    profiles.append(pred_info)
    expected = right.loc[right.label_permissive.isin([0, 1]) & (right.start_year > args.cutoff)]
    joined = predictions.merge(expected, on="nct_id", suffixes=("_prediction", "_source"), validate="one_to_one")
    prediction_match = (set(predictions.nct_id) == set(expected.nct_id)
                        and joined.label.eq(joined.label_permissive).all())
    for column in ["start_year", "canonical_drug", "primary_target", "disease"]:
        prediction_match = prediction_match and joined[column + "_prediction"].eq(joined[column + "_source"]).all()
    ci_rows = []
    if prediction_match:
        for column in [c for c in predictions if c.endswith("_prob")]:
            for unit in ["trial", "canonical_drug"]:
                groups = None
                if unit == "canonical_drug":
                    # Unknown drugs share one conservative cluster; record this assumption.
                    groups = entity_keys(predictions.canonical_drug).fillna("__unknown__").to_numpy()
                ci_rows.append(dict(model_probability=column, bootstrap_unit=unit,
                                    n=len(predictions), positives=int(predictions.label.sum()),
                                    repeats=args.bootstrap, **bootstrap_auc(predictions.label,
                                        predictions[column], groups, args.bootstrap)))
    pd.DataFrame(ci_rows).to_csv(args.output_dir / "deep_auc_intervals.csv", index=False)
    for name, rows in [("entity_overlap", overlaps), ("label_provenance_counts", label_counts),
                       ("coverage_flags", coverage), ("snapshot_comparison", comparisons)]:
        pd.DataFrame(rows).to_csv(args.output_dir / f"{name}.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "cutoff": args.cutoff, "sources": profiles,
        "common_nct_ids": len(common), "classification_only": len(left)-len(common),
        "full_only": len(right)-len(common), "review_queue_size": len(queue),
        "deep_predictions_match_full_snapshot": bool(prediction_match),
        "limitations": ["File modification dates are not verified generation dates.",
                        "No retraining, external source validation or manual adjudication performed.",
                        "CIs condition on saved predictions and omit training/selection uncertainty.",
                        "Unknown drugs share one cluster in drug bootstrap; canonical drugs are not full regimen identities.",
                        "Snapshot matching cannot establish historical database availability or model provenance.",
                        "Review queue is deliberately stratified; unweighted error rates are not population estimates."],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
