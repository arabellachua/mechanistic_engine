"""Training-period-only test of entity context; never score post-2015 trials."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_performance_experiments import CORE_COLUMNS, SEED, metrics, temporal_folds


CONTEXTS = {
    "numeric_reference": [],
    "target_context": ["primary_target", "disease", "modality"],
    "drug_context": ["canonical_drug", "primary_target", "disease", "modality"],
    "drug_pair_context": ["canonical_drug", "primary_target", "disease", "modality", "drug_disease_pair"],
}


def prepare_training_frame(source):
    # Filter by date first: later labels play no role in eligibility or evaluation.
    frame = source.loc[source.start_year.le(2015)].copy()
    frame = frame.loc[frame.label_strict.isin([0, 1])].reset_index(drop=True)
    for column in ["canonical_drug", "primary_target", "disease", "modality"]:
        frame[column] = frame[column].fillna("unknown").astype(str).str.strip().str.casefold()
        frame[column] = frame[column].replace({"": "unknown", "unmapped": "unknown"})
    frame["drug_disease_pair"] = frame.canonical_drug + " | " + frame.disease
    frame["target_disease_pair"] = frame.primary_target + " | " + frame.disease
    if frame.nct_id.isna().any() or frame.nct_id.duplicated().any():
        raise ValueError("Training trial IDs must be unique and present")
    return frame


def context_model(columns):
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    if not columns:
        # Match the previous experiment's numeric reference exactly.
        return make_pipeline(ColumnTransformer([("numeric", numeric, CORE_COLUMNS)]),
                             LogisticRegression(C=1.0, max_iter=3000, random_state=SEED))
    preprocessing = ColumnTransformer([
        ("numeric", numeric, CORE_COLUMNS),
        ("context", OneHotEncoder(handle_unknown="ignore"), columns),
    ])
    return make_pipeline(preprocessing, LogisticRegression(C=1.0, max_iter=3000, random_state=SEED))


def make_folds(frame):
    folds = {"temporal": temporal_folds(frame)}
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds["unseen_target_disease_pair"] = list(splitter.split(
        frame, frame.label_strict, frame.target_disease_pair))
    for mode, partitions in folds.items():
        for tr, va in partitions:
            if set(tr) & set(va):
                raise ValueError("Train/validation row overlap")
            if any(frame.iloc[idx].label_strict.nunique() < 2 for idx in [tr, va]):
                raise ValueError("Both outcome classes are required in each fold")
            if mode == "unseen_target_disease_pair" and (
                set(frame.iloc[tr].target_disease_pair) & set(frame.iloc[va].target_disease_pair)
            ):
                raise ValueError("Grouped validation contains a shared target-disease pair")
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path("mechanistic_engine_output/data/classification_matrix.csv")
    original = pd.read_csv(source)
    frame = prepare_training_frame(original)
    folds = make_folds(frame)
    estimators = {name: context_model(columns) for name, columns in CONTEXTS.items()}
    plan = {
        "seed": SEED, "train_period": "start_year <= 2015", "n_trials": len(frame),
        "n_positive": int(frame.label_strict.sum()), "n_post2015_scored": 0,
        "excluded_post2015_rows": int(original.start_year.gt(2015).sum()),
        "excluded_missing_year_rows": int(original.start_year.isna().sum()),
        "excluded_invalid_label_training_rows": int((original.start_year.le(2015) & ~original.label_strict.isin([0,1])).sum()),
        "contexts": CONTEXTS, "numeric_columns": CORE_COLUMNS,
        "ensemble": "fixed 50:50 probability average of numeric_reference and drug_context",
        "estimator": "LogisticRegression(C=1.0, max_iter=3000); preprocessing fitted on each training fold",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source, Path(__file__), Path('run_performance_experiments.py')]},
        "limitations": ["Development-only comparisons, including previously used temporal validation windows.",
                        "Grouped folds mix years within the training period; they are not a temporal external validation.",
                        "Unseen pairs may contain previously seen drugs and targets individually.",
                        "Identity effects can reflect label-construction shortcuts rather than mechanistic evidence.",
                        "Labels and database time leakage are unchanged; no main model replacement or claimed test improvement."]}
    (args.output_dir / "experiment_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results, predictions, splits = [], [], []
    for mode, partitions in folds.items():
        for fold, (tr, va) in enumerate(partitions):
            for role, indices in [("train", tr), ("validation", va)]:
                splits.extend({"mode": mode, "fold": fold, "role": role, "nct_id": frame.iloc[i].nct_id,
                               "year": int(frame.iloc[i].start_year)} for i in indices)
            probabilities = {}
            for name, estimator in estimators.items():
                model = clone(estimator).fit(frame.iloc[tr], frame.label_strict.iloc[tr])
                probabilities[name] = model.predict_proba(frame.iloc[va])[:, 1]
            probabilities["fixed_ensemble"] = (probabilities["numeric_reference"] + probabilities["drug_context"]) * 0.5
            for name, p in probabilities.items():
                results.append({"mode": mode, "fold": fold, "candidate": name,
                                "n_train": len(tr), "n_val": len(va),
                                **metrics(frame.label_strict.iloc[va], p)})
                predictions.extend({"mode": mode, "fold": fold, "candidate": name,
                                    "nct_id": frame.iloc[i].nct_id, "label": int(frame.iloc[i].label_strict),
                                    "probability": float(prob)} for i, prob in zip(va, p))
            print(f"Completed {mode} fold {fold}", flush=True)
    result = pd.DataFrame(results)
    summary = result.groupby(["mode", "candidate"], as_index=False).agg(
        mean_auc=("auc", "mean"), sd_auc=("auc", "std"), mean_brier=("brier", "mean"),
        mean_average_precision=("average_precision", "mean"))
    reference = result.loc[result.candidate == "numeric_reference", ["mode", "fold", "auc"]].rename(columns={"auc":"reference_auc"})
    paired = result.merge(reference, on=["mode", "fold"], validate="many_to_one")
    paired["auc_difference"] = paired.auc - paired.reference_auc
    summary.to_csv(args.output_dir / "validation_summary.csv", index=False)
    paired.to_csv(args.output_dir / "validation_folds.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(splits).to_csv(args.output_dir / "split_ids.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
