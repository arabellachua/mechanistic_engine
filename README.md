# Mechanistic Phase 2 Biology Engine

A standalone analytical engine that predicts Phase 2 clinical trial success from biology/mechanism alone.

## What it does

Estimates **P(positive Phase 2 proof-of-concept | biology/mechanism only)** by computing mechanistic features from public databases and training classification models.

## Data Sources

| Source | What it provides | Method |
|--------|-----------------|--------|
| ClinicalTrials.gov | Trial corpus (5,000+ Phase 2 oncology trials) | REST API |
| Open Targets | Target-disease association scores | GraphQL API |
| PubMed | Publication counts and evidence maturity | NCBI E-utilities |
| Curated dictionary | Drug-target-modality mappings for 100+ drugs | Local lookup |

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python run_pipeline.py
```

The pipeline takes ~15-20 minutes (mostly API queries, cached after first run).

To skip API calls and use curated data only (faster, ~2 minutes):
- Edit `config.py` and set `USE_REAL_APIS = False`

## Optional Deep Learning Branch

The default pipeline trains the existing RandomForest/LightGBM baselines. An
optional complementary branch adds learned drug/target/disease embeddings with a
small MLP and TrialTransformer. It does not change `labels.py` or replace the
tree models.

```bash
pip install -r requirements-deep.txt
RUN_DEEP_EXPERIMENTS=1 python run_pipeline.py
```

After `run_pipeline.py` has produced the feature matrix, the deep branch can
also be run standalone:

```bash
python run_deep_experiments.py
```

More details are in `docs/deep_learning_branch.md`.

## Refresh PubMed Only

Refresh all mapped target–disease pairs from the saved raw trial corpus without
replacing existing feature matrices or model reports:

```bash
python refresh_pubmed.py --as-of 2026-09-14 --output-dir mechanistic_engine_output/pubmed_refresh_2026-09-14
```

Repeat the same command to resume incomplete queries. See
`docs/pubmed_refresh.md` for cache validation, missingness, outputs, and the
remaining limitations of publication-date filtering.

## Output Files

| File | Description |
|------|-------------|
| `data/classification_matrix.csv` | Main file: 638 labelled trials × 92 columns (features + labels) |
| `reports/label_audit.csv` | Label audit findings (`audit_labels.py`) |
| `reports/evaluation_original.csv` | Uncertainty-aware evaluation (`evaluate_original.py`) |
| `data/classification_matrix.parquet` | Same in Parquet format |
| `data/data_dictionary.csv` | Description of every column |
| `reports/model_comparison.csv` | All model metrics (ROC AUC, PR AUC, F1, Brier, etc.) |
| `reports/cohort_report.csv` | Cohort-level summary statistics |
| `reports/asset_report.csv` | Per-trial predictions with score breakdowns |
| `reports/error_analysis.csv` | False positive / false negative analysis |
| `reports/cross_validation.csv` | 5-fold cross-validation results |
| `reports/sensitivity_analysis.csv` | Sensitivity analysis across subsets |
| `reports/counterfactual_results.csv` | Counterfactual perturbation results |
| `figures/*.png` | ROC curves, calibration, SHAP, confusion matrices, etc. |

## Project Structure

```
config.py              - Settings, paths, feature lists
drug_target_db.py      - Curated drug-target-modality dictionary
clients.py             - API clients (ClinicalTrials.gov, Open Targets, PubMed)
normalize.py           - Drug/disease/biomarker normalization
labels.py              - Label construction (strict/balanced/permissive)
features.py            - Raw feature computation + composite scores
modeling.py            - Model training, evaluation, calibration
counterfactual.py      - Counterfactual analysis module
reporting.py           - Reports, sensitivity analysis, cross-validation
audit_labels.py        - Adversarial audit of the outcome labels
evaluate_original.py   - Honest evaluation: null baselines, grouped splits, CIs
run_pipeline.py        - Main pipeline runner
run_deep_experiments.py - Optional standalone deep-learning runner
requirements.txt       - Python dependencies
requirements-deep.txt  - Optional PyTorch dependency for deep models
README.md              - This file
```

## Models Trained

- Logistic Regression (with isotonic calibration)
- LightGBM
- Random Forest

Each trained across 3 feature sets (raw, composite, hybrid) × 3 label definitions (strict, balanced, permissive) = 27+ model configurations.

## Label Definitions

`labels.py` defines three, but **they currently produce an identical vector** —
`balanced` is assigned from `strict`, and the `permissive` branch only fires on
rows that are already determinate. Run `python audit_labels.py` to confirm.

Actual distribution: 638 determinate trials, 19.3% positive.

## Evaluating

```bash
python audit_labels.py         # is the label trustworthy?
python evaluate_original.py    # what is the real, uncertainty-aware performance?
```

Read the audit first. Model numbers are only interpretable once you know what
the label encodes. Details in `docs/label_audit.md`.

## Key Results

Measured with `evaluate_original.py` (bootstrap CIs, permutation tests, null
baselines) on the shipped matrix:

| Protocol | Best AUC | 95% CI | Beats null? |
|---|---|---|---|
| Temporal split (pipeline default) | 0.538 | 0.43–0.64 | no (perm p≈0.24) |
| Temporal, unseen (target,disease) only | 0.606 | 0.47–0.74 | no |
| Grouped 5-fold CV on (target,disease) | 0.612 | 0.55–0.68 | yes (perm p=0.002) |
| Stratified CV (optimistic upper bound) | 0.640 | 0.59–0.69 | yes |

- Strongest null baseline (feature-availability count, no biology): AUC 0.513.
- Raw brief-summary character count predicts the label at AUC 0.638 — higher
  than the model achieves on the temporal split.
- The signal that survives grouping is **genomic** (COSMIC / DepMap / GWAS /
  somatic), not literature: literature-only features give AUC 0.535 (p=0.15,
  not significant) under grouped CV.

An earlier version of this README claimed "AUC ~0.87 on balanced labels" and
"27+ model configurations". Neither is supported: `reports/model_comparison.csv`
tops out at 0.633, and its 45 rows contain 9 distinct results (the three label
columns are identical, and `hybrid` duplicates `composite`).

## Notes

- No API keys required (all public APIs)
- All API results are cached locally in `mechanistic_engine_output/cache/`
- Second runs are fast because of caching
- This is a biology-only model; it intentionally ignores protocol design and site execution
