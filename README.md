# Mechanistic Trial Biology Engine — v9

Predicts whether an oncology Phase 2 or Phase 3 trial meets its objective from the
biology of the investigational agent's target in the trial's cancer type. v9 is a
rewrite of v6–v8 focused on validity: trial-specific labels, no outcome-proxy features,
time-bounded knowledge, a locked prospective holdout, and evaluation against baselines.

## Quick start

```bash
pip install -r requirements.txt
export NCBI_API_KEY=...            # strongly recommended (10 req/s instead of 3)
export NCBI_EMAIL=you@institution.edu
export MPE_OUTPUT=./mpe_output     # all outputs go here

python run_pipeline.py             # development run; holdout stays locked
python run_pipeline.py --no-pubmed --no-publications   # fast iteration
python tests/test_smoke.py         # offline smoke test on synthetic data
```

First run downloads a frozen CT.gov snapshot and the pinned external releases (the
ChEMBL SQLite dump is several GB). Before that first run:

1. Verify every URL and release name in `config.py` (`RELEASES`, `URLS`). Hosts
   reorganise paths between releases.
2. Optionally place licence-restricted files in `raw/`: `cancer_gene_census.csv`
   (COSMIC, registration required) and `thorsson_2018_immune_landscape.tsv`.
3. Fix `DATA_FREEZE_DATE` and the primary analysis in `config.py`. Do not change them
   after the holdout has been unlocked.

## Confirmatory run

```bash
python run_pipeline.py --unlock-holdout
```

Run this once, after pre-registering the analysis plan. Every holdout access is written
to `locks/holdout_access.log` with a hash of the analysis specification; re-access with
a different specification is flagged as post hoc.

## Module map (old → new)

| v6–v8 file | v9 file | What changed |
|---|---|---|
| `config.py` | `config.py` | Pre-specified scope, temporal design, pinned releases, feature registry with leakage classes |
| `clients.py` + `clients_v8_additions.py` | `clients.py` | Hashed cache, retries, no page cap, XML PubMed parsing, time-bounded counts, real cBioPortal endpoints |
| curated dicts in `clients*.py` | `data_sources.py` (new) | Versioned bulk downloads, sha256 manifest, no invented values |
| `drug_target_db.py` | `drug_target_db.py` + `drug_mapping.py` (new) | Mechanism-only seed; ChEMBL lexicon; whole-token matching; arm-aware investigational agent |
| `normalize.py` + `run_pipeline.normalize_trials` | `normalize.py` | One normaliser; word-boundary biomarker regex on inclusion criteria only; treatment-purpose filter |
| `labels.py` + `labels_v8.py` | `labels.py` | Trial-specific evidence tiers; no approval-based labels; Phase 3 transition label |
| `features.py` | `features.py` | Raw registry features only; time-bounded knowledge; disease-level IO context |
| — | `leakage.py` (new) | Feature-class enforcement and audits |
| — | `splits.py` (new) | Temporal split with embargo, frozen holdout, drug-grouped CV |
| `modeling.py` + `modeling_v8.py` | `modeling.py` | CV-based model selection, dynamic weights, temporal calibration, two-stage model |
| — | `evaluation.py` (new) | Cluster bootstrap CIs, calibration, decision curves, baselines, subgroups |
| `counterfactual.py` | `interpretation.py` | SHAP domain attribution + leave-one-domain-out ablation |
| `reporting.py` | `reporting.py` | Flow counts, label agreement, audit sample, data dictionary, model cards, TRIPOD+AI skeleton |
| `run_pipeline.py` | `run_pipeline.py` | CLI orchestration |

## Key outputs

| File | Use in the paper |
|---|---|
| `reports/flow_counts.csv` | Participant-flow diagram |
| `reports/label_provenance.csv`, `label_agreement.csv` | Outcome definition and label reliability |
| `data/label_audit_sample.csv` | Give to two blinded curators → `data/gold_labels.csv` |
| `reports/data_dictionary.csv` | Predictor table (with leakage class) |
| `reports/<analysis>_metrics.csv` | Discrimination, calibration, Brier with 95% CIs |
| `reports/<analysis>_comparisons.csv` | ΔAUC vs each baseline and stage 2 vs stage 1 |
| `reports/<analysis>_decision_curve.csv`, `figures/*_evaluation.png` | ROC, calibration, decision curves |
| `reports/<analysis>_subgroups.csv` | Stratum, disease, modality, seen vs novel drug |
| `reports/<analysis>_domain_importance.csv`, `_domain_ablation.csv` | Which biology domains carry signal |
| `reports/model_card_<analysis>.json` | Features, params, data manifest hash, holdout status |
| `raw/manifest.json` | Exact data versions (cite / deposit) |
| `reports/TRIPOD_AI_checklist.md` | Reporting checklist skeleton |

## Gold-standard labels

`data/gold_labels.csv` columns: `nct_id,label,source` (label 0/1). Gold labels override
all automated tiers. Build them by dual blinded curation of `label_audit_sample.csv`,
report Cohen's kappa, and adjudicate disagreements.

## Publication checklist

- [ ] Analysis plan pre-registered (OSF) before `--unlock-holdout`
- [ ] Gold-standard label sample dual-curated; kappa reported
- [ ] External validation set (EU CTR/CTIS, ISRCTN or WHO ICTRP)
- [ ] Primary model beats the target-history baseline (paired CI excludes 0), or this is reported as a null result
- [ ] Mapping-bias audit, label noise, CT.gov outcome edits and undated-knowledge sensitivity reported as limitations
- [ ] Clinical oncologist co-author reviewed label rules, backbone list and strata
- [ ] TRIPOD+AI checklist completed
- [ ] Code, `requirements.lock`, `raw/manifest.json` archived with a Zenodo DOI; preprint posted

## Known limitations

- Drug coverage depends on ChEMBL `drug_mechanism`; unmapped trials are excluded and
  the mapping-bias audit quantifies how that shifts the population.
- "Intrinsic" biology (DepMap, HPA, TCGA) was measured after many training trials ran.
  It measures physical properties rather than clinical outcomes, but state this.
- CT.gov primary outcomes can be edited after registration. Use the record-history API
  to recover originally registered endpoints for the final analysis.
- Strata with fewer than `MIN_TRAIN_POSITIVES` / `MIN_TEST_POSITIVES` are skipped and
  should be reported as not analysable, not dropped silently.
