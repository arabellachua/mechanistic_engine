# Week 2 biology notes

Owner: Arabella Chua · Covers: what the features capture, interpretations to avoid,
trial contexts to investigate, and where to focus the spot-check.

## 1. What the features capture (plain language)

| Domain | Features | What it measures | What it does **not** measure |
|---|---|---|---|
| Genetic / somatic | `somatic_mut_freq`, `cna_amp_freq`, `cna_homdel_freq` | How often the target is mutated, amplified or deleted in TCGA tumours of the trial's cancer type | Whether the alteration is activating (see proposed `mut_hotspot_frac`); fusions (see `sv_fusion_freq`); whether trial patients carry it |
| Genetic / germline | `gwas_cancer_hits_log`, `gwas_disease_hits_log` | Inherited-risk associations mapped to the target gene, published before trial start | Somatic dependence. GWAS loci are often mapped to the nearest gene, not necessarily the causal one |
| Functional dependency | `depmap_*` | How much cancer cell lines need the gene to grow (CRISPR knockout), overall and in the trial's lineage; `depmap_lineage_selectivity` asks whether the lineage needs it more than other cancers | Response in patients; effects that need a tumour microenvironment or immune system (so not meaningful for IO targets); drug-level inhibition, which is rarely complete like a knockout |
| Expression / window | `hpa_*` | How broadly or specifically the gene is expressed in normal tissues and cancers, and whether the protein is on the cell surface or secreted | Expression in this trial's tumour type (proposed `tumor_target_expr`); protein level in patients |
| Network / pathway | `string_*`, `reactome_pathways_log` | How connected the target is and how many pathways include it | Pathway activity in the tumour. These also partly track how well studied a gene is (more studied genes have more annotated interactions) |
| Chemical tractability | `chembl_*` | How many potent compounds existed before the trial, the best potency, and how long chemistry on the target had existed | Selectivity of the trial's own drug; whether those compounds reached the clinic (deliberately excluded) |
| Evidence maturity | `pubmed_*` | How much was published on the target, and on target + cancer type, before the trial | Whether that literature was positive. It is a measure of attention, not of validation |
| Disease context | `tcga_tmb_median_log`, `tcga_msi_high_frac`, `leukocyte_fraction` | Cancer-type averages of mutation burden, MSI and immune infiltration; most relevant to IO | Anything about the enrolled patients or the target |
| Drug | `modality_*`, `n_targets` | Molecule type and polypharmacology | Potency, dose or exposure |

## 2. Interpretations not supported by these data

Use these as guardrails for the Results and Discussion.

1. **Elastic Net coefficients are conditional associations, not drivers.** A positive
   coefficient means the feature is associated with success *given the other features in
   the model*. Correlated features split weight between them. For example,
   `pubmed_target_log` and `string_degree_log` both rise with how well studied a gene is,
   so either may carry the signal.
2. **`depmap_common_essential_frac` is not a toxicity measure.** The registry description
   calls it "on-target toxicity risk", but it is cell-line essentiality. Describe it as "pan-essential in vitro".
3. **Disease-context features describe cancer types, not patients.** A high
   `leukocyte_fraction` means that cancer type is immune-infiltrated on average. It does
   not mean the trial's patients were.
4. **Evidence-maturity features carry the familiarity confound found in v8.** If
   `pubmed_*` or `chembl_history_years` dominate, the model may be learning "well-studied
   target", not "good biology". Run the planned evidence-domain ablation before making any
   biological claim.
5. **INTRINSIC features were measured after many training trials** (TCGA 2018, DepMap
   24Q2, HPA 23). They describe tumour biology, but the measurements were not available to
   trialists at the time. This belongs in Limitations, not in the leakage class.
6. **A low genetic-alteration frequency does not mean the target is irrelevant.**
   Antibodies, ADCs and hormonal agents often act on wild-type, overexpressed targets.
7. **Trial-design features (stage 2) are not biology.** Biomarker selection, randomisation and
   endpoint are context. Keep them out of the primary biology model, as the plan requires.

## 3. Trial contexts to investigate (proposed subgroups)

Each is a biological question. Minimum support: report as "not analysable" when a stratum
has fewer positives than `MIN_TEST_POSITIVES` (10).

| Context | Variable | Question |
|---|---|---|
| Therapy class | `stratum` (targeted, io_checkpoint, cytotoxic, hormonal) | Do the same biology features predict success across mechanisms, or does IO need disease-context features while targeted agents need genetic or dependency ones? |
| Biomarker-defined enrollment | `biomarker_selected`, `biomarker_matches_target` | When trials enroll only patients with the target alteration, do genetic features (measured in unselected TCGA populations) lose relevance? Expect a weaker `somatic_mut_freq` effect in matched trials |
| Modality | `modality_*` | Do expression or membrane features matter for antibodies and ADCs but not small molecules? |
| Target class | Kinase vs non-kinase (from ChEMBL target class) | Kinase targets are well studied and chemically tractable, which is the familiarity confound. Does the model still discriminate within kinases? |
| Cancer type | `disease` (lung, breast, colorectal, haematological vs solid) | Is performance driven by one or two well-represented cancers? |
| Seen vs novel drug | Drug present in training years or not | Does the model work for first-in-class targets, the case that matters most for portfolio decisions? |
| Era | `start_year` (pre-2015 vs post-2015) | Same question as the v8 era mismatch: chemotherapy-heavy training vs IO-heavy testing |

## 4. Spot-check priorities

These are the extraction rules most likely to fail, based on reading `normalize.py` and
`features.py`. Treat them as hypotheses to check in `spotcheck.xlsx`, not as confirmed errors.

- **Biomarker found in exclusion criteria.** `inclusion_text()` cuts the eligibility text
  at the phrase "exclusion criteria". Records that use a heading such as "Exclusion:" or
  "Key exclusion" will not be cut, so a biomarker listed as an *exclusion* could set
  `biomarker_selected = 1`.
- **Negation.** Phrases like "EGFR mutation-negative" or "no known ALK fusion": check that
  these do not count as selection.
- **ER/PR wording.** The ER pattern expects "ER positive" or "ER+". Check records written
  as "ER/PR-positive" or "HR+/HER2−".
- **Endpoint flags.** `endpoint_response` matches any "response", so check that "duration
  of response" or "pathologic complete response" is classified as intended. Records can
  have several primary outcomes, so more than one flag can be set.
- **Investigational agent vs backbone.** Check that standard chemotherapy backbones are
  not counted as investigational, and that `is_combination` reflects the experimental arm.
- **Disease mapping.** Check that basket trials and "solid tumours" trials are mapped
  sensibly, or excluded.

Fill `correct` / `correct_value` / `note`, then run `python curation_tools.py quality` to
produce `feature_quality_table.csv`.
