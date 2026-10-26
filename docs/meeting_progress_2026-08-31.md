# Mechanistic Phase 2 Biology Engine

## Detailed progress memo for the August 31, 2026 meeting

## 1. Executive summary

I completed a clean rerun and a targeted technical audit of the current pipeline. The main result is stable rather than dramatically improved: the best held-out temporal ROC-AUC is approximately **0.67 under the permissive label** and **0.685 under the strict label**. This stability is useful because the code corrections did not manufacture a performance gain.

The audit identified and corrected four concrete implementation problems:

1. The raw and hybrid feature sets were not genuinely distinct from the composite feature set.
2. A duplicated `drug_is_mapped` column caused the two-stage model to fail.
3. Endpoint classification used only a narrow text field, causing nearly all endpoints to be classified as “other.”
4. PubMed requests were sent in bursts and did not retry HTTP 429 responses safely.

After these fixes:

- the composite, raw, and hybrid experiments use 33, 69, and 76 distinct features, respectively;
- the two-stage model runs successfully;
- OS, PFS, response, and biomarker endpoints are recognized from the full primary-outcome text;
- a live PubMed smoke test succeeds without a rate-limit error; and
- five regression tests pass.

The scientific interpretation is becoming clearer. The current evidence does **not** support a paper centered on a more complex model. It supports a paper centered on a more useful question:

> **When, and under what evaluation conditions, does public mechanistic biology generalize to future Phase 2 oncology trial outcomes?**

This framing turns the endpoint heterogeneity, data-source coverage, and missingness problems into explicit research questions rather than hiding them as engineering details.

## 2. What was rerun

### 2.1 Cohort construction

The clean run started from 5,000 ClinicalTrials.gov records and retained Phase 2 oncology trials according to the existing repository filters.

| Cohort step | Number of trials |
|---|---:|
| ClinicalTrials.gov records retrieved | 5,000 |
| Phase 2 oncology trials retained | 4,274 |
| Trials with determinate permissive labels | 832 |
| Training trials, start year ≤ 2015 | 617 |
| Test trials, start year > 2015 | 215 |

The split is temporal and uses **2015 as the cutoff**. The model is trained on determinate trials that started in or before 2015 and evaluated on determinate trials that started after 2015.

### 2.2 Current outcome labels

The labels are constructed from trial outcome-related information rather than from the biological features used by the model. Signals include:

- completion status;
- explicit positive or negative language in the trial record;
- whether results were posted;
- efficacy-related termination language; and
- for the permissive definition, whether a completed drug–indication pair appears in the current approved-indication mapping.

The value `-1` denotes an indeterminate trial, which is excluded from model training and evaluation.

Important clarification: in the current implementation, `label_balanced` is assigned directly from `label_strict`. Therefore, strict and balanced are currently aliases, not two independent outcome definitions. Their identical performance is expected and should not be presented as replication across separate labels.

Before the paper analysis, we should either:

1. define a scientifically distinct balanced label with explicit rules, or
2. remove the balanced label and report only strict and permissive analyses.

### 2.3 Feature groups

The pipeline integrates public mechanistic evidence at the drug–target, target, disease, and trial levels. Current feature families include:

- target–disease evidence from Open Targets;
- drug–target activity and tractability evidence;
- pathway, network, genetic, and somatic evidence;
- disease dependency and expression-related evidence;
- publication counts and publication acceleration;
- trial design and modality descriptors;
- biomarker and missingness indicators; and
- engineered composite scores summarizing orthogonal biological dimensions.

After the correction, the three comparison sets are:

| Feature set | Definition | Feature count |
|---|---|---:|
| Composite | Engineered mechanistic scores plus selected interaction and missingness terms | 33 |
| Raw | Source-level and trial-level variables, excluding the engineered composite scores | 69 |
| Hybrid | Union of raw variables and engineered composite scores | 76 |

This makes the intended scientific comparison valid: whether aggregation into biologically motivated composite scores improves generalization over individual source variables, and whether combining both representations adds value.

## 3. Code corrections and why they matter

### 3.1 Distinct raw, composite, and hybrid experiments

Previously, overlapping columns made the feature-set comparison ambiguous, and the hybrid set did not provide a clean union of raw and composite evidence.

The feature builder now:

- removes columns already present in the normalized trial table before concatenation;
- excludes composite scores from the raw set; and
- constructs the hybrid set as the de-duplicated union of raw and composite features.

Scientific consequence: model performance can now be attributed to a defined representation rather than to accidental column duplication.

### 3.2 Two-stage model repair

The second-stage design uses the Stage 1 biological prediction together with selected context variables such as modality, biomarker availability, immuno-oncology status, drug class, and endpoint type.

The earlier feature matrix contained two columns named `drug_is_mapped`, which made downstream preprocessing fail. Stage 2 column selection is now explicitly de-duplicated.

Scientific consequence: the two-stage hypothesis can now be evaluated, although the current result does not show an improvement.

### 3.3 Endpoint extraction repair

The earlier code classified endpoints using a narrow outcome field. In the retrieved records, useful endpoint language often appears in the description or time-frame fields. This caused an implausible result in which nearly every trial was assigned to “other.”

The normalized endpoint text now combines:

- primary outcome measure;
- primary outcome description; and
- primary outcome time frame.

The classifier recognizes common full names and abbreviations, including:

- overall survival and OS;
- progression-free survival and PFS;
- objective response rate and ORR;
- disease control and clinical benefit; and
- biomarker or molecular endpoints.

Scientific consequence: endpoint-stratified analysis is now possible, and the results show meaningful heterogeneity.

### 3.4 PubMed reliability repair

The earlier NCBI workflow could issue several requests in a burst, sleep only afterward, and treat HTTP 429 as a terminal failure. The client now:

- paces each NCBI request;
- retries HTTP 429 and server-side 5xx errors;
- respects `Retry-After` when supplied; and
- otherwise uses capped exponential backoff.

A live EGFR–NSCLC pair query succeeded and returned real publication features. This validates the retry path at smoke-test scale, but it does not replace a complete refreshed PubMed run.

### 3.5 Regression protection

Five automated regression tests now cover:

1. hybrid/raw/composite feature-set distinction;
2. absence of duplicate normalized feature columns;
3. classification of common endpoint abbreviations;
4. uniqueness of Stage 2 columns; and
5. retry behavior after an HTTP 429 response.

## 4. Current model results

### 4.1 Main held-out results

| Label definition | Best model | Best feature set | Test ROC-AUC | Test PR-AUC | Brier score |
|---|---|---|---:|---:|---:|
| Strict | LightGBM | Composite | 0.685 | 0.372 | 0.197 |
| Balanced | LightGBM | Composite | 0.685 | 0.372 | 0.197 |
| Permissive | Gradient Boosting | Composite | 0.667 | 0.542 | 0.209 |

The strict and balanced rows are identical because the current code defines balanced as strict. They should be treated as one result until the label definitions are revised.

For the permissive analysis, five-fold cross-validation on the training data produced a mean ROC-AUC of **0.677 ± 0.065**.

### 4.2 Interpretation of the baseline

The result is above chance, but it is not yet sufficient for a strong predictive claim. A defensible interpretation is:

- public mechanistic evidence contains some signal associated with later trial outcomes;
- biologically engineered composite features are currently more useful than simply adding model complexity;
- performance is sensitive to how success is labeled and which endpoint is studied; and
- the next contribution should be rigorous evaluation of generalization and failure modes.

The current analysis does not yet establish prospective clinical utility.

### 4.3 Deep-learning branch

The exploratory deep-learning branch reused the same split and did not outperform the tree baselines. The best observed deep result was approximately **0.53 ROC-AUC**. At the current sample size and feature coverage, neural complexity is not justified as the main direction.

This is still useful negative evidence: the immediate bottleneck is more likely label quality, temporal validity, entity leakage, endpoint heterogeneity, and source coverage than insufficient model capacity.

## 5. Endpoint-specific findings

After correcting endpoint extraction, the eligible cohort is distributed as follows:

| Endpoint group | Eligible trials | Share of 4,274 eligible trials |
|---|---:|---:|
| Other | 1,853 | 43.4% |
| Response | 1,019 | 23.8% |
| Endpoint text missing/unclassified | 852 | 19.9% |
| PFS | 394 | 9.2% |
| Overall survival | 115 | 2.7% |
| Biomarker | 41 | 1.0% |

Endpoint-specific permissive-label models currently give:

| Endpoint group | Determinate labeled trials | Test n | Test positive rate | Test ROC-AUC | Interpretation |
|---|---:|---:|---:|---:|---|
| PFS | 120 | 35 | 17.1% | 0.759 | Promising but highly uncertain because of the small test set |
| Response | 236 | 84 | 34.5% | 0.443 | Current biology representation does not generalize for this subgroup |
| Overall survival | 27 | — | — | Not estimated | Sample is too small for a stable model |
| Other/missing | 441 | 87 | 40.2% | 0.685 | Moderate signal, but the category is heterogeneous |

The PFS–response contrast could become a meaningful paper result if it remains after:

- bootstrap confidence intervals;
- drug- and target-disjoint splitting;
- label audit;
- minimum-sample sensitivity analysis; and
- correction for subgroup multiplicity.

It should currently be described as a hypothesis-generating observation, not a confirmed biological conclusion.

## 6. Two-stage model findings

The repaired two-stage model produced:

| Model | Test ROC-AUC |
|---|---:|
| Stage 1 biology model | 0.663 |
| Stage 2 logistic regression | 0.598 |
| Stage 2 random forest | 0.660 |

The second stage does not improve on Stage 1. The random forest essentially recovers the baseline, while logistic regression is worse.

Possible interpretations to test:

1. The context features do not add information beyond the biological score.
2. The sample size is too small for stable conditional modeling.
3. Label noise obscures subgroup-specific effects.
4. The chosen Stage 2 variables or functional form are insufficient.

The appropriate next step is ablation and error analysis, not a larger Stage 2 architecture.

## 7. Most important methodological risks

### 7.1 Feature-time leakage

The label builder is circularity-free in the narrow sense that it does not use the biological model features to assign outcomes. However, a temporal train/test split alone does not guarantee prospective validity.

Some public databases are queried in their current state. Their present-day values may include evidence generated after the start, completion, or publication of an older trial. A model can therefore receive future biological knowledge even when the trials themselves are split by start year.

Before making a prospective prediction claim, each source should be classified as:

- historically timestamped and reconstructable as of the trial index date;
- static or plausibly stable biological annotation;
- current aggregate evidence with possible post-outcome contamination; or
- unknown temporal provenance.

If historical snapshots cannot be reconstructed, the paper should describe the analysis as retrospective association or prioritization rather than simulated prospective prediction.

### 7.2 Drug and target leakage

The temporal test set can contain drugs or targets observed in training. The model may therefore learn entity-specific success patterns rather than transferable biological relationships.

Required sensitivity analyses are:

- unseen-drug test split;
- unseen-target test split;
- unseen drug–disease pair split; and, if sample size permits,
- leave-one-cancer-type-out evaluation.

### 7.3 Label validity

Current labels combine explicit outcome text, posting behavior, termination reasons, and an approved-indication mapping. These signals have different reliability.

Specific concerns include:

- neutral result text can be misclassified as negative;
- a completed trial without posted results is not necessarily unsuccessful;
- efficacy, safety, and accrual terminations have different meanings;
- current approval status may occur long after the Phase 2 trial; and
- the hand-maintained approval map can be incomplete or temporally inconsistent.

A stratified manual audit should report precision by label provenance, not only overall agreement.

### 7.4 Missingness and source coverage

Missingness is not random. Older trials, uncommon drugs, and less-studied targets are more likely to have sparse evidence. Missingness indicators may themselves encode trial era or research popularity.

The paper should report coverage by:

- calendar period;
- cancer type;
- drug modality;
- drug mapping status;
- target mapping status; and
- outcome class.

### 7.5 cBioPortal

The current cBioPortal extraction produced usable coverage for **0 of 315 queried combinations**. Until the mapping or API logic is repaired and validated, these variables should be treated as unavailable. Zero-filled cBioPortal variables must not be interpreted as biological absence.

## 8. Proposed paper plan

### 8.1 Working title

**When Does Public Mechanistic Evidence Generalize to Phase 2 Oncology Trial Outcomes? A Leakage-Aware Retrospective Evaluation**

Alternative title:

**Evaluating the Generalizability of Public Mechanistic Evidence for Phase 2 Oncology Trial Outcome Modeling**

### 8.2 Central research question

Can publicly available drug–target–disease evidence provide reproducible, generalizable signal for Phase 2 oncology outcomes after controlling for label uncertainty, time, repeated entities, endpoint type, and source coverage?

### 8.3 Specific aims

#### Aim 1: Construct and validate a reproducible outcome cohort

- Define strict and permissive outcome labels with explicit provenance.
- Quantify indeterminate cases instead of silently dropping them.
- Manually audit a stratified sample of labels.
- Report cohort flow, class balance, and temporal distribution.

#### Aim 2: Test whether mechanistic evidence generalizes

- Compare metadata-only, raw biology, composite biology, and hybrid models.
- Use temporal evaluation as the primary split.
- Add drug-, target-, and drug–disease-disjoint sensitivity analyses.
- Report discrimination, calibration, uncertainty, and decision-oriented metrics.

#### Aim 3: Identify where the model works and fails

- Stratify by endpoint, disease, modality, biomarker status, and mapping coverage.
- Quantify missingness and source availability.
- Perform ablations by evidence source and biological dimension.
- Analyze false positives and false negatives with label provenance.

### 8.4 Testable hypotheses

1. Mechanistic composite features improve temporal generalization relative to trial metadata alone.
2. The hybrid model will not necessarily outperform the composite model because raw features add noise and missingness.
3. Performance differs materially by endpoint type.
4. Performance will decline on unseen drugs and targets, revealing how much of the temporal result is entity-specific.
5. Better-mapped trials will show higher apparent performance, creating a coverage-related selection effect.

### 8.5 Primary and sensitivity analyses

| Analysis component | Proposed primary choice | Sensitivity analyses |
|---|---|---|
| Outcome | Strict, manually audited label | Permissive label and provenance-weighted analysis |
| Split | Temporal cutoff at 2015 | Rolling cutoffs; unseen drug; unseen target; unseen pair |
| Main comparison | Metadata vs composite biology vs hybrid | Individual source ablations |
| Primary metric | ROC-AUC with bootstrap 95% CI | PR-AUC, Brier score, calibration slope/intercept |
| Model class | Regularized logistic regression and gradient-boosted trees | Random forest; deep model as secondary negative result |
| Subgroups | Endpoint type | Cancer type, modality, biomarker, mapping coverage |

The strict label is proposed as primary because it is more interpretable. The permissive label provides more samples but depends partly on the approved-indication map. This choice should be discussed with Professor Rao and Arabella.

### 8.6 Minimum paper figures and tables

#### Main figures

1. **Cohort and evidence-flow diagram:** retrieval, filtering, mapping, labeling, train/test split, and source coverage.
2. **Primary performance figure:** ROC/PR results with bootstrap confidence intervals across metadata, raw, composite, and hybrid sets.
3. **Generalization stress test:** temporal versus unseen-drug, unseen-target, and unseen-pair performance.
4. **Heterogeneity figure:** endpoint-specific performance with sample sizes and uncertainty intervals.
5. **Evidence and missingness figure:** source coverage over time and by outcome/modality.

#### Main tables

1. Cohort characteristics and label provenance.
2. Primary and sensitivity-analysis metrics.
3. Data-source definitions, timestamps, coverage, and leakage risk.
4. Error-analysis examples or subgroup performance.

## 9. Venue strategy

### 9.1 Recommended first target: JCO Clinical Cancer Informatics

JCO Clinical Cancer Informatics is the best fit if the manuscript emphasizes clinically relevant cancer informatics, interpretable evaluation, and implications for oncology trial prioritization. Its stated scope includes artificial intelligence, advanced analytics, virtual clinical trials, and mechanistic models applied to real-world cancer problems.

Fit requirements for this project:

- make the clinical oncology use case central;
- avoid presenting the work as only a benchmark;
- explain how the findings affect evidence assessment or trial prioritization; and
- include rigorous validation and transparent limitations.

Official scope: https://ascopubs.org/cci/about

### 9.2 Strong alternatives

| Venue | Best framing | Fit | Main concern |
|---|---|---|---|
| JAMIA / JAMIA Open | Reproducible biomedical informatics evaluation and data integration | Good if methodology and evaluation design are central | Must establish broad informatics significance beyond oncology |
| Bioinformatics Advances | Interdisciplinary computational biology, cancer, translational medicine, and data/text mining | Good realistic computational target | Needs a clear biological/computational contribution |
| Bioinformatics | Significant new bioinformatics method or broadly important biological insight | Stretch target | Current modeling may appear incremental without stronger methodology or external validation |
| AMIA or ISMB | Preliminary methodological results and community feedback | Useful conference route | Submission cycles and paper formats need separate planning |

Official guidance:

- JAMIA Open Research and Applications articles allow up to 4,000 words and encourage public code and data for computational work: https://academic.oup.com/jamiaopen/pages/General_Instructions
- Bioinformatics Advances explicitly covers algorithms, statistics, data/text mining, translational medicine, and cancer; Original Articles allow up to eight pages: https://academic.oup.com/bioinformaticsadvances/pages/author-guidelines
- Bioinformatics emphasizes significant advances in algorithms or databases and requires strong comparison with existing methods: https://academic.oup.com/bioinformatics/pages/scope_guidelines

### 9.3 Venue decision rule

- Choose **JCO Clinical Cancer Informatics** if the final contribution is a clinically grounded evaluation of mechanistic evidence for oncology trial prioritization.
- Choose **JAMIA/JAMIA Open** if the strongest contribution becomes the leakage-aware informatics evaluation framework.
- Choose **Bioinformatics Advances** if the strongest contribution becomes multi-source mechanistic data integration plus generalizable computational analysis.
- Consider **Bioinformatics** only if the method becomes substantially more novel or the study gains strong external validation and biological discovery.

## 10. Concrete work plan

### Before the next full modeling run

1. Freeze a cohort identifier list and record the retrieval date.
2. Decide whether strict or permissive is the primary outcome.
3. Remove or redefine the duplicate balanced label.
4. Create a source manifest containing version, retrieval time, coverage, and temporal leakage risk.
5. Repair cBioPortal or exclude it from the primary model.

### Week 1: data and evaluation integrity

- Complete a fully refreshed PubMed run with request-success statistics.
- Produce a coverage report for every external source.
- Audit 150–200 labels, stratified by class, provenance, endpoint, and trial era.
- Implement drug-, target-, and pair-disjoint split generators.
- Add bootstrap confidence intervals and calibration metrics.

Deliverable: a frozen analysis cohort, label-audit table, source-quality table, and validated split definitions.

### Week 2: paper-ready experiments

- Run the complete baseline matrix on all approved splits.
- Run evidence-source and composite-score ablations.
- Repeat endpoint-specific analyses with uncertainty intervals.
- Complete false-positive and false-negative review.
- Generate the cohort-flow figure, main performance figure, and primary results table.

Deliverable: paper-ready Results v1 and an outline with figure/table placeholders.

### Later validation

- Evaluate a second temporal cutoff or rolling-origin design.
- Seek an external or independently assembled validation cohort if feasible.
- Perform a reproducibility run from an empty cache in a pinned environment.
- Archive raw response metadata where licensing and source terms permit.

## 11. Decisions to request from Professor Rao and Arabella

1. Do we agree that the central contribution should be “when mechanistic biology generalizes,” rather than a claim that a complex model wins?
2. What should the prediction target mean scientifically: explicit Phase 2 efficacy, broader trial success, or eventual drug–indication success?
3. Should strict be the primary label and permissive a sensitivity analysis?
4. Should the balanced label be redefined or removed?
5. Is temporal evaluation sufficient for the first paper, or are unseen-drug and unseen-target tests mandatory for the main result?
6. How conservative should we be about present-day biological sources that cannot be reconstructed historically?
7. Should cBioPortal be excluded unless nonzero, validated coverage is recovered?
8. Which audience should drive the manuscript: clinical cancer informatics or computational bioinformatics?
9. Is JCO Clinical Cancer Informatics the preferred first target?

## 12. Suggested meeting presentation

### 30-second version

> I reran the current pipeline and completed a technical audit. I fixed feature duplication, made the raw, composite, and hybrid comparisons genuinely distinct, repaired endpoint classification, fixed the two-stage model, and added PubMed rate-limit handling and regression tests. The corrected headline performance is stable: about 0.67 AUC under the permissive label and 0.685 under the strict label. Endpoint analysis now suggests substantial heterogeneity: PFS is promising but based on only 35 test trials, while response endpoints perform poorly. The two-stage and deep models do not improve the biology baseline. I therefore propose framing the paper around when public mechanistic evidence generalizes, with leakage-resistant and entity-disjoint evaluation, rather than around model complexity.

### Two-minute version

> Since our last discussion, I first reproduced the pipeline on a clean output directory using the same 2015 temporal cutoff. The run retrieved 5,000 ClinicalTrials.gov records, retained 4,274 Phase 2 oncology trials, and produced 832 determinate permissive labels, split into 617 training and 215 post-2015 test trials.
>
> I then audited several suspicious results. The hybrid feature set was not a clean hybrid, a duplicated drug-mapping column caused the two-stage model to fail, endpoint classification was using too little of the outcome text, and PubMed requests did not handle rate limiting safely. I corrected those issues and added five regression tests.
>
> After correction, the main result remains about 0.67 AUC for the permissive label and 0.685 for the strict label. The best representation is still the engineered composite biology set. More complex approaches do not currently help: the two-stage random forest is about 0.66, and the preliminary deep model is near 0.53.
>
> The endpoint correction revealed a more interesting direction. The PFS subgroup reaches 0.759 AUC, but its test set has only 35 trials, whereas the response subgroup is below chance at 0.443. I do not want to overinterpret this yet; it needs confidence intervals, disjoint splits, and a label audit. However, it suggests that endpoint heterogeneity and evidence coverage may be the scientific story.
>
> My proposed paper question is: when does public mechanistic biology generalize to future Phase 2 oncology outcomes? The next work would focus on label validity, feature-time leakage, unseen-drug and unseen-target evaluation, source missingness, and endpoint-specific failure modes. I would like guidance on whether strict or permissive should be primary and whether we should aim first at JCO Clinical Cancer Informatics or frame it more methodologically for JAMIA or Bioinformatics Advances.

## 13. What is established versus still preliminary

### Established in the current code and validation

- The clean cohort and temporal split can be reproduced from the frozen ClinicalTrials.gov retrieval.
- The feature-set duplication and Stage 2 duplication bugs are fixed.
- Endpoint classification now identifies common oncology endpoint types.
- Five targeted regression tests pass.
- The corrected main model result remains approximately 0.67–0.685 ROC-AUC.
- The two-stage and deep branches do not currently improve the baseline.

### Preliminary or unresolved

- PFS performance is based on a small test set and has no bootstrap interval yet.
- Full PubMed refresh has not been completed after the pacing fix.
- cBioPortal currently has zero usable coverage.
- Feature-time leakage has not been fully audited.
- Drug-, target-, and pair-disjoint evaluations have not yet been run.
- Label reliability has not yet been manually estimated.
- The latest targeted validation reused the frozen clean cohort and feature matrix rather than re-downloading every external source.

These distinctions should be preserved in the meeting so that the progress is concrete without overstating the maturity of the scientific conclusion.
