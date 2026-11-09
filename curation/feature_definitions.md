# Biological feature definitions (Week 1)

Owner: Arabella Chua · Draft for agreement with Tianze · Pipeline: v9 (`config.FEATURE_REGISTRY`)

## Gap these features fill

The v9 registry measures whether the target is mutated, amplified, deleted, essential in
cell lines, broadly or specifically expressed, well connected, chemically tractable and
well studied. It does **not** measure three things a translational biologist would check
first:

1. Is the target **rearranged** in this cancer type? Many of the most successful targeted
   agents act on fusions (ALK, ROS1, RET, NTRK), and the registry has no fusion feature.
2. When the target is mutated, are the mutations **recurrent activating hotspots** or
   scattered changes? Hotspots point to an oncogenic driver, while scattered or truncating
   mutations point to a passenger or tumour suppressor. `somatic_mut_freq` cannot tell
   these apart.
3. Is the target actually **expressed in this tumour type**? This matters most for
   antibodies and ADCs. The HPA features are target-level and pan-tissue, not specific
   to the trial's disease.

All three are pair-level (target × cancer type), INTRINSIC, and come from the TCGA
PanCancer Atlas through the cBioPortal client the pipeline already uses. None encodes
clinical development history.

**Recommended for Week 2: Feature A (fusion frequency).** It needs the smallest code change:
one more cBioPortal molecular profile, aggregated the same way as `cna_amp_freq`. It also
fills the clearest biological blind spot.

---

## Feature A: `sv_fusion_freq`

| Item | Definition |
|---|---|
| Domain / leakage class / level | genetic / INTRINSIC / pair |
| Biological question | Is the drug's target recurrently rearranged in the trial's cancer type? |
| Source | cBioPortal, TCGA PanCancer Atlas 2018, structural-variant profile of the study for the trial's cancer type (`<study>_structural_variants`) |
| Extraction rule | Number of profiled samples with ≥1 structural variant in which the target is either partner gene (`site1HugoSymbol` or `site2HugoSymbol`) and the event is in-frame or of unknown frame, **divided by** the number of samples with structural-variant profiling for that study |
| Multi-target drugs | Use the **maximum** over targets, not the mean (`GENE_LEVEL` currently averages). A multikinase inhibitor active on ALK should not have its ALK signal diluted by its other targets. Report a sensitivity analysis with the mean for consistency with the other features |
| Missing values | NaN if the study has no structural-variant profile, the target cannot be mapped to an Entrez ID, or the trial's disease has no TCGA study. **0 only when the profile exists and has no event**. Add `sv_fusion_freq_missing` (0/1) so missing is not read as "never fused" |
| Negation | Not applicable (structured data, no text) |
| Exclusion | Out-of-frame fusions; events where both partners are the same gene (intragenic deletions); samples not profiled |
| Historical availability | TCGA data were generated after many training trials ran (same caveat as `somatic_mut_freq`). It measures tumour biology, not clinical outcome, so leakage class INTRINSIC is appropriate. State this in Limitations |
| Expected pattern (to check, not assume) | High for ALK, ROS1 and RET in lung adenocarcinoma; near zero for most kinase targets in most tumours. Values will be small and sparse |
| Known limitation | TCGA under-samples rare fusion-defined subtypes, so true prevalence may be underestimated |

## Feature B: `mut_hotspot_frac`

| Item | Definition |
|---|---|
| Domain / leakage class / level | genetic / INTRINSIC / pair |
| Biological question | When the target is mutated in this cancer type, are the mutations recurrent activating hotspots (an oncogenic driver) or scattered (passenger or loss of function)? |
| Source | The cBioPortal mutation calls `somatic_mut_freq` already uses (`proteinChange`, `mutationType`) |
| Hotspot definition | A protein position in the target mutated (missense or in-frame indel) in **≥3 samples across all TCGA PanCancer studies**. Defined from the same TCGA data, so there is no external, undated hotspot list |
| Extraction rule | Among samples of the trial's cancer type with a non-silent mutation in the target: fraction whose mutation falls at a hotspot position |
| Companion feature | `mut_truncating_frac`: fraction of mutant samples with nonsense, frameshift or splice-site mutations (a tumour-suppressor pattern) |
| Multi-target drugs | Maximum over targets (as for A) |
| Missing values | **NaN when no sample in that cancer type carries a target mutation**, because the fraction is undefined. Do not set it to 0. Add `mut_hotspot_frac_missing` |
| Negation | Not applicable |
| Exclusion | Silent, intronic and UTR mutations; hypermutated samples (above the 99th percentile of TMB in that study) as a sensitivity analysis, since they produce non-recurrent passenger hits |
| Historical availability | As A (INTRINSIC, measured after many trials) |
| Expected pattern (to check) | High for BRAF in melanoma (V600), KRAS in pancreatic cancer, EGFR in lung adenocarcinoma; low for large or tumour-suppressor genes |
| Interpretation guard | Low values do not mean the target is irrelevant. Inhibiting an amplified or overexpressed wild-type target can still work |

## Feature C: `tumor_target_expr`

| Item | Definition |
|---|---|
| Domain / leakage class / level | expression / INTRINSIC / pair |
| Biological question | Is the target expressed in the tumour type the trial enrolls? |
| Source | cBioPortal TCGA PanCancer mRNA profile (`<study>_rna_seq_v2_mrna`, RSEM) |
| Extraction rule | `log2(median RSEM + 1)` of the target across samples of the trial's cancer type. Companion: `tumor_target_expr_frac` = fraction of samples above the pan-TCGA median for that gene |
| Multi-target drugs | Mean over targets for small molecules. For antibodies and ADCs, use the antigen (the first listed target) |
| Missing values | NaN if there is no mRNA profile or the gene is not quantified. Add a `_missing` flag |
| Negation | Not applicable |
| Exclusion | Normal-tissue samples; non-primary-tumour samples where the study labels them |
| Historical availability | As A |
| Expected pattern (to check) | High ERBB2 in breast cancer (driven by the HER2-amplified subset), high MS4A1 (CD20) in B-cell lymphoma |
| Interpretation guard | This is bulk tumour RNA. It includes stroma and immune cells and is not protein or membrane abundance. Do not describe it as "target accessibility" without the `hpa_membrane` context |

---

## To agree with Tianze (Week 1 "together" items)

- **Outcome definition:** `label_primary` = the trial met its own pre-specified primary
  objective (see the `rules` sheet in `outcome_review.xlsx`). `label_transition` stays
  secondary.
- **Prediction time:** trial start date. Time-bounded features use knowledge up to
  `start_year − 1`, as `features.py` already does.
- **Feature for Week 2:** Feature A, with B as the next candidate if A is implemented early.
- **Aggregation rule:** use max (not mean) for genetic-alteration features of multi-target
  drugs, with mean as a sensitivity analysis.
- **Pilot:** confirm the IDs of the shared 10-trial pilot, and which 5–10 extra trials I
  review.
