"""
config.py — Central configuration for Mechanistic Phase 2 Biology Engine v7.

v7 FIXES:
  - TEST_YEAR_CUTOFF moved from 2018 → 2015 to give more test trials (54 → ~150)
  - Unmapped drug penalty added to prevent model gaming
  - Composite feature set now includes key pair-level features
"""

import logging
import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR   = Path("mechanistic_engine_output")
CACHE_DIR  = BASE_DIR / "cache"
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR    = BASE_DIR / "figures"

for d in [BASE_DIR, CACHE_DIR, DATA_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── ClinicalTrials.gov ────────────────────────────────────────────────────────
MAX_PAGES   = 50
PAGE_SIZE   = 100

# ── Caching ───────────────────────────────────────────────────────────────────
CACHE_TTL_DAYS = 30
USE_CACHE      = True

# ── Modeling ─────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
# v7 FIX: moved from 2018 to 2015 — gives ~3x more test trials
# With cutoff 2018: only 54 test trials (7 positives) — too noisy to evaluate
# With cutoff 2015: ~150+ test trials — stable AUC estimates
TEST_YEAR_CUTOFF = 2015
CV_FOLDS         = 5

# ── Optional deep learning branch ────────────────────────────────────────────
# Disabled by default. This branch complements the tree baselines; it does not
# replace RandomForest/LightGBM and it reuses the same temporal split.
RUN_DEEP_EXPERIMENTS = os.getenv("RUN_DEEP_EXPERIMENTS", "0") == "1"
DEEP_LABEL_COL = os.getenv("DEEP_LABEL_COL", "label_permissive")
DEEP_FEATURE_SET = os.getenv("DEEP_FEATURE_SET", "composite")

DEEP_BATCH_SIZE = int(os.getenv("DEEP_BATCH_SIZE", "64"))
DEEP_EPOCHS = int(os.getenv("DEEP_EPOCHS", "100"))
DEEP_PATIENCE = int(os.getenv("DEEP_PATIENCE", "15"))
DEEP_LR = float(os.getenv("DEEP_LR", "0.001"))
DEEP_WEIGHT_DECAY = float(os.getenv("DEEP_WEIGHT_DECAY", "0.0001"))

USE_PRETRAINED_EMBEDDINGS = os.getenv("USE_PRETRAINED_EMBEDDINGS", "0") == "1"

# ── Label construction ────────────────────────────────────────────────────────
POSITIVE_TEXT_TRIGGERS = [
    "met primary endpoint", "significant improvement", "overall survival benefit",
    "progression-free survival benefit", "objective response", "complete response",
    "partial response", "clinical benefit", "statistically significant",
    "superior to", "demonstrated efficacy", "positive results",
    "phase 3", "phase iii", "approval", "approved",
]
NEGATIVE_TEXT_TRIGGERS = [
    "did not meet", "failed to meet", "no significant", "no improvement",
    "no benefit", "futility", "lack of efficacy", "poor response",
    "terminated for efficacy", "negative results", "did not demonstrate",
    "no statistically significant", "not superior",
]
# v9 FIX: the bare substring "efficacy" was removed. It matched successful
# early-stopping ("stopped early due to overwhelming efficacy") the same as
# genuine failure ("terminated due to lack of efficacy") — see
# docs/label_audit.md. Every phrase left here is an explicit statement that
# the drug did NOT work; a bare "efficacy" mention proves nothing on its own.
TERMINATED_EFFICACY_REASONS = [
    "futility", "lack of efficacy", "poor efficacy", "no efficacy",
    "insufficient efficacy", "no response", "lack of response",
    "failed to demonstrate efficacy", "did not demonstrate efficacy",
    "lack of clinical efficacy", "inadequate efficacy",
]

# v9 ADDED: the mirror image of the above. "Terminated ... efficacy" is
# ambiguous by itself — a DSMB can stop a trial early either because the drug
# is failing OR because interim results are so strong that continuing to dose
# the control arm becomes unethical. These phrases identify the second case,
# which is a genuine positive outcome and was previously being scored negative
# by the bare "efficacy" substring match.
TERMINATED_SUCCESS_REASONS = [
    "stopped early due to efficacy", "stopped for efficacy",
    "early stopping due to efficacy", "efficacy at interim",
    "efficacy at the interim", "overwhelming efficacy",
    "met the primary endpoint at interim", "interim analysis showed efficacy",
    "efficacy established", "demonstrated superior efficacy",
]
LABELABLE_STATUSES = [
    "COMPLETED", "TERMINATED", "WITHDRAWN", "ACTIVE_NOT_RECRUITING",
]

# ── Feature groups ─────────────────────────────────────────────────────────────
PAIR_LEVEL_FEATURES = [
    "ot_genetic_association", "ot_somatic_mutation", "ot_known_drug",
    "ot_animal_model", "ot_rna_expression", "ot_literature", "ot_overall_score",
    "alteration_frequency", "co_alteration_burden", "genomic_complexity",
    "lineage_specificity", "gwas_disease_specificity",
    "pubmed_pair_count", "clinical_trial_pub_count", "pair_pub_acceleration",
]

TARGET_LEVEL_FEATURES = [
    "network_degree", "clustering_coefficient", "betweenness_centrality",
    "tractability", "binding_evidence", "potency_proxy",
    "target_class", "druggability_tier", "mechanistic_maturity",
    "pathway_count", "pathway_evidence", "alternative_pathway_burden",
    "gwas_association_count", "cosmic_census_member", "somatic_evidence",
    "polypharmacology_proxy", "depmap_chronos", "depmap_essential", "depmap_selective",
    "disgenet_score", "disgenet_disease_count",
    "known_drug_count", "interaction_score",
    "omnipath_degree", "omnipath_references",
]

BIOMARKER_FEATURES = [
    "biomarker_directness", "biomarker_prevalence", "biomarker_clonality",
    "assay_precision", "biomarker_missing", "cbio_data_missing", "ot_data_missing",
    "pubmed_data_missing",
]

TRIAL_FEATURES = [
    "modality_score", "human_study_fraction", "translational_study_fraction",
    "tumor_expression", "normal_tissue_burden", "tumor_specificity",
    "context_specificity", "expression_variability",
    "perturbation_concordance", "target_disease_assoc",
    "pub_acceleration", "literature_diversity",
    # v7: drug mapping flag
    "drug_is_mapped",
    # v8: IO-specific BIOLOGY features (target-level, not trial-level)
    # These describe the biology of the target, not the specific trial
    "tmb_relevance",        # how much TMB predicts response for this target
    "pdl1_expression",      # PD-L1 expression level (target biology)
    "immune_infiltration",  # immune cell density in relevant tumours
    "io_mechanism",         # does this target directly affect immune checkpoint
    "is_io_target",         # binary: is this an IO target
    # v8: split TDS (addresses biologic vs small molecule bias)
    "tds_small_molecule", "tds_biologic", "tds_combined", "drug_class",
    # NOTE: is_io_trial, endpoint_type_num, completed_no_results are NOT
    # included here because they are trial-level signals that correlate with
    # the permissive label (IO drugs dominate approved indications list),
    # which would inflate AUC artificially. They are only used in Stage 2.
]

ALL_RAW_FEATURES = PAIR_LEVEL_FEATURES + TARGET_LEVEL_FEATURES + BIOMARKER_FEATURES + TRIAL_FEATURES
COMPOSITE_SCORES = ["CDS", "TDS", "BFS", "MCS", "TWS", "EMS", "BIOLOGY_SCORE"]

print("config.py loaded successfully.")
