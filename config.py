"""
config.py — Central, pre-specified configuration for the Mechanistic Trial Biology Engine (v9).

Everything that can change the scientific result is declared here BEFORE the locked
holdout is evaluated:

  * study scope (phases, therapeutic strata, primary vs secondary analyses)
  * temporal design (train / calibration / embargo / test / locked holdout)
  * pinned external data releases (each download is hashed into raw/manifest.json)
  * FEATURE_REGISTRY: every feature declares a domain and a *leakage class*.
    Only classes listed in STAGE1_ALLOWED can reach the biology model. Anything
    derived from clinical-development outcomes (approvals, "known drug" scores,
    Open Targets' ChEMBL datasource, DGIdb drug counts) is class OUTCOME_PROXY
    and is blocked by leakage.assert_allowed().

Treat any edit made after the holdout has been unlocked as a protocol deviation.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get("MPE_OUTPUT", "mechanistic_engine_output"))
CACHE_DIR = BASE_DIR / "cache"      # hashed API response cache (never silently expires)
RAW_DIR = BASE_DIR / "raw"          # versioned bulk downloads + manifest.json
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"
LOCK_DIR = BASE_DIR / "locks"       # frozen holdout IDs/labels + access log
for _d in (BASE_DIR, CACHE_DIR, RAW_DIR, DATA_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR, LOCK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(BASE_DIR / "pipeline.log"), logging.StreamHandler()],
)
log = logging.getLogger("mpe")

RANDOM_SEED = 42
N_BOOTSTRAP = int(os.environ.get("MPE_N_BOOTSTRAP", 2000))

# ── Study scope ──────────────────────────────────────────────────────────────
PHASES = ("PHASE2", "PHASE3")          # Phase 2/3 seamless trials are grouped with PHASE3
EXCLUDE_PHASE1_2 = True                # Phase 1/2 trials are mostly dose-finding
STRATA = ("targeted", "io_checkpoint", "cytotoxic", "hormonal")

# The ONE analysis the paper's headline claim rests on. Pre-specify it.
PRIMARY_ANALYSIS = {"name": "primary_p2_targeted", "phase": "PHASE2",
                    "stratum": "targeted", "label": "label_primary"}
SECONDARY_ANALYSES = [
    {"name": "p2_io_checkpoint", "phase": "PHASE2", "stratum": "io_checkpoint", "label": "label_primary"},
    {"name": "p2_all_strata", "phase": "PHASE2", "stratum": "all", "label": "label_primary"},
    {"name": "p3_all_strata", "phase": "PHASE3", "stratum": "all", "label": "label_primary"},
    {"name": "p2_targeted_transition", "phase": "PHASE2", "stratum": "targeted", "label": "label_transition"},
]
MIN_TRAIN_POSITIVES = 20
MIN_TEST_POSITIVES = 10

# ── Temporal design (issue 4) ────────────────────────────────────────────────
# start_year <= TRAIN_END_YEAR           -> training
#   of which CALIB_YEARS                 -> probability calibration only
# EMBARGO_YEARS                          -> excluded (their outcomes overlap the test era)
# TEST_START_YEAR..TEST_END_YEAR         -> development test set (look as often as you like)
# start_year >= HOLDOUT_START_YEAR       -> LOCKED prospective holdout: evaluated once
DATA_FREEZE_DATE = "2026-09-01"        # set once; never change after locking the holdout
TRAIN_START_YEAR = 2005                # FDAAA-era registration; earlier records are sparse
TRAIN_END_YEAR = 2014
CALIB_YEARS = (2013, 2014)
EMBARGO_YEARS = (2015,)
TEST_START_YEAR = 2016
TEST_END_YEAR = 2019
HOLDOUT_START_YEAR = 2020
HOLDOUT_MIN_YEARS_SINCE_COMPLETION = 1.0
TRANSITION_WINDOW_YEARS = 4            # label_transition window after Phase 2 primary completion
CV_FOLDS = 5                           # grouped by investigational drug

# ── Labels (issue 1) ─────────────────────────────────────────────────────────
GOLD_LABELS_CSV = DATA_DIR / "gold_labels.csv"   # manually dual-curated labels (optional)
DEFAULT_ALPHA = 0.05
COUNT_SAFETY_STOP_AS_FAILURE = False
PUBLICATION_FETCH = "all"              # "all" enables tier-agreement analysis; "missing_only" is faster
LABEL_CONFIDENCE = {                   # used as sample weights
    "gold": 1.0,
    "structured_results": 0.9,
    "publication": 0.75,
    "termination_text": 0.7,
}

# ── APIs ─────────────────────────────────────────────────────────────────────
CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CBIOPORTAL_API = "https://www.cbioportal.org/api"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")

# ── Pinned external releases (issue 2) ───────────────────────────────────────
# Verify every URL before the first run; hosts reorganise paths between releases.
# Open Targets association scores are deliberately NOT used as features: their
# overall score includes the ChEMBL clinical-phase datasource (an outcome proxy).
RELEASES = {
    "depmap": "DepMap Public 24Q2",
    "chembl": "34",
    "hpa": "23",
    "string": "12.0",
    "reactome": "current",             # file is sha256-pinned in the manifest
    "gwas_catalog": "current",         # sha256-pinned; associations carry publication dates
    "cbioportal": "TCGA PanCancer Atlas 2018",
}
URLS = {
    "depmap_index": "https://depmap.org/portal/api/download/files",
    "chembl_sqlite": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
                     "chembl_{v}/chembl_{v}_sqlite.tar.gz",
    "hpa_tsv": "https://v{v}.proteinatlas.org/download/proteinatlas.tsv.zip",
    "string_links": "https://stringdb-downloads.org/download/protein.links.v{v}/"
                    "9606.protein.links.v{v}.txt.gz",
    "string_info": "https://stringdb-downloads.org/download/protein.info.v{v}/"
                   "9606.protein.info.v{v}.txt.gz",
    "reactome_uniprot": "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt",
    "gwas_associations": "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative",
}
# Licence-restricted sources: place the files here yourself (optional).
COSMIC_CGC_FILE = RAW_DIR / "cancer_gene_census.csv"            # needs COSMIC registration
THORSSON_FILE = RAW_DIR / "thorsson_2018_immune_landscape.tsv"   # Immunity 2018 supplement

# ── Feature registry (issues 2, 3, 6, 8) ─────────────────────────────────────
INTRINSIC = "intrinsic"                  # measured biology; does not encode clinical history
TIME_BOUNDED = "time_bounded"            # accumulating knowledge, computed strictly before trial start
UNDATED_KNOWLEDGE = "undated_knowledge"  # curated, undatable knowledge: sensitivity analysis only
DRUG_PROPERTY = "drug_property"          # properties of the molecule itself
TRIAL_DESIGN = "trial_design"            # stage-2 model only
OUTCOME_PROXY = "outcome_proxy"          # never allowed; listed so audits can catch them

STAGE1_ALLOWED = {INTRINSIC, TIME_BOUNDED, DRUG_PROPERTY}
STAGE1_SENSITIVITY_ALLOWED = STAGE1_ALLOWED | {UNDATED_KNOWLEDGE}
STAGE2_ALLOWED = STAGE1_ALLOWED | {TRIAL_DESIGN}


def _f(domain: str, leak: str, level: str, desc: str) -> dict:
    return {"domain": domain, "leakage_class": leak, "level": level, "description": desc}


FEATURE_REGISTRY: dict[str, dict] = {
    # Genetic / somatic causality
    "somatic_mut_freq": _f("genetic", INTRINSIC, "pair", "Fraction of TCGA samples of the trial's cancer type with a non-silent mutation in the target."),
    "cna_amp_freq": _f("genetic", INTRINSIC, "pair", "Fraction of TCGA samples with GISTIC high-level amplification of the target."),
    "cna_homdel_freq": _f("genetic", INTRINSIC, "pair", "Fraction of TCGA samples with GISTIC deep deletion of the target."),
    "gwas_cancer_hits_log": _f("genetic", TIME_BOUNDED, "target", "log1p genome-wide-significant GWAS associations mapping to the target for any cancer trait, published before trial start."),
    "gwas_disease_hits_log": _f("genetic", TIME_BOUNDED, "pair", "As above, restricted to traits matching the trial's cancer type."),
    "cgc_member": _f("genetic", UNDATED_KNOWLEDGE, "target", "COSMIC Cancer Gene Census membership (undated; sensitivity analysis only)."),
    # Functional dependency
    "depmap_pan_effect": _f("dependency", INTRINSIC, "target", "Mean CRISPR (Chronos) gene effect across all DepMap models."),
    "depmap_common_essential_frac": _f("dependency", INTRINSIC, "target", "Fraction of all DepMap models with effect < -0.5 (on-target toxicity risk)."),
    "depmap_lineage_effect": _f("dependency", INTRINSIC, "pair", "Mean gene effect in DepMap models of the trial's lineage."),
    "depmap_lineage_dependent_frac": _f("dependency", INTRINSIC, "pair", "Fraction of lineage models with effect < -0.5."),
    "depmap_lineage_selectivity": _f("dependency", INTRINSIC, "pair", "Pan-cancer effect minus lineage effect (positive = more dependent in lineage)."),
    # Expression / therapeutic window
    "hpa_tissue_distribution": _f("expression", INTRINSIC, "target", "HPA RNA tissue distribution breadth (0 not detected .. 4 detected in all)."),
    "hpa_tissue_specificity": _f("expression", INTRINSIC, "target", "HPA RNA tissue specificity (0 low .. 3 tissue enriched)."),
    "hpa_cancer_specificity": _f("expression", INTRINSIC, "target", "HPA RNA cancer specificity (0 low .. 3 cancer enriched)."),
    "hpa_membrane": _f("expression", INTRINSIC, "target", "Predicted membrane protein or plasma-membrane localisation (antibody accessibility)."),
    "hpa_secreted": _f("expression", INTRINSIC, "target", "Predicted secreted protein."),
    # Network / pathway context
    "string_degree_log": _f("network", INTRINSIC, "target", "log1p STRING partners with combined score >= 700."),
    "string_weighted_degree_log": _f("network", INTRINSIC, "target", "log1p sum of STRING confidence over high-confidence partners."),
    "reactome_pathways_log": _f("network", INTRINSIC, "target", "log1p Reactome pathways (all levels) containing the target."),
    # Chemical tractability (time-bounded)
    "chembl_potent_compounds_log": _f("chemistry", TIME_BOUNDED, "target", "log1p distinct compounds with pChEMBL >= 6 on the target, from documents published before trial start."),
    "chembl_max_pchembl": _f("chemistry", TIME_BOUNDED, "target", "Maximum pChEMBL reported before trial start."),
    "chembl_history_years": _f("chemistry", TIME_BOUNDED, "target", "Years between first potent compound report and trial start."),
    # Evidence maturity (time-bounded)
    "pubmed_target_log": _f("evidence", TIME_BOUNDED, "target", "log1p PubMed records mentioning the target, published before trial start."),
    "pubmed_pair_log": _f("evidence", TIME_BOUNDED, "pair", "log1p PubMed records mentioning target AND cancer type, before trial start."),
    "pubmed_pair_recent_frac": _f("evidence", TIME_BOUNDED, "pair", "Share of the pair's pre-start literature published in the 3 years before start."),
    # Disease immune / genomic context (key for IO; issue 8)
    "tcga_tmb_median_log": _f("disease_context", INTRINSIC, "disease", "log1p median nonsynonymous TMB of the cancer type in TCGA."),
    "tcga_msi_high_frac": _f("disease_context", INTRINSIC, "disease", "Fraction of TCGA samples with MANTIS score > 0.4 (MSI-high)."),
    "leukocyte_fraction": _f("disease_context", INTRINSIC, "disease", "Mean TCGA leukocyte fraction of the cancer type (Thorsson 2018)."),
    # Drug properties
    "modality_small_molecule": _f("drug", DRUG_PROPERTY, "drug", "Investigational agent is a small molecule."),
    "modality_antibody": _f("drug", DRUG_PROPERTY, "drug", "Investigational agent is a monoclonal/bispecific antibody."),
    "modality_adc": _f("drug", DRUG_PROPERTY, "drug", "Investigational agent is an antibody-drug conjugate."),
    "modality_other": _f("drug", DRUG_PROPERTY, "drug", "Other modality (protein, cell, oligonucleotide, ...)."),
    "n_targets": _f("drug", DRUG_PROPERTY, "drug", "Number of annotated molecular targets (polypharmacology)."),
    # Trial design (stage 2 only)
    "biomarker_selected": _f("design", TRIAL_DESIGN, "trial", "Inclusion criteria or title require a molecular biomarker."),
    "biomarker_matches_target": _f("design", TRIAL_DESIGN, "trial", "Selecting biomarker gene equals a drug target (e.g. EGFR-mutant for an EGFR TKI)."),
    "is_randomized": _f("design", TRIAL_DESIGN, "trial", "Randomised allocation."),
    "placebo_controlled": _f("design", TRIAL_DESIGN, "trial", "Has a placebo comparator arm."),
    "n_arms": _f("design", TRIAL_DESIGN, "trial", "Number of arms."),
    "is_combination": _f("design", TRIAL_DESIGN, "trial", "More than one active agent."),
    "n_investigational_agents": _f("design", TRIAL_DESIGN, "trial", "Number of non-backbone experimental agents."),
    "endpoint_os": _f("design", TRIAL_DESIGN, "trial", "Primary endpoint is overall survival."),
    "endpoint_pfs": _f("design", TRIAL_DESIGN, "trial", "Primary endpoint is PFS/EFS/DFS."),
    "endpoint_response": _f("design", TRIAL_DESIGN, "trial", "Primary endpoint is response/remission/pCR."),
    "industry_sponsor": _f("design", TRIAL_DESIGN, "trial", "Lead sponsor class is INDUSTRY."),
    # Documented outcome proxies: must never appear in a model
    "ot_overall_score": _f("forbidden", OUTCOME_PROXY, "pair", "Open Targets overall score (includes ChEMBL clinical phase)."),
    "ot_known_drug": _f("forbidden", OUTCOME_PROXY, "pair", "Open Targets known-drug datasource (clinical phase reached)."),
    "known_drug_count": _f("forbidden", OUTCOME_PROXY, "target", "Number of approved/clinical drugs on target."),
    "interaction_score": _f("forbidden", OUTCOME_PROXY, "target", "DGIdb drug interaction score."),
    "drug_is_mapped": _f("forbidden", OUTCOME_PROXY, "trial", "Dictionary coverage flag (famous-drug proxy)."),
    "approved_in_indication": _f("forbidden", OUTCOME_PROXY, "pair", "Regulatory approval status."),
    "enrollment_actual": _f("forbidden", OUTCOME_PROXY, "trial", "Actual enrollment (shrinks when a trial stops early)."),
    "has_results": _f("forbidden", OUTCOME_PROXY, "trial", "Results posted flag."),
}
FORBIDDEN_NAME_PATTERNS = ("approved", "known_drug", "ot_overall", "max_phase", "dgidb",
                           "interaction_score", "drug_is_mapped", "has_results", "results_posted",
                           "enrollment", "why_stopped", "label_")

DOMAIN_LABELS = {
    "genetic": "Genetic/somatic causality", "dependency": "Functional dependency",
    "expression": "Expression/therapeutic window", "network": "Network/pathway context",
    "chemistry": "Chemical tractability", "evidence": "Evidence maturity",
    "disease_context": "Disease immune/genomic context", "drug": "Drug properties",
    "design": "Trial design",
}
