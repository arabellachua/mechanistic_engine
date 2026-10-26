"""
features.py — Feature engineering for the mechanistic biology engine.

v6 CHANGES (professor feedback):
  1. PAIR-LEVEL FEATURES: Features now vary per (target, disease) combination.
     Every pembrolizumab trial no longer gets identical scores — the model now
     sees different alteration frequencies, OT scores, and PubMed counts for
     pembrolizumab+melanoma vs pembrolizumab+NSCLC vs pembrolizumab+gastric.

  2. MISSINGNESS INDICATORS: Instead of silently imputing median values for
     missing biomarker data (which hides the fact that 79% of trials have no
     biomarker), we now add explicit binary indicator columns:
       - biomarker_missing (1 = no biomarker identified)
       - cbio_data_missing  (1 = no cBioPortal data for this pair)
       - ot_data_missing    (1 = no Open Targets data for this pair)
     The model can then learn the relationship between missingness and outcomes,
     rather than treating imputed medians as real observations.

  3. DEPMAP: Added DepMap essentiality scores (Chronos) as target-level features.

  4. HPA FIXED: HPA features are now populated (Ensembl ID fix in clients.py),
     so tumor_expression / normal_tissue_burden will have real data.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from config import log

# ── Composite score weights ───────────────────────────────────────────────────

CDS_WEIGHTS = {
    "cosmic_census_member": 0.25,
    "somatic_evidence": 0.20,
    "ot_somatic_mutation": 0.15,
    "ot_genetic_association": 0.15,
    "gwas_association_count": 0.10,
    "gwas_disease_specificity": 0.10,
    "disgenet_score": 0.05,
}

TDS_WEIGHTS = {
    "tractability": 0.20,
    "binding_evidence": 0.15,
    "potency_proxy": 0.15,
    "ot_known_drug": 0.15,
    "ot_overall_score": 0.10,
    "network_degree": 0.10,
    "depmap_essential": 0.10,
    "mechanistic_maturity": 0.05,
}

BFS_WEIGHTS = {
    "alteration_frequency": 0.30,      # PAIR-LEVEL (cancer-type specific)
    "biomarker_directness": 0.20,
    "biomarker_prevalence": 0.15,
    "biomarker_clonality": 0.10,
    "assay_precision": 0.10,
    "lineage_specificity": 0.15,       # PAIR-LEVEL (how specific to this lineage)
}

MCS_WEIGHTS = {
    "mechanistic_maturity": 0.20,
    "perturbation_concordance": 0.20,
    "pathway_evidence": 0.15,
    "ot_literature": 0.15,
    "pubmed_pair_count": 0.15,         # PAIR-LEVEL
    "clinical_trial_pub_count": 0.15,  # PAIR-LEVEL
}

TWS_WEIGHTS = {
    "tumor_expression": 0.30,
    "tumor_specificity": 0.25,
    "normal_tissue_burden": 0.20,
    "context_specificity": 0.15,
    "ot_rna_expression": 0.10,        # PAIR-LEVEL
}

EMS_WEIGHTS = {
    "human_study_fraction": 0.25,
    "translational_study_fraction": 0.20,
    "pubmed_pair_count": 0.20,         # PAIR-LEVEL
    "pair_pub_acceleration": 0.20,     # PAIR-LEVEL
    "literature_diversity": 0.15,
}

def _safe(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default

def _weighted_sum(row: dict, weights: dict) -> float:
    total_w = sum(weights.values())
    score = sum(weights[k] * _safe(row.get(k, 0)) for k in weights)
    return score / total_w if total_w > 0 else 0.0

# ── Per-trial raw feature extraction ─────────────────────────────────────────

def compute_raw_features(trial: pd.Series, pair_data: dict, target_data: dict) -> dict:
    """
    Extract all raw features for a trial.

    pair_data: dict keyed by (target_symbol, disease) → {ot_score, pubmed_count, ...}
    target_data: dict keyed by target_symbol → {string_degree, chembl_tractability, ...}

    Key v6 change: pair_data lookup uses BOTH target AND disease,
    so different disease contexts get different feature values.
    """
    symbol = trial.get("primary_target", "UNKNOWN")
    disease = trial.get("disease", "unknown")
    modality = trial.get("modality", "unknown")
    has_biomarker = bool(trial.get("has_biomarker", False))

    # Look up data for this specific (target, disease, start_year) triple. The
    # year matters because the PubMed features are bounded at the trial's start
    # year; fall back to the year-agnostic key for callers that still use it.
    start_year = trial.get("start_year")
    pair = {}
    if start_year and not pd.isna(start_year):
        pair = pair_data.get((symbol, disease, int(start_year)), {})
    if not pair:
        pair = pair_data.get((symbol, disease), {})

    # Fall back to the target-level data
    tgt = target_data.get(symbol, {})

    features = {}

    # ── PAIR-LEVEL features (vary by target+disease combination) ──────────────
    features["ot_overall_score"]       = _safe(pair.get("ot_overall_score", 0))
    features["ot_genetic_association"] = _safe(pair.get("ot_genetic_association", 0))
    features["ot_somatic_mutation"]    = _safe(pair.get("ot_somatic_mutation", 0))
    features["ot_known_drug"]          = _safe(pair.get("ot_known_drug", 0))
    features["ot_animal_model"]        = _safe(pair.get("ot_animal_model", 0))
    features["ot_rna_expression"]      = _safe(pair.get("ot_rna_expression", 0))
    features["ot_literature"]          = _safe(pair.get("ot_literature", 0))
    features["pubmed_pair_count"]      = _safe(pair.get("pubmed_pair_count"), np.nan)
    features["clinical_trial_pub_count"] = _safe(pair.get("clinical_trial_pub_count"), np.nan)
    features["pair_pub_acceleration"]  = _safe(pair.get("pair_pub_acceleration"), np.nan)
    features["lineage_specificity"]    = _safe(pair.get("lineage_specificity", 0))
    features["gwas_disease_specificity"] = _safe(pair.get("gwas_disease_specificity",
                                               tgt.get("gwas_disease_specificity", 0)))

    # cBioPortal pair-level (cancer-type specific alteration frequency)
    has_cbio = bool(pair.get("has_real_cbio_data", False))
    if has_cbio:
        features["alteration_frequency"] = _safe(pair.get("alteration_frequency", 0))
        features["co_alteration_burden"] = _safe(pair.get("co_alteration_burden", 0))
        features["genomic_complexity"]   = _safe(pair.get("genomic_complexity", 0))
    else:
        # v6: Use NaN + indicator flag instead of median imputation
        features["alteration_frequency"] = np.nan
        features["co_alteration_burden"] = np.nan
        features["genomic_complexity"]   = np.nan

    # ── MISSINGNESS INDICATORS (v6: explicit flags instead of silent imputation) ──
    features["ot_data_missing"]   = 0 if pair.get("has_real_ot_data", False) else 1
    features["cbio_data_missing"] = 0 if has_cbio else 1
    features["pubmed_data_missing"] = 0 if pair.get("has_real_pubmed_data", False) else 1
    features["biomarker_missing"] = 0 if has_biomarker else 1

    # ── TARGET-LEVEL features (same for all trials with same target) ───────────
    # STRING network topology
    features["network_degree"]           = _safe(tgt.get("network_degree", 0))
    features["clustering_coefficient"]   = _safe(tgt.get("clustering_coefficient", 0))
    features["betweenness_centrality"]   = _safe(tgt.get("betweenness_centrality", 0))

    # ChEMBL druggability
    features["tractability"]   = _safe(tgt.get("tractability", 0))
    features["binding_evidence"] = _safe(tgt.get("binding_evidence", 0))
    features["potency_proxy"]    = _safe(tgt.get("potency_proxy", 0))

    # UniProt biology
    features["mechanistic_maturity"] = _safe(tgt.get("mechanistic_maturity", 0))
    features["druggability_tier"]    = _safe(tgt.get("druggability_tier", 0))
    features["target_class"]         = _safe(tgt.get("target_class", 0))

    # Reactome pathway membership
    features["pathway_count"]              = _safe(tgt.get("pathway_count", 0))
    features["pathway_evidence"]           = _safe(tgt.get("pathway_evidence", 0))
    features["alternative_pathway_burden"] = _safe(tgt.get("alternative_pathway_burden", 0))

    # GWAS (target-level)
    features["gwas_association_count"] = _safe(tgt.get("gwas_association_count", 0))

    # COSMIC
    features["cosmic_census_member"] = _safe(tgt.get("cosmic_census_member", 0))
    features["somatic_evidence"]     = _safe(tgt.get("somatic_evidence", 0))

    # DepMap essentiality (NEW in v6)
    features["depmap_chronos"]   = _safe(tgt.get("depmap_chronos", 0))
    features["depmap_essential"] = _safe(tgt.get("depmap_essential", 0))
    features["depmap_selective"] = _safe(tgt.get("depmap_selective", 0))

    # HPA expression (now populated with real data via Ensembl ID fix)
    features["tumor_expression"]    = _safe(tgt.get("tumor_expression", 0.4))
    features["normal_tissue_burden"] = _safe(tgt.get("normal_tissue_burden", 0.5))
    features["tumor_specificity"]   = _safe(tgt.get("tumor_specificity", 0.0))
    features["expression_variability"] = _safe(tgt.get("expression_variability", 0.5))

    # OmniPath
    features["omnipath_degree"]     = _safe(tgt.get("omnipath_degree", 0))
    features["omnipath_references"] = _safe(tgt.get("omnipath_references", 0))

    # DGIdb
    features["known_drug_count"]  = _safe(tgt.get("known_drug_count", 0))
    features["interaction_score"] = _safe(tgt.get("interaction_score", 0))

    # DisGeNET
    features["disgenet_score"]         = _safe(tgt.get("disgenet_score", 0))
    features["disgenet_disease_count"] = _safe(tgt.get("disgenet_disease_count", 0))

    # ── TRIAL-LEVEL features (derived from trial metadata) ────────────────────
    # Modality score
    modality_map = {
        "antibody": 0.8, "adc": 0.85, "bispecific": 0.85,
        "small_molecule": 0.7, "cell_therapy": 0.75,
        "gene_therapy": 0.7, "peptide": 0.6,
        "vaccine": 0.5, "unknown": 0.3,
    }
    features["modality_score"] = modality_map.get(modality.lower(), 0.3)

    # v7: explicit drug mapping flag
    features["drug_is_mapped"] = 0.0 if symbol == "UNKNOWN" else 1.0

    # v8: IO-specific features from clients_v8_additions
    try:
        from clients_v8_additions import get_io_features, compute_split_tds
        io_feats = get_io_features(symbol)
        tds_split = compute_split_tds(symbol, modality, features)
        features.update(io_feats)
        features.update(tds_split)
    except Exception:
        # Fallback if v8 additions not available
        features["tmb_relevance"] = 0.2
        features["pdl1_expression"] = 0.3
        features["immune_infiltration"] = 0.4
        features["io_mechanism"] = 0.0
        features["is_io_target"] = 0.0
        features["tds_small_molecule"] = features.get("tractability", 0)
        features["tds_biologic"] = features.get("tumor_expression", 0.4)
        features["tds_combined"] = features.get("tractability", 0)
        features["drug_class"] = 0.0

    # NOTE: is_io_trial, endpoint_type_num, completed_no_results are intentionally
    # excluded from Stage 1 biology features. They are trial-level signals that
    # correlate with permissive labels (IO drugs dominate approved indications),
    # which would inflate Stage 1 AUC artificially. They are passed to Stage 2 only.

    # Polypharmacology risk
    features["polypharmacology_proxy"] = 1.0 - features["tractability"]

    # Human study fraction (from PubMed pair data)
    if not np.isfinite(features["pubmed_pair_count"]) or not np.isfinite(features["clinical_trial_pub_count"]):
        features["human_study_fraction"] = np.nan
    elif features["pubmed_pair_count"] > 0:
        features["human_study_fraction"] = min(
            features["clinical_trial_pub_count"] / max(features["pubmed_pair_count"], 0.01),
            1.0
        )
    else:
        features["human_study_fraction"] = 0.0

    features["translational_study_fraction"] = features["human_study_fraction"] * 0.7
    features["perturbation_concordance"] = min(
        features["pathway_evidence"] * 0.6 + features["ot_overall_score"] * 0.4, 1.0
    )
    features["target_disease_assoc"] = features["ot_overall_score"]

    # Publication-derived features
    features["pub_acceleration"] = features["pair_pub_acceleration"]
    features["literature_diversity"] = min(
        features["pubmed_pair_count"] * 0.5 + features["clinical_trial_pub_count"] * 0.5, 1.0
    )

    # Context specificity: how specific the target-disease link is
    features["context_specificity"] = min(
        features["lineage_specificity"] * 0.5 +
        features["ot_genetic_association"] * 0.3 +
        features["alteration_frequency"] * 0.2
        if not np.isnan(features["alteration_frequency"]) else
        features["lineage_specificity"] * 0.5 + features["ot_genetic_association"] * 0.5,
        1.0
    )

    # Biomarker features
    features["biomarker_directness"] = (
        0.9 if has_biomarker and features["alteration_frequency"] > 0.05 else
        0.5 if has_biomarker else
        np.nan if not has_biomarker else 0.3
    )
    features["biomarker_prevalence"] = (
        features["alteration_frequency"] if has_biomarker and not np.isnan(features["alteration_frequency"])
        else (0.3 if has_biomarker else np.nan)
    )
    features["biomarker_clonality"] = 0.7 if has_biomarker else np.nan
    features["assay_precision"] = 0.8 if has_biomarker else np.nan

    return features

# ── Composite score computation ───────────────────────────────────────────────

def compute_composite_scores(features: dict) -> dict:
    """Compute 6 composite dimension scores + overall BIOLOGY_SCORE."""
    # Replace NaN with 0 for composite score computation
    # (NaN handled by missingness indicators separately)
    f = {k: (0.0 if (isinstance(v, float) and np.isnan(v)) else v)
         for k, v in features.items()}

    CDS = _weighted_sum(f, CDS_WEIGHTS)
    TDS = _weighted_sum(f, TDS_WEIGHTS)
    BFS = _weighted_sum(f, BFS_WEIGHTS)
    MCS = _weighted_sum(f, MCS_WEIGHTS)
    TWS = _weighted_sum(f, TWS_WEIGHTS)
    EMS = _weighted_sum(f, EMS_WEIGHTS)

    # Overall biology score — weighted combination of 6 dimensions
    BIOLOGY_SCORE = (
        0.25 * CDS +   # Clinical disease support (genetic evidence)
        0.20 * TDS +   # Target druggability
        0.20 * MCS +   # Mechanistic coherence
        0.15 * TWS +   # Therapeutic window
        0.10 * BFS +   # Biomarker fidelity
        0.10 * EMS     # Evidence maturity
    )

    return {
        "CDS": round(CDS, 4),
        "TDS": round(TDS, 4),
        "BFS": round(BFS, 4),
        "MCS": round(MCS, 4),
        "TWS": round(TWS, 4),
        "EMS": round(EMS, 4),
        "BIOLOGY_SCORE": round(BIOLOGY_SCORE, 4),
    }

# ── Main feature computation pipeline ────────────────────────────────────────

def compute_all_features(trials_df: pd.DataFrame,
                          pair_data: dict,
                          target_data: dict) -> pd.DataFrame:
    """
    Compute raw features + composite scores for all trials.
    Returns trials_df with feature columns appended.
    """
    all_rows = []
    for _, trial in tqdm(trials_df.iterrows(), total=len(trials_df),
                         desc="Computing features"):
        raw_features = compute_raw_features(trial, pair_data, target_data)
        composite = compute_composite_scores(raw_features)
        all_rows.append({**raw_features, **composite})

    feature_df = pd.DataFrame(all_rows, index=trials_df.index)
    # Keep normalized trial columns as the single source of truth when a raw
    # feature (currently ``drug_is_mapped``) is recomputed downstream.
    overlapping_cols = feature_df.columns.intersection(trials_df.columns)
    if len(overlapping_cols):
        feature_df = feature_df.drop(columns=overlapping_cols)
    result = pd.concat([trials_df, feature_df], axis=1)

    # ── Missingness report ─────────────────────────────────────────────────────
    n = len(result)
    raw_cols = [c for c in feature_df.columns if c not in ["CDS", "TDS", "BFS", "MCS", "TWS", "EMS", "BIOLOGY_SCORE"]]

    print(f"\n  Features: {len(raw_cols)} raw + 7 composite for {n} trials")
    print(f"  has_real_ot_data   : {(result.get('ot_data_missing', pd.Series([1]*n)) == 0).sum()}/{n} ({(result.get('ot_data_missing', pd.Series([1]*n)) == 0).mean():.1%})")
    print(f"  has_real_cbio_data : {(result.get('cbio_data_missing', pd.Series([1]*n)) == 0).sum()}/{n} ({(result.get('cbio_data_missing', pd.Series([1]*n)) == 0).mean():.1%})")
    print(f"  has_biomarker      : {(result.get('biomarker_missing', pd.Series([1]*n)) == 0).sum()}/{n} ({(result.get('biomarker_missing', pd.Series([1]*n)) == 0).mean():.1%})")

    # NaN counts (honest missingness)
    nan_cols = [(c, feature_df[c].isna().sum()) for c in raw_cols
                if feature_df[c].isna().sum() > 0]
    if nan_cols:
        print(f"  Missing (NaN) values — explicit, not median-imputed:")
        for col, cnt in sorted(nan_cols, key=lambda x: -x[1])[:10]:
            print(f"    {col:<40} {cnt} ({cnt/n:.1%})")

    log.info(f"Features computed for {n} trials, {len(raw_cols)} raw features")
    return result

# ── Feature set builder for modeling ─────────────────────────────────────────

def build_feature_sets(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Build three feature sets for modeling.

    v7 CHANGE: "composite" now includes pair-level features alongside composite
    scores. Previously composite = only 7 scores, which meant pair-level features
    had 0% importance because they never reached the model in the primary feature set.

    - composite: 7 composite scores + key pair-level + missingness flags (PRIMARY)
    - raw: all raw features
    - hybrid: composite + all pair-level features
    """
    composite_cols = ["CDS", "TDS", "BFS", "MCS", "TWS", "EMS", "BIOLOGY_SCORE"]

    # v7: pair-level features now included in composite set
    # These vary per (target, disease) pair and capture cancer-specific signal
    key_pair_cols = [
        "ot_overall_score", "ot_genetic_association", "ot_somatic_mutation",
        "ot_known_drug", "ot_rna_expression", "ot_literature",
        "pubmed_pair_count", "clinical_trial_pub_count", "pair_pub_acceleration",
        "lineage_specificity",
        "depmap_essential", "depmap_chronos", "cosmic_census_member",
        "gwas_association_count", "disgenet_score",
        "tractability", "binding_evidence", "potency_proxy",
        "tumor_expression", "tumor_specificity", "normal_tissue_burden",
        "pathway_evidence", "modality_score",
        # Missingness indicators
        "ot_data_missing", "cbio_data_missing", "pubmed_data_missing", "biomarker_missing",
        # v7: drug mapping flag — only include once
        # (also in raw features, deduplicated by _prep)
    ]

    available_composite = [c for c in composite_cols if c in df.columns]
    available_raw = [
        c for c in feature_cols if c in df.columns and c not in composite_cols
    ]
    available_pair = [c for c in key_pair_cols if c in df.columns]
    # Primary feature set: composite scores + pair-level features
    available_enhanced = list(dict.fromkeys(available_composite + available_pair))
    available_hybrid = list(dict.fromkeys(available_raw + available_composite))

    def _prep(X: pd.DataFrame) -> pd.DataFrame:
        # Deduplicate columns — LightGBM crashes on duplicate column names
        X = X.loc[:, ~X.columns.duplicated()]
        return X.fillna(-1)

    return {
        "composite": _prep(df[available_enhanced]),
        "raw": _prep(df[list(dict.fromkeys(available_raw))]),
        "hybrid": _prep(df[available_hybrid]),
    }
