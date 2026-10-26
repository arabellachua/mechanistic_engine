"""
run_pipeline.py — Main pipeline orchestrator for Mechanistic Phase 2 Biology Engine v6.

v6 IMPROVEMENTS SUMMARY:
  1. PAIR-LEVEL FEATURES: target+disease specific features (fixes "same score for all pembrolizumab trials")
  2. CIRCULARITY-FREE LABELS: biology features never used in label assignment
  3. AGGRESSIVE CACHING: API results cached to disk; re-runs take ~30 sec instead of 35 min
  4. HPA FIXED: uses Ensembl IDs via MyGene lookup (was returning 404 for all targets)
  5. OmniPath FALLBACK: 502s handled gracefully
  6. DisGeNET & DrugBank: now use direct REST APIs instead of broken OmniPath proxy
  7. DepMap: curated CRISPR Chronos scores for oncology targets
  8. MISSINGNESS INDICATORS: NaN + indicator flags instead of silent median imputation
  9. STRONGER REGULARIZATION: addresses CV AUC (0.463) vs held-out AUC (0.570) gap
  10. PRISTINE ANALYSIS: explicitly tests whether performance reflects biology vs data richness
"""

import os
import re

import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    DATA_DIR, REPORT_DIR, FIG_DIR, MODEL_DIR, TEST_YEAR_CUTOFF,
    RANDOM_SEED, log, RUN_DEEP_EXPERIMENTS, DEEP_FEATURE_SET, DEEP_LABEL_COL,
)

# ── Step imports ──────────────────────────────────────────────────────────────
from drug_target_db import DRUG_TARGET_DB

def _safe_import(module_name):
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        log.error(f"Could not import {module_name}: {e}")
        raise

# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_trials(studies: list[dict]) -> pd.DataFrame:
    """Parse and filter ClinicalTrials.gov studies to Phase 2 drug trials."""
    rows = []
    for s in studies:
        proto = s.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        cond_mod = proto.get("conditionsModule", {})
        inter_mod = proto.get("armsInterventionsModule", {})
        outcome_mod = proto.get("outcomesModule", {})
        desc_mod = proto.get("descriptionModule", {})

        phases = design.get("phases", [])
        if "PHASE2" not in phases:
            continue

        interventions = inter_mod.get("interventions", [])
        drugs = [i.get("name", "") for i in interventions
                 if i.get("type", "").upper() in ("DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT")]
        if not drugs:
            continue

        status = status_mod.get("overallStatus", "")

        rows.append({
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("officialTitle", ident.get("briefTitle", "")),
            "status": status,
            "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
            "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
            "results_first_posted": status_mod.get("resultsFirstPostDateStruct", {}).get("date", ""),
            "why_stopped": status_mod.get("whyStopped", ""),
            "conditions": "|".join(cond_mod.get("conditions", [])),
            "intervention_names": "|".join(drugs),
            "primary_outcome": "|".join(
                " ".join(str(o.get(field, "") or "")
                         for field in ("measure", "description", "timeFrame"))
                for o in outcome_mod.get("primaryOutcomes", [])
            ),
            "brief_summary": desc_mod.get("briefSummary", ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep Phase 2 drug trials in oncology conditions
    def is_oncology(conditions):
        kws = ["cancer", "carcinoma", "tumor", "lymphoma", "leukemia",
               "melanoma", "sarcoma", "neoplasm", "myeloma", "glioma",
               "neuroblastoma", "mesothelioma", "adenocarcinoma"]
        c = str(conditions).lower()
        return any(kw in c for kw in kws)

    df = df[df["conditions"].apply(is_oncology)].copy()

    print(f"\n  Raw studies: {len(studies)}")
    print(f"  Phase 2 with drug (oncology): {len(df)}")
    status_counts = df["status"].value_counts()
    print(f"  Status distribution:\n{status_counts.to_string()}")
    log.info(f"Filtered to {len(df)} Phase 2 drug trials")
    return df.reset_index(drop=True)

# ── Normalization ─────────────────────────────────────────────────────────────

DISEASE_NORMALIZATION = {
    "breast": "breast cancer",
    "her2": "breast cancer",
    "nsclc": "nsclc",
    "non-small cell lung": "nsclc",
    "non small cell lung": "nsclc",
    "lung adeno": "nsclc",
    "colon": "colorectal cancer",
    "colorectal": "colorectal cancer",
    "rectal": "colorectal cancer",
    "prostate": "prostate cancer",
    "melanoma": "melanoma",
    "ovarian": "ovarian cancer",
    "pancreatic": "pancreatic cancer",
    "pancreas": "pancreatic cancer",
    "aml": "leukemia",
    "cll": "leukemia",
    "leukemia": "leukemia",
    "lymphoma": "lymphoma",
    "dlbcl": "lymphoma",
    "follicular": "lymphoma",
    "myeloma": "multiple myeloma",
    "multiple myeloma": "multiple myeloma",
    "hepatocellular": "hepatocellular carcinoma",
    "hcc": "hepatocellular carcinoma",
    "liver": "hepatocellular carcinoma",
    "head and neck": "hnscc",
    "hnscc": "hnscc",
    "head neck": "hnscc",
    "bladder": "bladder cancer",
    "urothelial": "bladder cancer",
    "gastric": "gastric cancer",
    "stomach": "gastric cancer",
    "renal": "renal cell carcinoma",
    "kidney": "renal cell carcinoma",
    "rcc": "renal cell carcinoma",
    "glioblastoma": "glioblastoma",
    "gbm": "glioblastoma",
    "endometrial": "endometrial cancer",
    "uterine": "endometrial cancer",
    "cervical": "cervical cancer",
    "thyroid": "thyroid cancer",
    "solid tumor": "solid tumors",
    "advanced solid": "solid tumors",
}

BIOMARKER_INDICATORS = [
    "egfr", "her2", "erbb2", "brca", "kras", "braf", "pdl1", "pd-l1",
    "msi", "tmb", "alk", "ros1", "met", "ntrk", "ret", "fgfr", "pik3ca",
    "ar", "er+", "hormone receptor", "biomarker", "mutation", "amplification",
    "overexpression", "biomarker-selected", "biomarker selected",
]


def classify_endpoint_type(endpoint_text: str) -> str | None:
    """Classify a primary-outcome description into a coarse endpoint group."""
    text = str(endpoint_text or "").lower()
    if not text.strip():
        return None
    if any(k in text for k in ["overall survival", "mortality", "time to death"]):
        return "survival"
    if re.search(r"\bos\b", text) and not re.search(r"\bdose\b", text):
        return "survival"
    if any(k in text for k in [
        "progression-free", "progression free", "time to progression",
    ]) or re.search(r"\bpfs\b", text):
        return "pfs"
    if any(k in text for k in [
        "response rate", "objective response", "complete response",
        "partial response", "disease control rate", "clinical benefit rate",
    ]) or re.search(r"\borr\b", text):
        return "response"
    if any(k in text for k in ["biomarker", "expression", "mutation"]):
        return "biomarker"
    return "other"

def normalize_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Map drug names to targets, normalize disease names, extract biomarkers."""
    print("  Normalizing drugs...")
    drug_lower = {k.lower(): v for k, v in DRUG_TARGET_DB.items()}

    canonical_drugs, primary_targets, modalities = [], [], []
    for _, row in df.iterrows():
        drug_names = str(row.get("intervention_names", "")).lower()
        matched_drug = next((k for k in drug_lower if k in drug_names), None)
        if matched_drug:
            info = drug_lower[matched_drug]
            canonical_drugs.append(matched_drug)
            t = info.get("targets", info.get("target", "UNKNOWN")); t = t[0] if isinstance(t, list) and t else (t if not isinstance(t, list) else "UNKNOWN"); primary_targets.append(t)
            modalities.append(info.get("modality", "unknown"))
        else:
            canonical_drugs.append("unmapped")
            primary_targets.append("UNKNOWN")
            modalities.append("unknown")

    df = df.copy()
    df["canonical_drug"] = canonical_drugs
    df["primary_target"] = primary_targets
    df["modality"] = modalities
    # v7: explicit flag — prevents model from learning unmapped=novel=positive
    df["drug_is_mapped"] = [0 if d == "unmapped" else 1 for d in canonical_drugs]

    print("  Normalizing diseases...")
    def normalize_disease(conditions):
        c = str(conditions).lower()
        for key, normalized in sorted(DISEASE_NORMALIZATION.items(), key=lambda x: -len(x[0])):
            if key in c:
                return normalized
        return "cancer nos"

    df["disease"] = df["conditions"].apply(normalize_disease)

    print("  Extracting biomarkers...")
    def has_biomarker(row):
        text = " ".join([
            str(row.get("title", "")),
            str(row.get("brief_summary", "")),
            str(row.get("conditions", "")),
        ]).lower()
        return any(bm in text for bm in BIOMARKER_INDICATORS)

    df["has_biomarker"] = df.apply(has_biomarker, axis=1)

    # IO trial flag — needed for separate IO/non-IO models in Step 17
    IO_TARGETS_SET = {"PDCD1", "CD274", "CTLA4"}
    IO_MODALITIES = {"antibody"}
    df["is_io_trial"] = df.apply(
        lambda r: 1 if (
            r.get("primary_target", "") in IO_TARGETS_SET or
            (r.get("modality", "") in IO_MODALITIES and
             r.get("primary_target", "") in IO_TARGETS_SET)
        ) else 0, axis=1
    )

    # Endpoint type from primary outcome text
    df["endpoint_type"] = df["primary_outcome"].apply(classify_endpoint_type)
    df["endpoint_type_num"] = df["endpoint_type"].map(
        {"survival": 0, "pfs": 1, "response": 2, "biomarker": 3, "other": 4}
    ).fillna(-1)

    # Distinguish completed-no-results from completed-with-results
    # This addresses the CT.gov signals question from professor
    df["completed_no_results"] = (
        (df["status"] == "COMPLETED") &
        (~df["results_first_posted"].notna() | (df["results_first_posted"] == ""))
    ).astype(int)

    io_count = df["is_io_trial"].sum()
    ep_dist = df[df["endpoint_type"].notna()]["endpoint_type"].value_counts().to_dict()
    print(f"    IO trials flagged: {io_count}")
    print(f"    Endpoint types: {ep_dist}")

    # Extract start year for temporal split
    def _year(date_str):
        try:
            return int(str(date_str)[:4])
        except (ValueError, TypeError):
            return None

    df["start_year"] = df["start_date"].apply(_year)

    # Summary
    mapped = (df["canonical_drug"] != "unmapped").sum()
    print(f"\n  Normalization complete:")
    print(f"    Total trials:       {len(df)}")
    print(f"    Drug mapped:        {mapped} ({mapped/len(df):.1%})")
    print(f"    Unique drugs:       {df[df['canonical_drug'] != 'unmapped']['canonical_drug'].nunique()}")
    print(f"    Unique targets:     {df[df['primary_target'] != 'UNKNOWN']['primary_target'].nunique()}")
    print(f"    Unique diseases:    {df['disease'].nunique()}")
    print(f"    Trials with biomarker: {df['has_biomarker'].sum()}")
    log.info(f"Normalization: {mapped}/{len(df)} drugs mapped, {df['has_biomarker'].sum()} with biomarkers")
    return df

# ── Dataset split ─────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, label_col: str = "label_permissive",
                   group_purge: bool = True):
    """
    Temporal split: trials starting ≤ cutoff → train, > cutoff → test.
    v7: Uses label_permissive as default for more determinate trials.
    Prints year distribution to verify test set size.

    v9 ADDED `group_purge` (default True): the biology features in this
    pipeline are a pure function of (primary_target, disease) — they do not
    vary by trial. A temporal-only split lets a test trial share its entire
    feature vector with a training trial that has the same (target, disease)
    pair (an audit of the shipped matrix found 43% of test rows did). That
    lets a model memorise rather than generalise, and inflates every held-out
    metric. With group_purge, any test row whose (target, disease) pair also
    appears in train is dropped — the reported test set only contains biology
    the model has genuinely not seen. See docs/label_audit.md.
    """
    det = df[df[label_col] != -1].copy()

    # Show distribution so we can verify test set is large enough
    year_dist = det.groupby("start_year")[label_col].agg(["sum", "count"])
    year_dist.columns = ["pos", "total"]
    year_dist["neg"] = year_dist["total"] - year_dist["pos"]
    print(f"  Determinate trials by year:")
    for yr, row in year_dist.iterrows():
        marker = " <-- train/test cutoff" if yr == TEST_YEAR_CUTOFF else ""
        print(f"    {int(yr) if yr == yr else 'N/A'}: {int(row['total'])} trials "
              f"({int(row['pos'])} pos, {int(row['neg'])} neg){marker}")

    train = det[det["start_year"] <= TEST_YEAR_CUTOFF]
    test  = det[det["start_year"] > TEST_YEAR_CUTOFF]

    if group_purge and len(train) and len(test):
        seen_pairs = set(zip(train["primary_target"], train["disease"]))
        in_train = test.apply(
            lambda r: (r["primary_target"], r["disease"]) in seen_pairs, axis=1)
        n_purged = int(in_train.sum())
        if n_purged:
            print(f"  Group purge: removing {n_purged}/{len(test)} test rows "
                  f"whose (target, disease) pair also appears in train")
            test = test[~in_train]

    print(f"\n  TEMPORAL SPLIT: train ≤{TEST_YEAR_CUTOFF} ({len(train)}), "
          f"test >{TEST_YEAR_CUTOFF} ({len(test)})")
    print(f"    Train pos rate: {(train[label_col] == 1).mean():.3f}")
    if len(test):
        print(f"    Test pos rate:  {(test[label_col] == 1).mean():.3f}")

    if len(test) < 80:
        print(f"  ⚠ Test set small ({len(test)} trials). "
              f"Consider lowering TEST_YEAR_CUTOFF in config.py.")
    return train, test

def build_modeling_data(df_features: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Build train/test splits and feature matrices for all label definitions."""
    from features import build_feature_sets

    label_defs = ["label_strict", "label_balanced", "label_permissive"]
    primary_label = "label_permissive"  # v7: use permissive for more training data

    train_df, test_df = temporal_split(df_features, "label_permissive")

    # Build feature sets
    train_features = build_feature_sets(train_df, feature_cols)
    test_features = build_feature_sets(test_df, feature_cols)

    # Align columns
    feature_sets = {}
    for fs_name in train_features:
        X_tr = train_features[fs_name]
        X_te = test_features[fs_name].reindex(columns=X_tr.columns, fill_value=-1)
        feature_sets[fs_name] = (X_tr, X_te)

    train_labels = {ln: (train_df[ln] if ln in train_df.columns
                         else pd.Series([0]*len(train_df), index=train_df.index))
                   for ln in label_defs}
    test_labels = {ln: (test_df[ln] if ln in test_df.columns
                        else pd.Series([0]*len(test_df), index=test_df.index))
                  for ln in label_defs}

    print(
        "  Features: "
        + ", ".join(
            f"{name}={len(matrices[0].columns)}"
            for name, matrices in feature_sets.items()
        )
    )

    return {
        "feature_sets": feature_sets,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_df": train_df,
        "test_df": test_df,
    }

# ── Error analysis ────────────────────────────────────────────────────────────

def error_analysis(best_models: dict, data: dict, label_name: str = "label_permissive"):
    best = best_models.get(label_name)
    if not best or not best.get("model"):
        return

    model = best["model"]
    fs_name = best["features"]
    X_te_all = data["feature_sets"][fs_name][1]
    y_te_all = data["test_labels"][label_name]
    test_df = data["test_df"]

    # Filter indeterminate labels
    clean_mask = y_te_all.isin([0, 1])
    X_te = X_te_all[clean_mask]
    y_te = y_te_all[clean_mask]
    test_df = test_df[test_df.index.isin(X_te.index)]

    if len(y_te) == 0 or len(y_te.unique()) < 2:
        print("  Insufficient labeled test data for error analysis")
        return

    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    asset = test_df.copy()
    asset["predicted_p"] = probs
    asset["pred_label"] = preds
    asset["true_label"] = y_te.values

    fps = asset[(asset["pred_label"] == 1) & (asset["true_label"] == 0)]
    fns = asset[(asset["pred_label"] == 0) & (asset["true_label"] == 1)]

    print(f"\n  === ERROR ANALYSIS ===")
    print(f"  False Positives: {len(fps)} (predicted success, actually failed)")
    if len(fps):
        print(f"    Most common targets: {fps['primary_target'].value_counts().head(3).to_dict()}")
        print(f"    Most common diseases: {fps['disease'].value_counts().head(3).to_dict()}")

    print(f"  False Negatives: {len(fns)} (predicted failure, actually succeeded)")
    if len(fns):
        print(f"    Most common targets: {fns['primary_target'].value_counts().head(3).to_dict()}")
        print(f"    Most common diseases: {fns['disease'].value_counts().head(3).to_dict()}")

    return {"false_positives": fps, "false_negatives": fns}

# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("  MECHANISTIC PHASE 2 BIOLOGY ENGINE (v6)")
    print("  Pair-level features | Circularity-free labels | Disk cache")
    print("  13 databases | Missingness indicators | Stronger regularization")
    print("=" * 70)

    # ── Dynamic imports (after config is verified) ──
    from clients import download_trials, enrich_all
    from labels import build_labels
    from features import compute_all_features
    from config import ALL_RAW_FEATURES
    from modeling import (
        train_all_models, cross_validate, plot_evaluation,
        plot_feature_importance, sensitivity_analysis, pristine_analysis,
    )
    from counterfactual import run_counterfactual

    # ── STEP 1: Download ──────────────────────────────────────────────────────
    print("\n[STEP 1] Downloading trials...")
    studies = download_trials()

    # ── STEP 2: Filter ────────────────────────────────────────────────────────
    print("\n[STEP 2] Filtering...")
    df = filter_trials(studies)

    # ── STEP 3: Normalize ────────────────────────────────────────────────────
    print("\n[STEP 3] Normalizing...")
    df = normalize_trials(df)

    # ── STEP 4: Circularity-free labels ──────────────────────────────────────
    print("\n[STEP 4] Building labels (circularity-free — no biology used)...")
    df = build_labels(df)
    # v8: set USE_V8_LABELS=True to enrich labels with CT.gov results + PubMed
    # Adds ~30 min first run (cached after). Default False for speed.
    #TODO: Can we use V8 as default or at least compare it with V8 = TRUE?
    USE_V8_LABELS = False
    if USE_V8_LABELS:
        try:
            from labels_v8 import build_labels_v8
            print("  Enriching labels with structured results + publications...")
            df = build_labels_v8(df, fetch_live=True)
        except Exception as e:
            log.warning(f"v8 label enrichment failed: {e}")
    labelable = df[df["label_balanced"] != -1]
    pos = (labelable["label_balanced"] == 1).sum()
    neg = (labelable["label_balanced"] == 0).sum()
    print(f"\n  Labels (circularity-free):")
    print(f"    Full corpus: {len(df)}")
    print(f"    Labelable:   {len(labelable)}")
    print(f"    DETERMINATE: {len(labelable)} (pos={pos}, neg={neg})")
    log.info(f"Labels: {pos} positive, {neg} negative, "
             f"indeterminate={len(df)-len(labelable)}, "
             f"pos_rate={pos/(pos+neg):.3f}")

    # ── STEP 5: API enrichment ────────────────────────────────────────────────
    print("\n[STEP 5] Querying 13+ databases (cached — fast on re-runs)...")

    mapped = df[df["primary_target"] != "UNKNOWN"]
    targets = sorted(mapped["primary_target"].unique().tolist())
    # (target, disease, start_year): the year bounds the PubMed queries so a
    # trial's features only see literature that existed when it started.
    pairs = sorted(set(
        (t, d, int(y))
        for t, d, y in zip(mapped["primary_target"], mapped["disease"],
                           mapped["start_year"].fillna(0))
        if y and y > 0
    ))

    enrichment = enrich_all(pairs, targets)
    pair_data = enrichment["pair_data"]
    target_data = enrichment["target_data"]

    # ── STEP 6: Features ─────────────────────────────────────────────────────
    print("\n[STEP 6] Computing features (pair-level + missingness indicators)...")
    df = compute_all_features(df, pair_data, target_data)
    try:
        from embedding_features import apply_precomputed_cbio_features
        df = apply_precomputed_cbio_features(df)
    except Exception as e:
        log.warning(f"Precomputed cBioPortal merge skipped: {e}")

    feature_cols = [c for c in df.columns
                    if c in ALL_RAW_FEATURES
                    or c in ["CDS", "TDS", "BFS", "MCS", "TWS", "EMS", "BIOLOGY_SCORE"]
                    or c.endswith("_missing")]
    df.to_csv(DATA_DIR / "full_feature_matrix.csv", index=False)
    try:
        df.to_parquet(DATA_DIR / "full_feature_matrix.parquet", index=False)
    except Exception:
        pass

    # ── STEP 7: Modeling dataset ──────────────────────────────────────────────
    print("\n[STEP 7] Building modeling dataset...")
    data = build_modeling_data(df, feature_cols)

    det = df[df["label_permissive"] != -1]
    pos = (det["label_permissive"] == 1).sum()
    neg = (det["label_permissive"] == 0).sum()
    print(f"  Determinate (permissive): {len(det)}")
    print(f"  Positive: {pos} ({pos/len(det):.1%})")
    print(f"  Negative: {neg} ({neg/len(det):.1%})")

    # ── STEP 8: Train models ──────────────────────────────────────────────────
    print("\n[STEP 8] Training models (stronger regularization)...")
    model_output = train_all_models(data)
    trained_models = model_output["trained"]
    best_models = model_output["best"]

    if RUN_DEEP_EXPERIMENTS:
        print("\n[STEP 8B] Deep learning branch (learned embeddings)...")
        try:
            from modeling_deep import run_deep_experiments
            run_deep_experiments(
                df_features=df,
                modeling_data=data,
                tree_model_output=model_output,
                feature_set=DEEP_FEATURE_SET,
                label_col=DEEP_LABEL_COL,
            )
        except ImportError as e:
            print(f"  Deep branch skipped: {e}")
            print("  Install optional dependencies with `.venv/bin/pip install -r requirements-deep.txt`.")
        except Exception as e:
            log.warning(f"Deep branch failed: {e}")

    # ── STEP 9: Evaluation plots ──────────────────────────────────────────────
    print("\n[STEP 9] Evaluation plots...")
    plot_evaluation(best_models, data)
    imp_df = plot_feature_importance(best_models, data)

    # ── STEP 10: Classification matrix ───────────────────────────────────────
    print("\n[STEP 10] Classification matrix...")
    label_cols = ["label_strict", "label_balanced", "label_permissive", "label_provenance"]
    meta_cols = ["nct_id", "title", "status", "canonical_drug", "primary_target",
                 "modality", "disease", "has_biomarker", "start_year"]
    available_cols = [c for c in meta_cols + label_cols + feature_cols if c in df.columns]
    det_df = df[df["label_balanced"] != -1][available_cols]
    det_df.to_csv(DATA_DIR / "classification_matrix.csv", index=False)
    try:
        det_df.to_parquet(DATA_DIR / "classification_matrix.parquet", index=False)
    except Exception:
        pass
    print(f"  Classification matrix: {det_df.shape[0]} rows x {det_df.shape[1]} columns")
    log.info(f"Classification matrix: {det_df.shape}")

    # ── STEP 11: Cross-validation ─────────────────────────────────────────────
    print("\n[STEP 11] Cross-validation...")
    print("  === 5-FOLD CROSS-VALIDATION ===")
    X_tr, _ = data["feature_sets"]["composite"]
    y_tr = data["train_labels"]["label_balanced"]
    cv_results = cross_validate(X_tr, y_tr)

    # ── STEP 12: Sensitivity analysis ─────────────────────────────────────────
    print("\n[STEP 12] Sensitivity analysis...")
    sensitivity_analysis(data)

    # ── STEP 13: Counterfactual ───────────────────────────────────────────────
    print("\n[STEP 13] Counterfactual analysis...")
    try:
        run_counterfactual(best_models, data, n_trials=min(100, len(data["test_df"])))
    except Exception as e:
        log.warning(f"Counterfactual failed: {e}")

    # ── STEP 14: Error analysis ───────────────────────────────────────────────
    print("\n[STEP 14] Error analysis...")
    error_analysis(best_models, data)

    # ── STEP 15: Pristine (OT-characterized) analysis ────────────────────────
    print("\n[STEP 15] Pristine dataset analysis...")
    pristine_analysis(data, df)

    # ── STEP 16: Asset predictions ────────────────────────────────────────────
    print("\n[STEP 16] Asset report...")
    best = best_models.get("label_permissive")
    if best and best.get("model"):
        fs_name = best["features"]
        X_te_all = data["feature_sets"][fs_name][1]
        y_te_all = data["test_labels"]["label_permissive"]
        clean_m = y_te_all.isin([0, 1])
        X_te = X_te_all[clean_m]
        test_df = data["test_df"][data["test_df"].index.isin(X_te.index)]
        probs = best["model"].predict_proba(X_te)[:, 1]
        asset = test_df[["nct_id", "canonical_drug", "disease", "primary_target",
                         "modality"]].copy()
        asset["predicted_p"] = probs.round(3)
        asset = asset.sort_values("predicted_p", ascending=False)
        asset.to_csv(REPORT_DIR / "asset_predictions.csv", index=False)
        print("  Top 5 predicted successes:")
        print(asset.head(5)[["nct_id", "canonical_drug", "disease", "predicted_p"]].to_string(index=False))
        print("  Bottom 5 predicted failures:")
        print(asset.tail(5)[["nct_id", "canonical_drug", "disease", "predicted_p"]].to_string(index=False))

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL REPORT (v6)")
    print("=" * 70)
    print(f"\n  CORPUS: {len(studies)} ingested, {len(det)} determinate")
    print(f"  Train/Test: {len(data['train_df'])}/{len(data['test_df'])}")
    print(f"  Positive: {pos} ({pos/len(det):.1%})")
    print(f"\n  BEST MODELS:")
    for label_name, info in best_models.items():
        print(f"    {label_name:<22} -> {info['model_name']} ({info['features']}) AUC={info['auc']:.3f}")
    print(f"\n  CV (LightGBM/composite): AUC={cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
    print(f"\n  v6 KEY FIXES APPLIED:")
    print(f"    ✓ Pair-level features: target+disease specific (not just target)")
    print(f"    ✓ Circularity-free labels: biology features never used in label assignment")
    print(f"    ✓ Disk caching: API results persist between runs (~30s on re-runs)")
    print(f"    ✓ HPA fixed: Ensembl ID lookup via MyGene.info")
    print(f"    ✓ DisGeNET & DGIdb: direct REST APIs (not broken OmniPath proxy)")
    print(f"    ✓ DepMap: curated CRISPR Chronos scores for 35 oncology targets")
    print(f"    ✓ Missingness: NaN + indicator flags (not silent median imputation)")
    print(f"    ✓ Regularization: stronger L1/L2, fewer leaves, larger min_child_samples")

    # ── STEP 17: v8 improvements ─────────────────────────────────────────────
    print("\n[STEP 17] v8 improvements (IO features, endpoint models, two-stage)...")
    try:
        from clients_v8_additions import get_io_features, compute_split_tds
        from modeling_v8 import train_endpoint_models, train_io_split_models, train_continuous_model, train_stage2

        # Add IO features and split TDS
        print("  Adding IO-specific features and split TDS to feature set...")
        io_rows = []
        for _, row in df.iterrows():
            sym = row.get("primary_target", "UNKNOWN")
            mod = row.get("modality", "unknown")
            io_rows.append({**get_io_features(sym), **compute_split_tds(sym, mod, dict(row))})
        io_feat_df = pd.DataFrame(io_rows, index=df.index)
        df_v8 = pd.concat([df, io_feat_df], axis=1)
        new_fcols = feature_cols + [c for c in io_feat_df.columns if c not in feature_cols]

        # Endpoint-specific models
        ep_results = train_endpoint_models(df_v8, new_fcols)

        # IO vs non-IO models
        io_split_results = train_io_split_models(df_v8, new_fcols)

        # Continuous outcome regression
        cont_results = train_continuous_model(df_v8, new_fcols)

        # Two-stage model
        print("\n  Two-stage model (Stage 1=biology, Stage 2=biology+trial features):")
        best_perm = best_models.get("label_permissive")
        if best_perm and best_perm.get("model"):
            fs = best_perm["features"]
            X_tr_s1, X_te_s1 = data["feature_sets"][fs]
            y_tr_all = data["train_labels"]["label_permissive"]
            y_te_all = data["test_labels"]["label_permissive"]
            mask_tr = y_tr_all.isin([0, 1])
            mask_te = y_te_all.isin([0, 1])

            # Stage 1 train probs via cross-val to avoid leakage
            from sklearn.model_selection import StratifiedKFold
            try:
                import lightgbm as lgb
                s1_probs_tr = np.zeros(mask_tr.sum())
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                X_c, y_c = X_tr_s1[mask_tr], y_tr_all[mask_tr]
                for ti, vi in skf.split(X_c, y_c):
                    m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15,
                                            scale_pos_weight=4.0, random_state=42, verbose=-1)
                    m.fit(X_c.iloc[ti], y_c.iloc[ti])
                    s1_probs_tr[vi] = m.predict_proba(X_c.iloc[vi])[:, 1]
                s1_probs_te = best_perm["model"].predict_proba(X_te_s1[mask_te])[:, 1]

                tr_s2 = data["train_df"][mask_tr].copy()
                te_s2 = data["test_df"][mask_te].copy()
                tr_s2["biology_score_s1"] = s1_probs_tr
                te_s2["biology_score_s1"] = s1_probs_te

                s2_cols = [c for c in ["biology_score_s1","drug_is_mapped","modality_score",
                           "biomarker_missing","is_io_trial"] if c in tr_s2.columns]
                if len(s2_cols) >= 2:
                    s2_res = train_stage2(tr_s2[s2_cols].fillna(-1), y_tr_all[mask_tr],
                                          te_s2[s2_cols].fillna(-1), y_te_all[mask_te])
                    if "error" not in s2_res:
                        s2_auc = s2_res["best"]["auc"]
                        s1_auc = best_perm["auc"]
                        delta = s2_auc - s1_auc
                        print(f"  Stage 1 (biology only):          AUC={s1_auc:.3f}")
                        print(f"  Stage 2 (biology+trial features): AUC={s2_auc:.3f} (delta={delta:+.3f})")
                        if delta > 0.01:
                            print("  ✓ Two-stage model improves on biology-only")
                        else:
                            print("  Trial-specific features add limited signal over biology alone")
            except Exception as e2:
                print(f"  Two-stage failed: {e2}")

    except Exception as e:
        import traceback
        log.warning(f"v8 steps failed: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
