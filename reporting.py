"""
reporting.py — Publication artefacts. (Issues 9, 10)

v6 reporting.py imported names that no longer existed in config (RAW_FEATURES,
COMPOSITE_FEATURES, LABEL_NAMES) and could not run. This version writes:

  flow_counts.csv               participant-flow numbers for a TRIPOD/CONSORT-style diagram
  label_provenance.csv          label tier x outcome counts
  label_agreement.csv           agreement / Cohen's kappa between independent evidence tiers
  label_audit_sample.csv        stratified sample for blinded dual manual curation
  mapping_unmapped_top.csv      most frequent unmapped intervention names (lexicon gaps)
  lexicon_disagreements.csv     seed vs ChEMBL target conflicts
  data_dictionary.csv           every feature, domain, leakage class, description
  model_card_<analysis>.json    features, params, CV, data manifest hash, holdout status
  TRIPOD_AI_checklist.md        skeleton pointing to where each item is answered
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import (DATA_DIR, FEATURE_REGISTRY, RAW_DIR, RANDOM_SEED, RELEASES, REPORT_DIR, log)


def flow_report(flow: dict) -> pd.DataFrame:
    df = pd.DataFrame([{"step": k, "n": v} for k, v in flow.items()])
    df.to_csv(REPORT_DIR / "flow_counts.csv", index=False)
    log.info("Flow:\n" + df.to_string(index=False))
    return df


def label_report(df: pd.DataFrame) -> None:
    prov = df.groupby(["phase_group", "label_tier"])["label_primary"].value_counts().unstack(fill_value=0)
    prov.to_csv(REPORT_DIR / "label_provenance.csv")
    rows = []
    for a, b in [("label_structured", "label_publication"), ("label_structured", "label_termination"),
                 ("label_publication", "label_termination"), ("label_primary", "label_transition")]:
        if a not in df or b not in df:
            continue
        both = df[df[a].isin([0, 1]) & df[b].isin([0, 1])]
        if len(both) < 10 or both[a].nunique() < 2 or both[b].nunique() < 2:
            continue
        rows.append({"tier_a": a, "tier_b": b, "n_both": len(both),
                     "raw_agreement": float((both[a] == both[b]).mean()),
                     "cohen_kappa": float(cohen_kappa_score(both[a], both[b]))})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "label_agreement.csv", index=False)
    for r in rows:
        log.info(f"Label agreement {r['tier_a']} vs {r['tier_b']}: n={r['n_both']} "
                 f"agree={r['raw_agreement']:.2f} kappa={r['cohen_kappa']:.2f}")


def export_label_audit_sample(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Stratified (tier x label) sample for two blinded curators; merge results into gold_labels.csv."""
    det = df[df["label_primary"].isin([0, 1])]
    if det.empty:
        return det
    per = max(1, n // max(1, det.groupby(["label_tier", "label_primary"]).ngroups))
    idx = [i for _, g in det.groupby(["label_tier", "label_primary"])
           for i in g.sample(min(len(g), per), random_state=RANDOM_SEED).index]
    sample = det.loc[idx]
    cols = ["nct_id", "title", "phase_group", "status", "why_stopped", "publication_pmid"]
    out = sample[[c for c in cols if c in sample]].copy()
    out = out.sample(frac=1, random_state=RANDOM_SEED)            # hide tier ordering from curators
    out["curator1_label"] = ""
    out["curator2_label"] = ""
    out["adjudicated_label"] = ""
    out["notes"] = ""
    out.to_csv(DATA_DIR / "label_audit_sample.csv", index=False)
    sample[["nct_id", "label_tier", "label_primary", "label_evidence"]].to_csv(
        DATA_DIR / "label_audit_sample_KEY_do_not_share_with_curators.csv", index=False)
    log.info(f"Label audit sample written ({len(out)} trials) -> {DATA_DIR / 'label_audit_sample.csv'}")
    return out


def mapping_report(df: pd.DataFrame, lexicon=None) -> None:
    c = Counter(n.lower() for lst in df.get("unmapped_interventions", []) for n in lst)
    pd.DataFrame(c.most_common(300), columns=["intervention", "n_trials"]).to_csv(
        REPORT_DIR / "mapping_unmapped_top.csv", index=False)
    if lexicon is not None and lexicon.disagreements:
        pd.DataFrame(lexicon.disagreements).to_csv(REPORT_DIR / "lexicon_disagreements.csv", index=False)
    by = df.groupby("phase_group")["investigational_drug"].apply(lambda s: s.notna().mean())
    log.info(f"Mapping rate by phase: {by.round(3).to_dict()}")


def data_dictionary() -> pd.DataFrame:
    dd = pd.DataFrame([{"feature": k, **v} for k, v in FEATURE_REGISTRY.items()])
    dd.to_csv(REPORT_DIR / "data_dictionary.csv", index=False)
    return dd


def _manifest_hash() -> str | None:
    p = RAW_DIR / "manifest.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def model_card(res, evaluation: dict | None) -> dict:
    card = {
        "analysis": res.spec, "label": res.label, "model": res.model_name, "params": res.params,
        "features": res.features, "stage2_features": res.stage2_features,
        "n_train": len(res.train_df), "n_train_pos": int(res.train_df[res.label].sum()),
        "n_test": len(res.test_df), "n_test_pos": int(res.test_df[res.label].sum()),
        "cv_selection": res.cv_table.head(10).to_dict(orient="records"),
        "releases": RELEASES, "data_manifest_sha256": _manifest_hash(),
        "holdout_evaluated": res.holdout_confirmatory is not None,
        "holdout_confirmatory": res.holdout_confirmatory,
    }
    if evaluation and "summary" in evaluation:
        card["test_metrics"] = evaluation["summary"].to_dict(orient="records")
    (REPORT_DIR / f"model_card_{res.spec['name']}.json").write_text(json.dumps(card, indent=2, default=str))
    return card


TRIPOD_ITEMS = [
    ("Title/abstract", "Identify as development + temporal validation of a prediction model; state outcome, population, AUC with CI."),
    ("Source of data", "CT.gov snapshot date + raw/manifest.json (sha256 of every external release)."),
    ("Participants", "reports/flow_counts.csv; eligibility in normalize.apply_scope_filters."),
    ("Outcome", "labels.py tiers; reports/label_provenance.csv; label_agreement.csv; gold-standard kappa."),
    ("Predictors", "reports/data_dictionary.csv (leakage class per feature); time-bounding rules in features.py."),
    ("Sample size", "Events per candidate feature in train; report in model_card_*.json."),
    ("Missing data", "Missingness table from features.compute_features log; handling in modeling.make_estimator."),
    ("Analysis methods", "Grouped CV model selection, temporal Platt calibration, cluster bootstrap."),
    ("Model performance", "reports/*_metrics.csv (discrimination, calibration, Brier), *_decision_curve.csv."),
    ("Model comparison", "reports/*_comparisons.csv vs baselines; stage2 - stage1."),
    ("Fairness/subgroups", "reports/*_subgroups.csv (stratum, disease, modality, seen vs novel drug)."),
    ("Validation", "Locked holdout: locks/holdout.json, locks/holdout_access.log; external registry if available."),
    ("Interpretation", "reports/*_domain_importance.csv, *_domain_ablation.csv."),
    ("Limitations", "Mapping bias audit, label noise, CT.gov outcome edits, undated knowledge sensitivity."),
    ("Registration/protocol", "OSF pre-registration DOI (fill in)."),
    ("Code/data availability", "Repository + Zenodo DOI; requirements lock; manifest."),
]


def tripod_checklist() -> None:
    lines = ["# TRIPOD+AI checklist (skeleton — complete against the official checklist)\n",
             "| Item | Where it is answered |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in TRIPOD_ITEMS]
    (REPORT_DIR / "TRIPOD_AI_checklist.md").write_text("\n".join(lines) + "\n")


def export_dataset(df: pd.DataFrame, name: str = "analysis_dataset") -> None:
    flat = df.drop(columns=[c for c in df.columns
                            if df[c].map(lambda v: isinstance(v, (dict, list))).any()], errors="ignore")
    for c in flat.columns:
        if flat[c].dtype == object:
            flat[c] = flat[c].astype("string")
    try:
        flat.to_parquet(DATA_DIR / f"{name}.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Parquet export failed ({exc}); writing CSV")
    flat.to_csv(DATA_DIR / f"{name}.csv", index=False)
