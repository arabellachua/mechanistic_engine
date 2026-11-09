"""
run_pipeline.py — End-to-end orchestration (v9).

    python run_pipeline.py                      # development run (holdout stays locked)
    python run_pipeline.py --no-pubmed          # faster: skip time-bounded PubMed features
    python run_pipeline.py --unlock-holdout     # ONE confirmatory run after pre-registration

Steps
  1  CT.gov snapshot (frozen after first download) -> parse
  2  scope filters (interventional, TREATMENT purpose, Phase 2/3, drug intervention)
  3  disease + biomarker normalisation
  4  drug mapping (ChEMBL + seed) -> strata
  5  trial-specific labels (tiers) + Phase 3 transition label
  6  mapping-bias audit; restrict to mapped trials
  7  versioned sources -> time-bounded features -> time-bound audit
  8  freeze holdout
  9  primary + secondary analyses (+ undated-knowledge sensitivity for the primary)
 10  evaluation, interpretation, holdout (if unlocked), reports
"""
from __future__ import annotations

import argparse
from collections import OrderedDict

import pandas as pd

import clients
import data_sources
import drug_mapping
import evaluation
import features
import interpretation
import labels
import leakage
import modeling
import normalize
import reporting
import splits
from config import PHASES, PRIMARY_ANALYSIS, SECONDARY_ANALYSES, log


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Mechanistic Trial Biology Engine v9")
    ap.add_argument("--refresh-trials", action="store_true", help="re-download CT.gov (new snapshot)")
    ap.add_argument("--no-chembl", action="store_true", help="seed lexicon only (development only)")
    ap.add_argument("--no-publications", action="store_true", help="skip PubMed results publications")
    ap.add_argument("--no-pubmed", action="store_true", help="skip time-bounded PubMed count features")
    ap.add_argument("--no-cbio", action="store_true", help="skip cBioPortal TCGA features")
    ap.add_argument("--skip-sources", nargs="*", default=[], help="e.g. depmap string gwas chembl_activity")
    ap.add_argument("--unlock-holdout", action="store_true", help="evaluate the locked holdout (logged)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    flow: "OrderedDict[str, int]" = OrderedDict()

    # 1–2 trials and scope
    snap = clients.download_trials(PHASES, force=args.refresh_trials)
    df = pd.DataFrame([clients.parse_study(s) for s in clients.iter_studies(snap)])
    df = normalize.add_basic_fields(df)
    df, f = normalize.apply_scope_filters(df, PHASES)
    flow.update(f)

    # 3–4 normalisation and drug mapping
    df = normalize.normalize_all(df)
    lexicon = drug_mapping.build_lexicon(use_chembl=not args.no_chembl)
    df = drug_mapping.map_all(df, lexicon)
    df = normalize.add_biomarker_target_match(df)
    flow["investigational_agent_identified"] = int(df["investigational_drug"].notna().sum())

    # 5 labels
    df = labels.build_labels(df, fetch_publications=not args.no_publications)
    df = labels.add_transition_labels(df)
    flow["label_primary_determinate"] = int(df["label_primary"].isin([0, 1]).sum())
    reporting.label_report(df)
    reporting.export_label_audit_sample(df)
    reporting.mapping_report(df, lexicon)

    # 6 mapping bias, then restrict
    leakage.audit_mapping_bias(df, "label_primary")
    pop = df[df["primary_target"].notna()].copy()
    flow["modelling_population"] = len(pop)

    # 7 features
    genes = sorted({g for ts in pop["drug_targets"] for g in ts if g != "DNA"})
    sources = data_sources.load_all(genes, skip=set(args.skip_sources))
    feats = features.compute_features(pop, sources, use_pubmed=not args.no_pubmed, use_cbio=not args.no_cbio)
    pop = pd.concat([pop, feats[[c for c in feats.columns if c not in pop.columns]]], axis=1)
    leakage.audit_time_bounds(pop)

    # 8 holdout
    splits.freeze_holdout(pop)
    pop["period"] = splits.assign_period(pop)
    flow.update({f"period_{k}": int(v) for k, v in pop["period"].value_counts().items()})
    reporting.flow_report(flow)
    reporting.data_dictionary()
    reporting.tripod_checklist()
    reporting.export_dataset(pop)

    # 9–10 analyses
    runs = [(PRIMARY_ANALYSIS, False), (PRIMARY_ANALYSIS, True)] + [(s, False) for s in SECONDARY_ANALYSES]
    for spec, undated in runs:
        spec_run = dict(spec, name=spec["name"] + ("_sens_undated" if undated else ""))
        res = modeling.run_analysis(pop, spec_run, unlock_holdout=args.unlock_holdout, include_undated=undated)
        if res is None:
            continue
        ev = evaluation.evaluate_analysis(res, "test")
        if not undated:
            interpretation.interpret(res)
        if args.unlock_holdout and (res.predictions["period"] == "holdout").any():
            evaluation.evaluate_analysis(res, "holdout")
            log.info(f"Holdout evaluated for {spec_run['name']} "
                     f"({'CONFIRMATORY' if res.holdout_confirmatory else 'POST HOC — report as such'})")
        reporting.model_card(res, ev)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
