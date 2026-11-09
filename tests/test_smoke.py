"""
Offline smoke test (no network): exercises every module on synthetic data.
Run:  MPE_OUTPUT=/tmp/mpe_smoke python -m pytest -q tests/test_smoke.py
  or: MPE_OUTPUT=/tmp/mpe_smoke python tests/test_smoke.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPE_OUTPUT", tempfile.mkdtemp(prefix="mpe_smoke_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import clients
import drug_mapping
import evaluation
import interpretation
import labels
import leakage
import modeling
import normalize
import reporting
import splits
from config import FEATURE_REGISTRY, PRIMARY_ANALYSIS, STAGE1_ALLOWED, TRIAL_DESIGN


def fake_study(i, phase="PHASE2", drug="osimertinib", cond="Non-Small Cell Lung Cancer", year=2010,
               p_value="0.01", status="COMPLETED"):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": f"NCT{i:08d}", "officialTitle": f"Study of {drug} in EGFR-mutant {cond}"},
            "statusModule": {"overallStatus": status, "startDateStruct": {"date": f"{year}-03"},
                             "primaryCompletionDateStruct": {"date": f"{year + 2}-06"}},
            "designModule": {"studyType": "INTERVENTIONAL", "phases": [phase],
                             "designInfo": {"allocation": "RANDOMIZED", "primaryPurpose": "TREATMENT"}},
            "conditionsModule": {"conditions": [cond]},
            "armsInterventionsModule": {
                "armGroups": [{"label": "A", "type": "EXPERIMENTAL"}, {"label": "B", "type": "ACTIVE_COMPARATOR"}],
                "interventions": [{"type": "DRUG", "name": f"{drug} + carboplatin", "armGroupLabels": ["A"]},
                                  {"type": "DRUG", "name": "Carboplatin", "armGroupLabels": ["A", "B"]}]},
            "outcomesModule": {"primaryOutcomes": [{"measure": "Progression-free survival"}]},
            "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n- EGFR mutation positive metastatic carcinoma\n"
                                                         "Exclusion Criteria:\n- prior treatment"},
            "sponsorCollaboratorsModule": {"leadSponsor": {"class": "INDUSTRY"}},
        },
        "hasResults": True,
        "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [
            {"type": "PRIMARY", "analyses": [{"pValue": p_value, "paramType": "Hazard Ratio (HR)", "paramValue": "0.6",
                                              "pValueComment": "one-sided alpha of 0.10"}]}]}},
    }


def test_parse_and_normalize():
    rec = clients.parse_study(fake_study(1))
    assert rec["nct_id"] == "NCT00000001" and rec["primary_purpose"] == "TREATMENT"
    assert "enrollment" not in " ".join(rec.keys())
    df = normalize.add_basic_fields(pd.DataFrame([rec]))
    df, flow = normalize.apply_scope_filters(df, ("PHASE2", "PHASE3"))
    df = normalize.normalize_all(df)
    assert df.loc[0, "disease"] == "nsclc"
    assert df.loc[0, "biomarkers"] == ["EGFR mutation"]
    # v6 bug: 'metastatic carcinoma' / 'treatment' must NOT trigger MET/AR/RET
    assert normalize.extract_biomarkers("Metastatic carcinoma treatment", "") == []
    # exclusion criteria are not selection
    assert normalize.extract_biomarkers("", "Inclusion Criteria: adults\nExclusion Criteria: EGFR mutation positive") == []
    assert normalize.normalize_disease(["Non-Hodgkin Lymphoma"], [], "")[0] == "lymphoma"


def test_drug_mapping():
    lex = drug_mapping.DrugLexicon(None)
    rec = clients.parse_study(fake_study(1))
    m = drug_mapping.map_trial(rec, lex)
    assert m["investigational_drug"] == "osimertinib" and m["primary_target"] == "EGFR"
    assert m["stratum"] == "targeted" and "carboplatin" in m["backbone_agents"]
    assert lex.find("Keytruda (MK-3475)")[0].canonical == "pembrolizumab"
    assert lex.find("carboplatin")[0].canonical == "carboplatin"
    assert not lex.find("paracetamol"), "no substring matches"


def test_labels():
    assert labels.parse_p_value("<0.001") == (0.001, "<")
    assert labels.parse_alpha("one-sided alpha of 0.10") == 0.10
    rec = clients.parse_study(fake_study(1, p_value="0.08"))
    assert labels.label_from_structured(rec["primary_results"])[0] == 1        # alpha 0.10 parsed
    assert labels.label_from_structured([{"analyses": [{"pValue": "0.30"}]}])[0] == 0
    sections = [("CONCLUSIONS", "The addition of X did not improve progression-free survival.")]
    assert labels.classify_abstract(sections)[0] == 0
    sections = [("CONCLUSIONS", "X showed promising activity and warrants further investigation in phase III.")]
    assert labels.classify_abstract(sections)[0] == 1
    assert labels.label_from_termination("TERMINATED", "Slow accrual")[0] == -1
    assert labels.label_from_termination("TERMINATED", "Futility at interim analysis")[0] == 0
    assert labels.label_from_termination("TERMINATED", "Stopped early for efficacy")[0] == 1


def synthetic_population(n=1400, seed=0):
    rng = np.random.default_rng(seed)
    drugs = [f"drug{k}" for k in range(260)]
    targets = [f"GENE{k}" for k in range(60)]
    diseases = ["nsclc", "breast", "colorectal", "melanoma", "aml", "prostate"]
    rows = []
    for i in range(n):
        d = rng.choice(drugs)
        year = int(rng.integers(2005, 2024))
        rows.append({"nct_id": f"NCT{i:08d}", "phase_group": "PHASE2" if rng.random() < 0.8 else "PHASE3",
                     "stratum": rng.choice(["targeted", "targeted", "io_checkpoint", "cytotoxic"]),
                     "investigational_drug": d, "primary_target": targets[hash(d) % 60],
                     "drug_targets": [targets[hash(d) % 60]], "disease": rng.choice(diseases),
                     "modality": rng.choice(["small_molecule", "antibody", "adc"]), "start_year": float(year),
                     "primary_completion_date_dt": pd.Timestamp(f"{year + 2}-06-01"),
                     "completion_date_dt": pd.Timestamp(f"{year + 3}-01-01"),
                     "status": "COMPLETED", "why_stopped": "", "title": "t"})
    df = pd.DataFrame(rows)
    for c, meta in FEATURE_REGISTRY.items():
        if meta["leakage_class"] in STAGE1_ALLOWED | {TRIAL_DESIGN}:
            df[c] = rng.normal(size=n)
    for c in ("biomarker_selected", "biomarker_matches_target", "is_randomized"):
        df[c] = (rng.random(n) < 0.4).astype(float)
    df.loc[rng.random(n) < 0.2, "depmap_lineage_effect"] = np.nan
    lin = 1.2 * df["somatic_mut_freq"] - 1.0 * df["depmap_lineage_effect"].fillna(0) + 0.8 * df["biomarker_matches_target"] - 1.2
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    df["label_primary"] = np.where(rng.random(n) < 0.1, -1, y)
    df["label_confidence"] = rng.choice([0.9, 0.75, 0.7], size=n)
    df["label_tier"] = rng.choice(["structured_results", "publication", "termination_text"], size=n)
    df["label_structured"] = np.where(rng.random(n) < 0.5, df["label_primary"], -1)
    df["label_publication"] = np.where(rng.random(n) < 0.5, np.where(rng.random(n) < 0.85, df["label_primary"], 1 - y), -1)
    df["label_termination"] = -1
    df["label_evidence"] = "synthetic"
    df["publication_pmid"] = None
    df["label_transition"] = np.where(df["phase_group"] == "PHASE2", y, -1)
    df["label_transition_confidence"] = 0.8
    df["knowledge_cutoff_year"] = df["start_year"] - 1
    return df


def test_leakage_guard():
    try:
        leakage.assert_allowed(["somatic_mut_freq", "ot_known_drug"], "stage1")
    except leakage.LeakageError:
        pass
    else:
        raise AssertionError("outcome proxy was not blocked")
    try:
        leakage.assert_allowed(["biomarker_selected"], "stage1")
    except leakage.LeakageError:
        pass
    else:
        raise AssertionError("design feature leaked into stage 1")


def test_end_to_end_modeling():
    pop = synthetic_population()
    leakage.audit_time_bounds(pop)
    splits.freeze_holdout(pop)
    res = modeling.run_analysis(pop, PRIMARY_ANALYSIS, unlock_holdout=True, models=("logreg", "lightgbm"))
    assert res is not None
    assert not any(FEATURE_REGISTRY[f]["leakage_class"] == TRIAL_DESIGN for f in res.features)
    ev = evaluation.evaluate_analysis(res, "test")
    auc = ev["summary"].set_index("predictor").loc["model", "roc_auc"]
    assert auc > 0.6, auc
    evaluation.evaluate_analysis(res, "holdout")
    out = interpretation.interpret(res)
    assert "ablation" in out and len(out["ablation"])
    reporting.model_card(res, ev)
    reporting.label_report(pop)
    reporting.export_label_audit_sample(pop, n=60)
    reporting.data_dictionary()
    reporting.tripod_checklist()
    reporting.export_dataset(pop)
    print("Test AUC:", round(auc, 3))
    print(ev["comparisons"].round(3).to_string())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
