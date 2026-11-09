"""
normalize.py — Disease, phase, date and biomarker normalisation (single implementation).

v6 problems fixed
  * Two diverging normalisers (normalize.py and run_pipeline.normalize_trials) -> one.
  * Biomarker detection used bare substrings ("ar", "met", "ret", "er+"), so
    "carcinoma", "metastatic" and "treatment" flagged almost every trial as biomarker-
    selected. Now: word-boundary regexes that require a positivity/alteration context,
    searched only in the title and INCLUSION criteria (exclusion criteria mentioning
    EGFR mutations are the opposite of selection).
  * Each disease carries the TCGA studies, DepMap lineage and PubMed term used by
    features.py, so pair-level features are defined consistently.
  * Supportive-care trials (e.g. antiemetics in cancer patients) are removed by
    requiring primary purpose = TREATMENT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import EXCLUDE_PHASE1_2, log


@dataclass(frozen=True)
class Disease:
    key: str
    pattern: str
    tcga: tuple[str, ...] = ()
    depmap_lineage: str | None = None
    depmap_disease: str | None = None       # regex on DepMap OncotreePrimaryDisease
    pubmed: str | None = None


def _pa(code: str) -> str:
    return f"{code}_tcga_pan_can_atlas_2018"


# Priority order: specific before general (NSCLC before SCLC before "lung").
DISEASES: list[Disease] = [
    Disease("nsclc", r"non[\s-]?small[\s-]?cell|\bnsclc\b", (_pa("luad"), _pa("lusc")), "Lung", r"Non-Small Cell",
            '("non-small cell lung"[tiab] OR NSCLC[tiab])'),
    Disease("sclc", r"small[\s-]?cell lung|\bsclc\b", (), "Lung", r"(?<!Non-)Small Cell",
            '("small cell lung"[tiab] OR SCLC[tiab])'),
    Disease("breast", r"breast", (_pa("brca"),), "Breast", None, '"breast cancer"[tiab]'),
    Disease("colorectal", r"colorectal|\bcolon\b|\brectal\b|\bcrc\b", (_pa("coadread"),), "Bowel", None,
            '(colorectal[tiab] OR "colon cancer"[tiab] OR "rectal cancer"[tiab])'),
    Disease("pancreatic", r"pancrea", (_pa("paad"),), "Pancreas", None, '"pancreatic cancer"[tiab]'),
    Disease("gastric", r"gastric|stomach|gastro[\s-]?o?esophageal junction|\bgej\b", (_pa("stad"),),
            "Esophagus/Stomach", r"Stomach|Gastric|Esophagogastric", '(gastric[tiab] AND cancer[tiab])'),
    Disease("esophageal", r"o?esophag", (_pa("esca"),), "Esophagus/Stomach", r"Esophag",
            '(esophageal[tiab] AND cancer[tiab])'),
    Disease("melanoma", r"melanoma", (_pa("skcm"),), "Skin", r"Melanoma", "melanoma[tiab]"),
    Disease("rcc", r"renal cell|kidney cancer|\brcc\b", (_pa("kirc"), _pa("kirp")), "Kidney", None,
            '("renal cell"[tiab] OR RCC[tiab])'),
    Disease("hcc", r"hepatocellular|liver cancer|\bhcc\b", (_pa("lihc"),), "Liver", None,
            '(hepatocellular[tiab] OR HCC[tiab])'),
    Disease("biliary", r"cholangio|biliary|gallbladder", (_pa("chol"),), "Biliary Tract", None,
            '(cholangiocarcinoma[tiab] OR "biliary tract cancer"[tiab])'),
    Disease("ovarian", r"ovar|fallopian|primary peritoneal", (_pa("ov"),), "Ovary/Fallopian Tube", None,
            '"ovarian cancer"[tiab]'),
    Disease("endometrial", r"endometri|uterine", (_pa("ucec"),), "Uterus", None,
            '(endometrial[tiab] AND cancer[tiab])'),
    Disease("cervical", r"cervical cancer|cervix|cervical carcinoma", (_pa("cesc"),), "Cervix", None,
            '"cervical cancer"[tiab]'),
    Disease("prostate", r"prostat", (_pa("prad"),), "Prostate", None, '"prostate cancer"[tiab]'),
    Disease("bladder", r"bladder|urothelial", (_pa("blca"),), "Bladder/Urinary Tract", None,
            '(urothelial[tiab] OR "bladder cancer"[tiab])'),
    Disease("hnscc", r"head and neck|oropharyn|laryn|oral cavity|\bhnscc\b|\bscchn\b|hypopharyn",
            (_pa("hnsc"),), "Head and Neck", None, '("head and neck"[tiab] AND (cancer[tiab] OR carcinoma[tiab]))'),
    Disease("glioma", r"glioblastoma|glioma|\bgbm\b", (_pa("gbm"), _pa("lgg")), "CNS/Brain", r"Glioma|Glioblastoma",
            "(glioblastoma[tiab] OR glioma[tiab])"),
    Disease("thyroid", r"thyroid", (_pa("thca"),), "Thyroid", None, '"thyroid cancer"[tiab]'),
    Disease("mesothelioma", r"mesothelioma", (_pa("meso"),), "Pleura", None, "mesothelioma[tiab]"),
    Disease("sarcoma", r"sarcoma", (_pa("sarc"),), "Soft Tissue", None, "sarcoma[tiab]"),
    Disease("aml", r"acute myel|\baml\b", (_pa("laml"),), "Myeloid", r"Acute Myeloid",
            '("acute myeloid leukemia"[tiab] OR AML[tiab])'),
    Disease("cml", r"chronic myel|\bcml\b", (), "Myeloid", r"Chronic Myelo", '("chronic myeloid leukemia"[tiab] OR CML[tiab])'),
    Disease("all", r"acute lymphoblastic|acute lymphocytic|\bb-all\b|\bt-all\b", (), "Lymphoid", r"Lymphoblastic",
            '"acute lymphoblastic leukemia"[tiab]'),
    Disease("cll", r"chronic lymphocytic|\bcll\b|small lymphocytic", (), "Lymphoid", r"Chronic Lymphocytic",
            '("chronic lymphocytic leukemia"[tiab] OR CLL[tiab])'),
    Disease("myeloma", r"myeloma", (), "Lymphoid", r"Plasma Cell|Myeloma", '"multiple myeloma"[tiab]'),
    Disease("dlbcl", r"diffuse large|\bdlbcl\b", (_pa("dlbc"),), "Lymphoid", r"Diffuse Large",
            '("diffuse large B-cell"[tiab] OR DLBCL[tiab])'),
    Disease("hodgkin", r"(?<!non-)(?<!non )hodgkin", (), "Lymphoid", r"Hodgkin", '"Hodgkin lymphoma"[tiab]'),
    Disease("lymphoma", r"lymphoma", (), "Lymphoid", None, "lymphoma[tiab]"),
]
DISEASE_BY_KEY = {d.key: d for d in DISEASES}
GENERIC = [("solid_tumors", r"solid tumou?r|advanced (malignan|cancer)|metastatic cancer"),
           ("cancer_nos", r"cancer|neoplasm|carcinoma|tumou?r|malignan")]


def normalize_disease(conditions: list[str], mesh: list[str], title: str) -> tuple[str, int]:
    text = " ".join(list(conditions or []) + list(mesh or [])).lower()
    hits = [d.key for d in DISEASES if re.search(d.pattern, text)]
    if not hits:
        hits = [d.key for d in DISEASES if re.search(d.pattern, (title or "").lower())][:1]
    hits = list(dict.fromkeys(hits))
    if len(hits) >= 3:
        return "basket", len(hits)
    if hits:
        return hits[0], len(hits)
    for key, pat in GENERIC:
        if re.search(pat, text):
            return key, 0
    return "cancer_nos", 0


# ── Biomarkers ───────────────────────────────────────────────────────────────
POS = r"(?:[\s-]*(?:positive|\+|mutat\w*|mutant|amplif\w*|overexpress\w*|rearrang\w*|fusion|alteration|altered|translocation))"
BIOMARKERS: list[tuple[str, str | None, str]] = [
    ("EGFR mutation", "EGFR", r"\begfr" + POS + r"|\begfr[\s-]*(?:exon|t790m|l858r|del(?:etion)?\s*19|sensitizing)"),
    ("ALK fusion", "ALK", r"\balk" + POS + r"|\beml4[\s-]*alk\b"),
    ("ROS1 fusion", "ROS1", r"\bros1" + POS),
    ("RET fusion", "RET", r"\bret" + POS),
    ("NTRK fusion", "NTRK1", r"\bntrk\d?" + POS),
    ("BRAF V600", "BRAF", r"\bbraf" + POS + r"|\bbraf[\s-]*v600"),
    ("KRAS G12C", "KRAS", r"\bkras[\s-]*g12c\b"),
    ("KRAS mutation", "KRAS", r"\bkras" + POS),
    ("HER2 positive", "ERBB2", r"\bher[\s-]?2" + POS + r"|\berbb2" + POS + r"|\bher2[\s-]*(?:3\+|ihc\s*3)"),
    ("MET alteration", "MET", r"\bc-met\b|\bmet[\s-]*(?:exon\s*14|amplif\w*|overexpress\w*)"),
    ("PIK3CA mutation", "PIK3CA", r"\bpik3ca" + POS),
    ("FGFR alteration", "FGFR2", r"\bfgfr\d?" + POS),
    ("IDH mutation", "IDH1", r"\bidh[12]?" + POS),
    ("FLT3 mutation", "FLT3", r"\bflt3" + POS + r"|\bflt3[\s-]*itd\b"),
    ("BRCA/HRD", "BRCA1", r"\bg?brca[12]?" + POS + r"|homologous recombination (?:deficien|repair)|\bhrd\b"),
    ("MSI-H/dMMR", None, r"\bmsi[\s-]*(?:h|high)\b|microsatellite instability[\s-]*high|\bdmmr\b|mismatch repair[\s-]*deficien"),
    ("TMB-high", None, r"\btmb[\s-]*(?:h|high)\b|high tumou?r mutational burden"),
    ("PD-L1 positive", "CD274", r"\bpd-?l1\b[^.;]{0,40}(?:positive|≥|>=|>\s*\d|tps|cps|expression)"),
    ("ER/HR positive", "ESR1", r"\b(?:er|hr|hormone receptor|estrogen receptor)\)?[\s-]*(?:positive|\+)"),
    ("AR positive", "AR", r"androgen receptor[\s-]*(?:positive|expression|\+)"),
    ("CD20 positive", "MS4A1", r"\bcd20[\s-]*(?:positive|\+|expressing)"),
    ("CD19 positive", "CD19", r"\bcd19[\s-]*(?:positive|\+|expressing)"),
    ("CD30 positive", "TNFRSF8", r"\bcd30[\s-]*(?:positive|\+|expressing)"),
    ("CD38 positive", "CD38", r"\bcd38[\s-]*(?:positive|\+|expressing)"),
    ("CD33 positive", "CD33", r"\bcd33[\s-]*(?:positive|\+|expressing)"),
    ("Claudin 18.2", "CLDN18", r"claudin[\s-]*18\.?2|\bcldn18"),
    ("Trop-2", "TACSTD2", r"\btrop-?2[\s-]*(?:positive|expression|high)"),
    ("PSMA positive", "FOLH1", r"\bpsma[\s-]*(?:positive|avid|expression|pet)"),
    ("DLL3", "DLL3", r"\bdll3[\s-]*(?:positive|expression)"),
    ("Triple negative", None, r"triple[\s-]*negative|\btnbc\b"),
]
_BIOMARKER_RE = [(name, gene, re.compile(p, re.I)) for name, gene, p in BIOMARKERS]


def inclusion_text(eligibility: str) -> str:
    m = re.search(r"exclusion criteria", eligibility or "", re.I)
    return (eligibility or "")[: m.start()] if m else (eligibility or "")


def extract_biomarkers(title: str, eligibility: str) -> list[tuple[str, str | None]]:
    text = f"{title or ''}\n{inclusion_text(eligibility)}"
    return [(name, gene) for name, gene, rx in _BIOMARKER_RE if rx.search(text)]


# ── Basic fields ─────────────────────────────────────────────────────────────
def phase_group(phases) -> str:
    ph = set(phases or [])
    if "PHASE3" in ph:
        return "PHASE3"
    if ph == {"PHASE2"}:
        return "PHASE2"
    if ph == {"PHASE1", "PHASE2"}:
        return "PHASE1_2"
    return "OTHER"


def _year(date_str) -> float:
    try:
        return float(str(date_str)[:4])
    except (TypeError, ValueError):
        return np.nan


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phase_group"] = df["phases"].apply(phase_group)
    df["start_year"] = df["start_date"].apply(_year)
    df["primary_completion_year"] = df["primary_completion_date"].apply(_year)
    for c in ("start_date", "primary_completion_date", "completion_date", "results_first_posted"):
        df[c + "_dt"] = pd.to_datetime(df[c], errors="coerce")
    return df


def apply_scope_filters(df: pd.DataFrame, phases) -> tuple[pd.DataFrame, dict]:
    flow = {"ctgov_records": len(df)}
    df = df[df["study_type"].eq("INTERVENTIONAL")]
    flow["interventional"] = len(df)
    df = df[df["primary_purpose"].eq("TREATMENT")]
    flow["treatment_purpose"] = len(df)
    keep = set(phases) | (set() if EXCLUDE_PHASE1_2 else {"PHASE1_2"})
    df = df[df["phase_group"].isin(keep)]
    flow["phase_in_scope"] = len(df)
    df = df[df["start_year"].notna()]
    flow["has_start_date"] = len(df)
    has_drug = df["interventions"].apply(
        lambda ivs: any(i.get("type", "").upper() in {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT", "GENETIC"}
                        for i in ivs or []))
    df = df[has_drug]
    flow["has_drug_intervention"] = len(df)
    return df.reset_index(drop=True), flow


def normalize_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dis = df.apply(lambda r: normalize_disease(r["conditions"], r.get("mesh_conditions", []), r["title"]), axis=1)
    df["disease"] = dis.str[0]
    df["n_diseases_matched"] = dis.str[1]
    bms = df.apply(lambda r: extract_biomarkers(r["title"], r["eligibility"]), axis=1)
    df["biomarkers"] = bms.apply(lambda b: [n for n, _ in b])
    df["biomarker_genes"] = bms.apply(lambda b: sorted({g for _, g in b if g}))
    df["biomarker_selected"] = bms.apply(lambda b: int(bool(b)))
    log.info(f"Diseases: {df['disease'].value_counts().head(12).to_dict()}")
    log.info(f"Biomarker-selected trials: {df['biomarker_selected'].mean():.1%}")
    return df


def add_biomarker_target_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["biomarker_matches_target"] = df.apply(
        lambda r: int(bool(set(r.get("biomarker_genes") or []) & set(r.get("drug_targets") or []))), axis=1)
    return df
