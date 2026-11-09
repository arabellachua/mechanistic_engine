"""
features.py — Registry-driven raw features. (Issues 2, 3, 6, 8)

What changed
  * No hand-weighted composite scores (CDS/TDS/BFS/...) are fed to models. Models learn
    their own weighting from raw features; domain-level summaries are computed after
    training from SHAP values (interpretation.py), so they are data-derived.
  * Every feature is defined in config.FEATURE_REGISTRY with a leakage class.
  * Knowledge features are time-bounded to the trial's start year: PubMed counts use
    publication date < start year, ChEMBL potency uses document year < start year, and
    GWAS hits use association publication date < start year.
  * IO context is DISEASE-level (TMB, MSI-H fraction, leukocyte fraction of the cancer
    type) instead of v8's per-target constants (a "TMB relevance of EGFR" is not a
    measurable quantity).
  * Missing data stay NaN. LightGBM handles NaN natively; linear models impute inside
    a fitted pipeline with explicit missingness indicators. No fillna(-1).
  * Multi-target drugs: gene-level features are averaged over up to 5 targets.
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import FEATURE_REGISTRY, TRIAL_DESIGN, log
from normalize import DISEASE_BY_KEY

GENE_LEVEL = [k for k, v in FEATURE_REGISTRY.items() if v["level"] in ("target", "pair")]

# Gene symbols that are ordinary words: query the protein name instead.
PUBMED_GENE_TERMS = {
    "AR": '"androgen receptor"[tiab]', "MET": '("c-Met"[tiab] OR "MET amplification"[tiab] OR "MET exon"[tiab])',
    "RET": '("RET fusion"[tiab] OR "RET rearrangement"[tiab] OR "RET kinase"[tiab] OR "RET mutation"[tiab])',
    "KIT": '("c-KIT"[tiab] OR "KIT mutation"[tiab] OR CD117[tiab])', "SRC": '"Src kinase"[tiab]',
    "ESR1": '("estrogen receptor"[tiab] OR ESR1[tiab])', "TUBB": "tubulin[tiab]",
    "ALK": '("ALK"[tiab] AND (kinase[tiab] OR rearrangement[tiab] OR fusion[tiab]))',
    "MTOR": '(mTOR[tiab] OR "mammalian target of rapamycin"[tiab])', "CD274": '("PD-L1"[tiab] OR CD274[tiab])',
    "PDCD1": '("PD-1"[tiab] OR PDCD1[tiab])', "ERBB2": '(HER2[tiab] OR ERBB2[tiab])',
    "MS4A1": "CD20[tiab]", "TNFRSF8": "CD30[tiab]", "TACSTD2": '(TROP2[tiab] OR "Trop-2"[tiab])',
}


def _gene_term(gene: str) -> str:
    return PUBMED_GENE_TERMS.get(gene, f'"{gene}"[tiab]')


# ── cBioPortal pre-computation ───────────────────────────────────────────────
def precompute_cbio(genes: list[str], diseases: list[str], thorsson: pd.Series | None) -> tuple[dict, dict]:
    import clients
    entrez = clients.cbio_entrez(genes)
    pair, ctx = {}, {}
    for dkey in diseases:
        d = DISEASE_BY_KEY.get(dkey)
        if d is None or not d.tcga:
            continue
        agg = defaultdict(lambda: np.zeros(3))
        n_seq = n_cna = 0
        tmb, msi = [], []
        for study in d.tcga:
            res = clients.cbio_study_alterations(study, list(entrez.values()))
            if res is None:
                continue
            n_seq += res["n_seq"]
            n_cna += res["n_cna"]
            for eid, c in res["counts"].items():
                agg[eid] += np.array(c)
            tmb += clients.cbio_clinical_values(study, "TMB_NONSYNONYMOUS")
            msi += clients.cbio_clinical_values(study, "MSI_SCORE_MANTIS")
        for sym, eid in entrez.items():
            m, a, dl = agg.get(eid, np.zeros(3))
            pair[(sym, dkey)] = {
                "somatic_mut_freq": m / n_seq if n_seq else np.nan,
                "cna_amp_freq": a / n_cna if n_cna else np.nan,
                "cna_homdel_freq": dl / n_cna if n_cna else np.nan,
            }
        lf = np.nan
        if thorsson is not None:
            codes = [s.split("_")[0].upper() for s in d.tcga]
            vals = [thorsson.get(c) for c in codes if c in thorsson.index]
            lf = float(np.nanmean(vals)) if vals else np.nan
        ctx[dkey] = {
            "tcga_tmb_median_log": float(np.log1p(np.median(tmb))) if tmb else np.nan,
            "tcga_msi_high_frac": float(np.mean(np.array(msi) > 0.4)) if msi else np.nan,
            "leukocyte_fraction": lf,
        }
    return pair, ctx


# ── Feature builder ──────────────────────────────────────────────────────────
class FeatureBuilder:
    def __init__(self, sources: dict, cbio_pair: dict, cbio_ctx: dict, use_pubmed: bool = True):
        self.s = sources
        self.cbio_pair, self.cbio_ctx = cbio_pair, cbio_ctx
        self.use_pubmed = use_pubmed
        act = sources.get("chembl_activity")
        self.chembl = {g: grp for g, grp in act.groupby("gene")} if act is not None else None
        gw = sources.get("gwas")
        self.gwas = {g: grp for g, grp in gw.groupby("gene")} if gw is not None else None

    # intrinsic, target-level
    @lru_cache(maxsize=None)
    def target_intrinsic(self, gene: str) -> dict:
        f = {}
        dm = self.s.get("depmap")
        if dm is not None and gene in dm.effect.columns:
            col = dm.effect[gene].dropna()
            f["depmap_pan_effect"] = float(col.mean())
            f["depmap_common_essential_frac"] = float((col < -0.5).mean())
        hpa = self.s.get("hpa")
        if hpa is not None and gene in hpa.index:
            row = hpa.loc[gene]
            for k in ("hpa_tissue_distribution", "hpa_tissue_specificity", "hpa_cancer_specificity",
                      "hpa_membrane", "hpa_secreted"):
                f[k] = float(row[k]) if pd.notna(row[k]) else np.nan
            rx = self.s.get("reactome")
            if rx is not None and isinstance(row.get("uniprot"), str):
                ids = [u.strip() for u in row["uniprot"].split(",")]
                f["reactome_pathways_log"] = float(np.log1p(max((rx.get(u, 0) for u in ids), default=0)))
        st = self.s.get("string")
        if st is not None and gene in st.index:
            f.update(st.loc[gene].to_dict())
        cgc = self.s.get("cgc")
        if cgc is not None:
            f["cgc_member"] = float(gene in cgc)
        return f

    # intrinsic, pair-level
    @lru_cache(maxsize=None)
    def pair_intrinsic(self, gene: str, disease: str) -> dict:
        f = dict(self.cbio_pair.get((gene, disease), {}))
        dm, d = self.s.get("depmap"), DISEASE_BY_KEY.get(disease)
        if dm is not None and d is not None and d.depmap_lineage and gene in dm.effect.columns:
            mask = dm.models["OncotreeLineage"].eq(d.depmap_lineage)
            if d.depmap_disease and "OncotreePrimaryDisease" in dm.models:
                mask &= dm.models["OncotreePrimaryDisease"].fillna("").str.contains(d.depmap_disease, regex=True)
            vals = dm.effect.loc[mask.values, gene].dropna()
            if len(vals) >= 5:
                pan = float(dm.effect[gene].mean())
                f["depmap_lineage_effect"] = float(vals.mean())
                f["depmap_lineage_dependent_frac"] = float((vals < -0.5).mean())
                f["depmap_lineage_selectivity"] = pan - float(vals.mean())
        return f

    # time-bounded knowledge (everything strictly before start_year)
    @lru_cache(maxsize=None)
    def time_bounded(self, gene: str, disease: str, start_year: int) -> dict:
        f = {}
        if self.chembl is not None:
            grp = self.chembl.get(gene)
            if grp is not None:
                prior = grp[(grp["year"] < start_year) & (grp["pchembl"] >= 6)]
                f["chembl_potent_compounds_log"] = float(np.log1p(prior["molregno"].nunique()))
                f["chembl_max_pchembl"] = float(prior["pchembl"].max()) if len(prior) else np.nan
                f["chembl_history_years"] = float(start_year - prior["year"].min()) if len(prior) else 0.0
            else:
                f["chembl_potent_compounds_log"], f["chembl_history_years"] = 0.0, 0.0
        if self.gwas is not None:
            grp = self.gwas.get(gene)
            d = DISEASE_BY_KEY.get(disease)
            if grp is None:
                f["gwas_cancer_hits_log"] = 0.0
                f["gwas_disease_hits_log"] = 0.0 if d else np.nan
            else:
                prior = grp[grp["year"] < start_year]
                f["gwas_cancer_hits_log"] = float(np.log1p(prior["is_cancer"].sum()))
                f["gwas_disease_hits_log"] = (float(np.log1p(prior["trait"].str.contains(d.pattern, regex=True).sum()))
                                              if d else np.nan)
        if self.use_pubmed:
            from clients import pubmed_count
            gt = _gene_term(gene)
            n_t = pubmed_count(gt, start_year)
            f["pubmed_target_log"] = float(np.log1p(n_t)) if n_t is not None else np.nan
            d = DISEASE_BY_KEY.get(disease)
            if d is not None and d.pubmed:
                pair_term = f"({gt}) AND ({d.pubmed})"
                n_p = pubmed_count(pair_term, start_year)
                n_r = pubmed_count(pair_term, start_year, since_year=start_year - 3)
                f["pubmed_pair_log"] = float(np.log1p(n_p)) if n_p is not None else np.nan
                f["pubmed_pair_recent_frac"] = (n_r / n_p) if (n_p and n_r is not None) else np.nan
        return f

    def trial_features(self, row) -> dict:
        genes = [g for g in (row.get("drug_targets") or []) if g != "DNA"][:5]
        disease, year = row["disease"], int(row["start_year"])
        per_gene = [{**self.target_intrinsic(g), **self.pair_intrinsic(g, disease),
                     **self.time_bounded(g, disease, year)} for g in genes]
        f = {}
        for k in GENE_LEVEL:
            vals = [pg[k] for pg in per_gene if k in pg and pd.notna(pg[k])]
            f[k] = float(np.mean(vals)) if vals else np.nan
        f.update(self.cbio_ctx.get(disease, {}))
        mod = row.get("modality") or "other"
        f["modality_small_molecule"] = float(mod == "small_molecule")
        f["modality_antibody"] = float(mod == "antibody")
        f["modality_adc"] = float(mod == "adc")
        f["modality_other"] = float(mod not in ("small_molecule", "antibody", "adc"))
        f["n_targets"] = float(len(row.get("drug_targets") or []))
        f.update(design_features(row))
        f["knowledge_cutoff_year"] = year - 1
        return f


# ── Trial-design features (stage 2 only) ─────────────────────────────────────
_EP = {
    "endpoint_os": re.compile(r"overall survival|\bos\b", re.I),
    "endpoint_pfs": re.compile(r"progression[\s-]free|\bpfs\b|event[\s-]free|disease[\s-]free|\befs\b|\bdfs\b", re.I),
    "endpoint_response": re.compile(r"response|\borr\b|remission|\bpcr\b|pathologic(al)? complete", re.I),
}


def design_features(row) -> dict:
    """Registered design. Note: CT.gov records can be edited after the fact (outcome
    switching); the history API should be used to recover the originally registered
    primary outcome for the paper."""
    outcomes = " ".join(row.get("primary_outcomes") or [])
    arm_types = row.get("arm_types") or {}
    f = {k: float(bool(rx.search(outcomes))) for k, rx in _EP.items()}
    f.update({
        "biomarker_selected": float(row.get("biomarker_selected", 0)),
        "biomarker_matches_target": float(row.get("biomarker_matches_target", 0)),
        "is_randomized": float(row.get("allocation") == "RANDOMIZED"),
        "placebo_controlled": float("PLACEBO_COMPARATOR" in arm_types.values()),
        "n_arms": float(row.get("n_arms") or 0),
        "is_combination": float(row.get("is_combination", 0)),
        "n_investigational_agents": float(row.get("n_investigational_agents", 0)),
        "industry_sponsor": float(row.get("sponsor_class") == "INDUSTRY"),
    })
    return f


def compute_features(df: pd.DataFrame, sources: dict, use_pubmed: bool = True,
                     use_cbio: bool = True) -> pd.DataFrame:
    genes = sorted({g for ts in df["drug_targets"] for g in ts if g != "DNA"})
    diseases = sorted(df["disease"].unique())
    cbio_pair, cbio_ctx = ({}, {})
    if use_cbio:
        try:
            cbio_pair, cbio_ctx = precompute_cbio(genes, diseases, sources.get("thorsson"))
        except Exception as exc:  # noqa: BLE001
            log.error(f"cBioPortal features failed: {exc}; TCGA features will be NaN")
    fb = FeatureBuilder(sources, cbio_pair, cbio_ctx, use_pubmed=use_pubmed)
    rows = [fb.trial_features(r) for _, r in tqdm(df.iterrows(), total=len(df), desc="Features")]
    feats = pd.DataFrame(rows, index=df.index)
    for k in FEATURE_REGISTRY:
        if k not in feats.columns and FEATURE_REGISTRY[k]["leakage_class"] != "outcome_proxy":
            feats[k] = np.nan
    miss = feats[[c for c in feats.columns if c in FEATURE_REGISTRY
                  and FEATURE_REGISTRY[c]["leakage_class"] != TRIAL_DESIGN]].isna().mean().sort_values(ascending=False)
    log.info(f"Feature missingness (top):\n{miss.head(15).round(3).to_string()}")
    return feats
