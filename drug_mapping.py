"""
drug_mapping.py — Drug normalisation and investigational-agent identification. (Issue 5)

v6 problems fixed
  * Coverage: ~100 hand-entered drugs -> ChEMBL drug_mechanism (thousands of clinical
    candidates, including research codes such as "AZD9291"), with the seed lexicon as
    fallback.
  * Substring matching ("ar" in "carboplatin", "t-dxd" in any string) -> whole-token
    n-gram matching.
  * Combinations: v6 took the first dictionary hit, so "pembrolizumab + carboplatin"
    vs "carboplatin" could be coded as either agent. Now the investigational agent is an
    agent given in an EXPERIMENTAL arm, absent from comparator arms, preferring
    non-backbone drugs.
  * `drug_is_mapped` is no longer a model feature. Unmapped trials are excluded from the
    modelling population and the mapping gap is audited (leakage.audit_mapping_bias).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from config import log
from drug_target_db import (ADC_NAME_PARTS, BACKBONE_AGENTS, BRAND_ALIASES, CHECKPOINT_TARGETS,
                            CYTOTOXIC_MECH_RE, CYTOTOXIC_MOAS, HORMONAL_TARGETS, NON_DRUG_TERMS,
                            SEED_DRUGS)

DRUG_INTERVENTION_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT", "GENETIC"}
EXPERIMENTAL_ARMS = {"EXPERIMENTAL", "OTHER", ""}
COMPARATOR_ARMS = {"ACTIVE_COMPARATOR", "PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION"}
_SALTS = re.compile(r"\b(hydrochloride|dihydrochloride|hcl|mesylate|mesilate|dimesylate|maleate|"
                    r"tosylate|sulfate|sulphate|sodium|potassium|citrate|phosphate|acetate|malate|"
                    r"tartrate|besylate|fumarate|succinate|monohydrate|hydrate|injection|tablets?|"
                    r"capsules?|oral|iv|intravenous|for injection)\b")
MAX_TARGETS = 5


def norm_name(text: str) -> str:
    t = str(text).lower().replace("®", " ").replace("™", " ")
    t = re.sub(r"[^a-z0-9\-\s]", " ", t)
    t = _SALTS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass(frozen=True)
class DrugRecord:
    canonical: str
    targets: tuple[str, ...]
    modality: str
    moa: str
    source: str
    is_backbone: bool = False


def classify_modality(molecule_type: str | None, name: str) -> str:
    n = name.lower()
    if any(p in n for p in ADC_NAME_PARTS):
        return "adc"
    mt = (molecule_type or "").lower()
    if mt == "antibody" or n.endswith("mab"):
        return "antibody"
    if mt == "small molecule":
        return "small_molecule"
    if mt in ("protein", "enzyme"):
        return "protein"
    if mt == "cell":
        return "cell_therapy"
    if mt == "oligonucleotide":
        return "oligonucleotide"
    return "other"


def stratum_for(rec: DrugRecord | None) -> str | None:
    if rec is None:
        return None
    t = set(rec.targets)
    if t & CHECKPOINT_TARGETS:
        return "io_checkpoint"
    if rec.moa in CYTOTOXIC_MOAS or re.search(CYTOTOXIC_MECH_RE, rec.moa or ""):
        return "cytotoxic"
    if t & HORMONAL_TARGETS:
        return "hormonal"
    return "targeted"


class DrugLexicon:
    def __init__(self, chembl: tuple[pd.DataFrame, pd.DataFrame] | None = None):
        self.entries: dict[str, DrugRecord] = {}
        self.disagreements: list[dict] = []
        if chembl is not None:
            self._add_chembl(*chembl)
        self._add_seed()
        self.max_ngram = min(6, max((len(k.split()) for k in self.entries), default=1))
        log.info(f"Drug lexicon: {len(self.entries)} names "
                 f"({sum(r.source == 'chembl' for r in self.entries.values())} from ChEMBL)")

    def _add_chembl(self, names: pd.DataFrame, mech: pd.DataFrame) -> None:
        by_parent = {}
        for pid, grp in mech.groupby("parent_molregno"):
            genes = tuple(sorted(set(grp["gene"].dropna())))[:MAX_TARGETS]
            if genes:
                by_parent[pid] = (genes, str(grp["mechanism_of_action"].dropna().iloc[0]).lower()
                                  if grp["mechanism_of_action"].notna().any() else "")
        for row in names.itertuples(index=False):
            if row.parent_molregno not in by_parent:
                continue
            key = norm_name(row.name)
            if len(key) < 4 or key in self.entries:
                continue
            genes, moa = by_parent[row.parent_molregno]
            canonical = norm_name(row.pref_name or row.name)
            self.entries[key] = DrugRecord(canonical, genes, classify_modality(row.molecule_type, canonical),
                                           moa, "chembl", canonical in BACKBONE_AGENTS)

    def _add_seed(self) -> None:
        for name, (targets, modality, moa) in SEED_DRUGS.items():
            key = norm_name(name)
            seed = DrugRecord(key, targets, modality, moa, "seed", key in BACKBONE_AGENTS)
            existing = self.entries.get(key)
            if existing is None:
                self.entries[key] = seed
                continue
            if set(existing.targets) != set(targets):
                self.disagreements.append({"drug": key, "chembl": existing.targets, "seed": targets})
            if modality == "adc" and existing.modality != "adc":
                self.entries[key] = DrugRecord(existing.canonical, existing.targets, "adc",
                                               existing.moa, existing.source, existing.is_backbone)
            if moa in CYTOTOXIC_MOAS:   # keep a stable cytotoxic MoA tag for stratification
                e = self.entries[key]
                self.entries[key] = DrugRecord(e.canonical, e.targets, e.modality, moa, e.source, e.is_backbone)
        for brand, generic in BRAND_ALIASES.items():
            g = self.entries.get(norm_name(generic))
            if g is not None:
                self.entries.setdefault(norm_name(brand), g)

    def find(self, text: str) -> list[DrugRecord]:
        tokens = norm_name(text).split()
        found, i = [], 0
        while i < len(tokens):
            for n in range(min(self.max_ngram, len(tokens) - i), 0, -1):
                rec = self.entries.get(" ".join(tokens[i:i + n]))
                if rec is not None:
                    found.append(rec)
                    i += n
                    break
            else:
                i += 1
        return found


def build_lexicon(use_chembl: bool = True) -> DrugLexicon:
    chembl = None
    if use_chembl:
        from data_sources import load_chembl_lexicon
        chembl = load_chembl_lexicon()
        if chembl is None:
            log.warning("ChEMBL lexicon unavailable: falling back to the seed lexicon only. "
                        "Coverage will be low; do not use this configuration for the paper.")
    return DrugLexicon(chembl)


def map_trial(row, lexicon: DrugLexicon) -> dict:
    arm_types = row.get("arm_types") or {}
    exp_arms = {a for a, t in arm_types.items() if t in EXPERIMENTAL_ARMS}
    comp_arms = {a for a, t in arm_types.items() if t in COMPARATOR_ARMS}
    agents: dict[str, tuple[DrugRecord, bool, bool]] = {}
    unmapped = []
    for iv in row.get("interventions") or []:
        if iv.get("type", "").upper() not in DRUG_INTERVENTION_TYPES:
            continue
        names = [iv.get("name", "")] + list(iv.get("other_names") or [])
        recs = [r for n in names for r in lexicon.find(n)]
        arms = set(iv.get("arms") or [])
        in_exp = (not arms) or bool(arms & exp_arms) or not arm_types
        in_comp = bool(arms & comp_arms)
        if not recs and not any(t in iv.get("name", "").lower() for t in NON_DRUG_TERMS):
            unmapped.append(iv.get("name", ""))
        for r in recs:
            prev = agents.get(r.canonical)
            agents[r.canonical] = (r, (prev[1] if prev else False) or in_exp, (prev[2] if prev else False) or in_comp)

    experimental_only = [r for r, e, c in agents.values() if e and not c]
    investigational = [r for r in experimental_only if not r.is_backbone]
    if not investigational:
        investigational = experimental_only or [r for r, _, _ in agents.values() if not r.is_backbone]
    primary = investigational[0] if investigational else None
    return {
        "investigational_drug": primary.canonical if primary else None,
        "drug_targets": list(primary.targets) if primary else [],
        "primary_target": primary.targets[0] if primary else None,
        "modality": primary.modality if primary else None,
        "moa": primary.moa if primary else None,
        "drug_source": primary.source if primary else None,
        "stratum": stratum_for(primary),
        "n_investigational_agents": len({r.canonical for r in investigational}),
        "is_combination": int(len(agents) > 1),
        "backbone_agents": sorted(r.canonical for r, _, _ in agents.values() if r.is_backbone),
        "unmapped_interventions": unmapped,
        "has_unmapped_experimental_agent": int(bool(unmapped)),
    }


def map_all(df: pd.DataFrame, lexicon: DrugLexicon) -> pd.DataFrame:
    mapped = pd.DataFrame([map_trial(r, lexicon) for _, r in df.iterrows()], index=df.index)
    out = pd.concat([df, mapped], axis=1)
    rate = out["investigational_drug"].notna().mean()
    log.info(f"Drug mapping: {rate:.1%} of trials have an identified investigational agent; "
             f"strata={out['stratum'].value_counts(dropna=False).to_dict()}")
    if lexicon.disagreements:
        log.info(f"Seed vs ChEMBL target disagreements: {len(lexicon.disagreements)} (see reports)")
    return out
