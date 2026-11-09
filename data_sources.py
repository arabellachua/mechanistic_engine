"""
data_sources.py — Versioned bulk data that replaces every hand-curated dictionary in
v6–v8 (OT_CURATED, DEPMAP_CHRONOS, DISGENET_CURATED, DGIDB_CURATED, GWAS_CURATED,
OMNIPATH_CURATED, COSMIC tiers, IO_FEATURES).  (Issue 2)

Rules
  * Each file is downloaded once, hashed (sha256) and recorded in raw/manifest.json with
    its pinned release, URL and timestamp. Cite the manifest in the paper's methods.
  * Loaders never invent values. A missing source yields NaN features; nothing is
    replaced by a "reasonable default".
  * Outcome-derived resources are not loaded as features: Open Targets association
    scores (contain the ChEMBL clinical-phase datasource), DGIdb/DrugBank drug counts,
    ChEMBL max_phase, and HPA's "FDA approved drug targets" protein class.
  * Time-varying knowledge (ChEMBL activities, GWAS associations) is kept with its
    publication year so features.py can bound it to the trial start date. (Issue 3)
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (COSMIC_CGC_FILE, RAW_DIR, RELEASES, THORSSON_FILE, URLS, log)

MANIFEST = RAW_DIR / "manifest.json"


# ── Manifest ─────────────────────────────────────────────────────────────────
def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _read_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def record_manifest(name: str, url: str, path: Path, release: str) -> None:
    manifest = _read_manifest()
    manifest[name] = {
        "url": url, "path": str(path), "release": release,
        "sha256": _sha256(path), "bytes": path.stat().st_size,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def fetch(name: str, url: str, filename: str, release: str, required: bool = False) -> Path | None:
    """Download url -> raw/filename once, then verify the sha256 against the manifest."""
    dest = RAW_DIR / filename
    manifest = _read_manifest()
    if dest.exists():
        if name in manifest and manifest[name]["sha256"] != _sha256(dest):
            raise RuntimeError(f"{dest} changed since it was recorded; the analysis is no longer "
                               f"reproducible. Restore the original file or re-record deliberately.")
        if name not in manifest:
            record_manifest(name, url, dest, release)
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        log.info(f"Downloading {name} ({release}) from {url}")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)
        tmp.rename(dest)
        record_manifest(name, url, dest, release)
        return dest
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        msg = f"Could not download {name}: {exc}"
        if required:
            raise RuntimeError(msg) from exc
        log.error(msg + " -> features from this source will be NaN")
        return None


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


# ── DepMap (CRISPR gene effect) ──────────────────────────────────────────────
@dataclass
class DepMapData:
    effect: pd.DataFrame      # models x genes (Chronos)
    models: pd.DataFrame      # ModelID -> OncotreeLineage, OncotreePrimaryDisease


def load_depmap(release: str = RELEASES["depmap"]) -> DepMapData | None:
    index = fetch("depmap_index", URLS["depmap_index"], "depmap_files_index.csv", "live index")
    if index is None:
        return None
    files = pd.read_csv(index)
    rel_col = next((c for c in files.columns if c.lower() == "release"), None)
    if rel_col is None or "filename" not in files.columns or "url" not in files.columns:
        log.error("Unexpected DepMap index format; update data_sources.load_depmap")
        return None
    sub = files[files[rel_col] == release]
    urls = dict(zip(sub["filename"], sub["url"]))
    if "CRISPRGeneEffect.csv" not in urls or "Model.csv" not in urls:
        log.error(f"DepMap release '{release}' lacks CRISPRGeneEffect.csv/Model.csv")
        return None
    s = _slug(release)
    eff_p = fetch("depmap_gene_effect", urls["CRISPRGeneEffect.csv"], f"depmap_{s}_CRISPRGeneEffect.csv", release)
    mdl_p = fetch("depmap_models", urls["Model.csv"], f"depmap_{s}_Model.csv", release)
    if eff_p is None or mdl_p is None:
        return None
    effect = pd.read_csv(eff_p, index_col=0).astype("float32")
    effect.columns = [c.split(" (")[0] for c in effect.columns]
    effect = effect.loc[:, ~effect.columns.duplicated()]
    keep = {"ModelID", "OncotreeLineage", "OncotreePrimaryDisease"}
    models = pd.read_csv(mdl_p, usecols=lambda c: c in keep).set_index("ModelID")
    models = models.reindex(effect.index)
    return DepMapData(effect=effect, models=models)


# ── Human Protein Atlas ──────────────────────────────────────────────────────
_DIST = {"not detected": 0, "detected in single": 1, "detected in some": 2,
         "detected in many": 3, "detected in all": 4}
_SPEC = {"low tissue specificity": 0, "tissue enhanced": 1, "group enriched": 2, "tissue enriched": 3,
         "low cancer specificity": 0, "cancer enhanced": 1, "cancer enriched": 3}


def load_hpa(version: str = RELEASES["hpa"]) -> pd.DataFrame | None:
    p = fetch("hpa", URLS["hpa_tsv"].format(v=version), f"hpa_v{version}_proteinatlas.tsv.zip", f"HPA v{version}")
    if p is None:
        return None
    raw = pd.read_csv(p, sep="\t", compression="zip", low_memory=False)

    def col(name):
        return raw[name].astype(str).str.lower() if name in raw.columns else pd.Series("", index=raw.index)

    pclass, subcell = col("Protein class"), col("Subcellular location")
    out = pd.DataFrame({
        "gene": raw["Gene"],
        "uniprot": raw.get("Uniprot"),
        "hpa_tissue_distribution": col("RNA tissue distribution").map(_DIST),
        "hpa_tissue_specificity": col("RNA tissue specificity").map(_SPEC),
        "hpa_cancer_specificity": col("RNA cancer specificity").map(_SPEC),
        # Deliberately NOT used: "fda approved drug targets" and "cancer-related genes"
        # protein classes — both encode clinical/curation outcomes.
        "hpa_membrane": (pclass.str.contains("predicted membrane proteins")
                         | subcell.str.contains("plasma membrane")).astype(float),
        "hpa_secreted": pclass.str.contains("predicted secreted proteins").astype(float),
    })
    return out.drop_duplicates("gene").set_index("gene")


# ── STRING ───────────────────────────────────────────────────────────────────
def load_string(version: str = RELEASES["string"]) -> pd.DataFrame | None:
    links = fetch("string_links", URLS["string_links"].format(v=version), f"string_v{version}_links.txt.gz", f"STRING v{version}")
    info = fetch("string_info", URLS["string_info"].format(v=version), f"string_v{version}_info.txt.gz", f"STRING v{version}")
    if links is None or info is None:
        return None
    names = pd.read_csv(info, sep="\t", usecols=[0, 1])
    names.columns = ["protein", "gene"]
    lk = pd.read_csv(links, sep=" ", dtype={"combined_score": "int16"})
    hc = lk[lk["combined_score"] >= 700]
    agg = hc.groupby("protein1")["combined_score"].agg(["size", "sum"])
    agg = agg.join(names.set_index("protein"), how="left").dropna(subset=["gene"])
    return pd.DataFrame({
        "string_degree_log": np.log1p(agg["size"].values),
        "string_weighted_degree_log": np.log1p(agg["sum"].values / 1000.0),
    }, index=agg["gene"].values).groupby(level=0).max()


# ── Reactome ─────────────────────────────────────────────────────────────────
def load_reactome() -> pd.Series | None:
    p = fetch("reactome", URLS["reactome_uniprot"], "reactome_UniProt2Reactome_All_Levels.txt", RELEASES["reactome"])
    if p is None:
        return None
    df = pd.read_csv(p, sep="\t", header=None, names=["uniprot", "pathway", "url", "name", "evidence", "species"])
    return df[df["species"] == "Homo sapiens"].groupby("uniprot")["pathway"].nunique()


# ── GWAS Catalog (dated) ─────────────────────────────────────────────────────
CANCER_TRAIT_RE = re.compile(r"cancer|carcinoma|neoplasm|tumou?r|lymphoma|leukemia|leukaemia|"
                             r"melanoma|glioma|myeloma|sarcoma|blastoma|mesothelioma", re.I)


def load_gwas() -> pd.DataFrame | None:
    p = fetch("gwas_catalog", URLS["gwas_associations"], "gwas_catalog_associations_ontology.tsv", RELEASES["gwas_catalog"])
    if p is None:
        return None
    usecols = ["DATE", "DISEASE/TRAIT", "MAPPED_GENE", "MAPPED_TRAIT", "P-VALUE"]
    df = pd.read_csv(p, sep="\t", usecols=lambda c: c in usecols, low_memory=False)
    df["pval"] = pd.to_numeric(df["P-VALUE"], errors="coerce")
    df = df[df["pval"] <= 5e-8].copy()
    df["year"] = pd.to_datetime(df["DATE"], errors="coerce").dt.year
    df["trait"] = (df["DISEASE/TRAIT"].fillna("") + " | " + df["MAPPED_TRAIT"].fillna("")).str.lower()
    df["gene"] = df["MAPPED_GENE"].fillna("").str.split(r"\s*(?:,|;| - | x )\s*")
    df = df.explode("gene")
    df = df[df["gene"].str.fullmatch(r"[A-Z0-9][A-Z0-9\-\.]+", na=False)]
    df["is_cancer"] = df["trait"].str.contains(CANCER_TRAIT_RE)
    return df[["gene", "year", "trait", "is_cancer"]].reset_index(drop=True)


# ── ChEMBL (SQLite dump) ─────────────────────────────────────────────────────
def chembl_db(version: str = RELEASES["chembl"]) -> Path | None:
    db = RAW_DIR / f"chembl_{version}.db"
    if db.exists():
        return db
    tgz = fetch("chembl_sqlite", URLS["chembl_sqlite"].format(v=version), f"chembl_{version}_sqlite.tar.gz", f"ChEMBL {version}")
    if tgz is None:
        return None
    log.info("Extracting ChEMBL SQLite (large; one-off)")
    with tarfile.open(tgz, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(".db")), None)
        if member is None:
            log.error("No .db file inside ChEMBL archive")
            return None
        with tar.extractfile(member) as src, open(db, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
    return db


def load_chembl_lexicon(version: str = RELEASES["chembl"]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(names, mechanisms) for drug normalisation. max_phase is intentionally not selected."""
    cache_n = RAW_DIR / f"chembl_{version}_names.parquet"
    cache_m = RAW_DIR / f"chembl_{version}_mechanisms.parquet"
    if cache_n.exists() and cache_m.exists():
        return pd.read_parquet(cache_n), pd.read_parquet(cache_m)
    db = chembl_db(version)
    if db is None:
        return None
    parent_cte = """
        WITH parent AS (
          SELECT md.molregno, COALESCE(mh.parent_molregno, md.molregno) AS parent_molregno
          FROM molecule_dictionary md LEFT JOIN molecule_hierarchy mh ON md.molregno = mh.molregno)
    """
    names_sql = parent_cte + """
        , names AS (
          SELECT molregno, LOWER(pref_name) AS name FROM molecule_dictionary WHERE pref_name IS NOT NULL
          UNION SELECT molregno, LOWER(synonyms) FROM molecule_synonyms WHERE synonyms IS NOT NULL
          UNION SELECT molregno, LOWER(chembl_id) FROM molecule_dictionary)
        SELECT DISTINCT n.name, p.parent_molregno, LOWER(pm.pref_name) AS pref_name, pm.molecule_type
        FROM names n JOIN parent p ON n.molregno = p.molregno
        JOIN molecule_dictionary pm ON pm.molregno = p.parent_molregno
    """
    mech_sql = parent_cte + """
        SELECT DISTINCT p.parent_molregno, dm.mechanism_of_action, dm.action_type,
               td.target_type, td.organism, cs.component_synonym AS gene
        FROM drug_mechanism dm
        JOIN parent p ON dm.molregno = p.molregno
        LEFT JOIN target_dictionary td ON dm.tid = td.tid
        LEFT JOIN target_components tc ON td.tid = tc.tid
        LEFT JOIN component_synonyms cs ON tc.component_id = cs.component_id AND cs.syn_type = 'GENE_SYMBOL'
    """
    with sqlite3.connect(db) as con:
        names = pd.read_sql(names_sql, con)
        mech = pd.read_sql(mech_sql, con)
    mech = mech[(mech["organism"] == "Homo sapiens") | mech["target_type"].fillna("").str.contains("NUCLEIC")]
    mech.loc[mech["target_type"].fillna("").str.contains("NUCLEIC"), "gene"] = "DNA"
    mech = mech.dropna(subset=["gene"])
    names.to_parquet(cache_n)
    mech.to_parquet(cache_m)
    return names, mech


def load_chembl_activity(genes: list[str], version: str = RELEASES["chembl"]) -> pd.DataFrame | None:
    """Potency records with the publication year of the source document (for time-bounding)."""
    key = hashlib.sha1(",".join(sorted(genes)).encode()).hexdigest()[:12]
    cache = RAW_DIR / f"chembl_{version}_activity_{key}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    db = chembl_db(version)
    if db is None:
        return None
    placeholders = ",".join("?" * len(genes))
    sql = f"""
        SELECT cs.component_synonym AS gene, act.molregno, act.pchembl_value AS pchembl, d.year
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_synonyms cs ON tc.component_id = cs.component_id
        JOIN docs d ON act.doc_id = d.doc_id
        WHERE td.target_type = 'SINGLE PROTEIN' AND td.organism = 'Homo sapiens'
          AND cs.syn_type = 'GENE_SYMBOL' AND act.pchembl_value IS NOT NULL
          AND act.standard_relation = '=' AND a.confidence_score >= 8
          AND d.year IS NOT NULL AND cs.component_synonym IN ({placeholders})
    """
    with sqlite3.connect(db) as con:
        df = pd.read_sql(sql, con, params=list(genes))
    df.to_parquet(cache)
    return df


# ── Optional licence-restricted / supplementary files ────────────────────────
def load_cgc() -> set[str] | None:
    if not COSMIC_CGC_FILE.exists():
        log.info("COSMIC CGC file not provided; cgc_member will be NaN (sensitivity feature only)")
        return None
    df = pd.read_csv(COSMIC_CGC_FILE)
    col = next((c for c in df.columns if c.lower() in ("gene symbol", "gene_symbol")), df.columns[0])
    record_manifest("cosmic_cgc", "user-supplied", COSMIC_CGC_FILE, "user-supplied")
    return set(df[col].astype(str))


def load_thorsson() -> pd.Series | None:
    """Mean leukocyte fraction per TCGA study code (e.g. 'LUAD')."""
    if not THORSSON_FILE.exists():
        log.info("Thorsson immune-landscape table not provided; leukocyte_fraction will be NaN")
        return None
    df = pd.read_csv(THORSSON_FILE, sep="\t")
    study = next((c for c in df.columns if "study" in c.lower()), None)
    lf = next((c for c in df.columns if "leukocyte" in c.lower()), None)
    if study is None or lf is None:
        log.error("Thorsson table missing study/leukocyte columns")
        return None
    record_manifest("thorsson_2018", "user-supplied", THORSSON_FILE, "Thorsson et al. 2018")
    return pd.to_numeric(df[lf], errors="coerce").groupby(df[study].str.upper()).mean()


def load_all(genes: list[str], skip: set[str] | None = None) -> dict:
    """Load every source; a failure in one source never stops the others."""
    skip = skip or set()
    loaders = {
        "depmap": load_depmap, "hpa": load_hpa, "string": load_string,
        "reactome": load_reactome, "gwas": load_gwas, "cgc": load_cgc, "thorsson": load_thorsson,
        "chembl_activity": lambda: load_chembl_activity(genes),
    }
    out = {}
    for name, fn in loaders.items():
        if name in skip:
            out[name] = None
            continue
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001
            log.error(f"Source {name} failed: {exc}")
            out[name] = None
    status = {k: ("ok" if v is not None else "MISSING") for k, v in out.items()}
    log.info(f"Data sources: {status}")
    return out
