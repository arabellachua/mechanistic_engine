"""
drug_target_db.py — Small, manually curated SEED lexicon (mechanism facts only).

Role in v9: fallback for drugs the ChEMBL drug_mechanism table does not cover, and a
modality override for ADCs. ChEMBL is the primary, versioned source (drug_mapping.py);
disagreements between the two are logged.

Deleted from v6: TARGET_PROPERTIES, STRONG_PAIRS, MEDIUM_PAIRS. Those tables encoded
which target–disease pairs are known to work, i.e. the quantity being predicted.
Also fixed: "VEGFR2" is not an HGNC symbol (it is KDR).
"""
# name -> (targets, modality, moa)
SEED_DRUGS: dict[str, tuple[tuple[str, ...], str, str]] = {
    # Checkpoint inhibitors
    "pembrolizumab": (("PDCD1",), "antibody", "checkpoint_inhibitor"),
    "nivolumab": (("PDCD1",), "antibody", "checkpoint_inhibitor"),
    "cemiplimab": (("PDCD1",), "antibody", "checkpoint_inhibitor"),
    "dostarlimab": (("PDCD1",), "antibody", "checkpoint_inhibitor"),
    "atezolizumab": (("CD274",), "antibody", "checkpoint_inhibitor"),
    "durvalumab": (("CD274",), "antibody", "checkpoint_inhibitor"),
    "avelumab": (("CD274",), "antibody", "checkpoint_inhibitor"),
    "ipilimumab": (("CTLA4",), "antibody", "checkpoint_inhibitor"),
    "tremelimumab": (("CTLA4",), "antibody", "checkpoint_inhibitor"),
    "relatlimab": (("LAG3",), "antibody", "checkpoint_inhibitor"),
    # PARP
    "olaparib": (("PARP1", "PARP2"), "small_molecule", "parp_inhibitor"),
    "rucaparib": (("PARP1", "PARP2"), "small_molecule", "parp_inhibitor"),
    "niraparib": (("PARP1", "PARP2"), "small_molecule", "parp_inhibitor"),
    "talazoparib": (("PARP1", "PARP2"), "small_molecule", "parp_inhibitor"),
    "veliparib": (("PARP1", "PARP2"), "small_molecule", "parp_inhibitor"),
    # KRAS / EGFR / HER2
    "sotorasib": (("KRAS",), "small_molecule", "kras_inhibitor"),
    "adagrasib": (("KRAS",), "small_molecule", "kras_inhibitor"),
    "osimertinib": (("EGFR",), "small_molecule", "egfr_inhibitor"),
    "erlotinib": (("EGFR",), "small_molecule", "egfr_inhibitor"),
    "gefitinib": (("EGFR",), "small_molecule", "egfr_inhibitor"),
    "afatinib": (("EGFR", "ERBB2"), "small_molecule", "egfr_inhibitor"),
    "dacomitinib": (("EGFR",), "small_molecule", "egfr_inhibitor"),
    "cetuximab": (("EGFR",), "antibody", "egfr_antibody"),
    "panitumumab": (("EGFR",), "antibody", "egfr_antibody"),
    "amivantamab": (("EGFR", "MET"), "antibody", "bispecific_antibody"),
    "trastuzumab": (("ERBB2",), "antibody", "her2_antibody"),
    "pertuzumab": (("ERBB2",), "antibody", "her2_antibody"),
    "trastuzumab deruxtecan": (("ERBB2",), "adc", "her2_adc"),
    "ado-trastuzumab emtansine": (("ERBB2",), "adc", "her2_adc"),
    "trastuzumab emtansine": (("ERBB2",), "adc", "her2_adc"),
    "lapatinib": (("ERBB2", "EGFR"), "small_molecule", "her2_inhibitor"),
    "neratinib": (("ERBB2", "EGFR"), "small_molecule", "her2_inhibitor"),
    "tucatinib": (("ERBB2",), "small_molecule", "her2_inhibitor"),
    # BRAF / MEK
    "vemurafenib": (("BRAF",), "small_molecule", "braf_inhibitor"),
    "dabrafenib": (("BRAF",), "small_molecule", "braf_inhibitor"),
    "encorafenib": (("BRAF",), "small_molecule", "braf_inhibitor"),
    "trametinib": (("MAP2K1", "MAP2K2"), "small_molecule", "mek_inhibitor"),
    "cobimetinib": (("MAP2K1",), "small_molecule", "mek_inhibitor"),
    "binimetinib": (("MAP2K1", "MAP2K2"), "small_molecule", "mek_inhibitor"),
    "selumetinib": (("MAP2K1", "MAP2K2"), "small_molecule", "mek_inhibitor"),
    # ALK / ROS1 / NTRK / RET / MET
    "crizotinib": (("ALK", "MET", "ROS1"), "small_molecule", "alk_inhibitor"),
    "alectinib": (("ALK",), "small_molecule", "alk_inhibitor"),
    "lorlatinib": (("ALK", "ROS1"), "small_molecule", "alk_inhibitor"),
    "ceritinib": (("ALK",), "small_molecule", "alk_inhibitor"),
    "brigatinib": (("ALK",), "small_molecule", "alk_inhibitor"),
    "larotrectinib": (("NTRK1", "NTRK2", "NTRK3"), "small_molecule", "trk_inhibitor"),
    "entrectinib": (("NTRK1", "NTRK2", "NTRK3", "ROS1", "ALK"), "small_molecule", "trk_inhibitor"),
    "selpercatinib": (("RET",), "small_molecule", "ret_inhibitor"),
    "pralsetinib": (("RET",), "small_molecule", "ret_inhibitor"),
    "capmatinib": (("MET",), "small_molecule", "met_inhibitor"),
    "tepotinib": (("MET",), "small_molecule", "met_inhibitor"),
    # Cell cycle / apoptosis / BTK / PI3K-mTOR / JAK
    "palbociclib": (("CDK4", "CDK6"), "small_molecule", "cdk_inhibitor"),
    "ribociclib": (("CDK4", "CDK6"), "small_molecule", "cdk_inhibitor"),
    "abemaciclib": (("CDK4", "CDK6"), "small_molecule", "cdk_inhibitor"),
    "venetoclax": (("BCL2",), "small_molecule", "bcl2_inhibitor"),
    "ibrutinib": (("BTK",), "small_molecule", "btk_inhibitor"),
    "acalabrutinib": (("BTK",), "small_molecule", "btk_inhibitor"),
    "zanubrutinib": (("BTK",), "small_molecule", "btk_inhibitor"),
    "everolimus": (("MTOR",), "small_molecule", "mtor_inhibitor"),
    "temsirolimus": (("MTOR",), "small_molecule", "mtor_inhibitor"),
    "alpelisib": (("PIK3CA",), "small_molecule", "pi3k_inhibitor"),
    "idelalisib": (("PIK3CD",), "small_molecule", "pi3k_inhibitor"),
    "capivasertib": (("AKT1", "AKT2", "AKT3"), "small_molecule", "akt_inhibitor"),
    "ruxolitinib": (("JAK1", "JAK2"), "small_molecule", "jak_inhibitor"),
    # Angiogenesis / multikinase
    "bevacizumab": (("VEGFA",), "antibody", "vegf_antibody"),
    "ramucirumab": (("KDR",), "antibody", "vegfr_antibody"),
    "aflibercept": (("VEGFA", "PGF"), "protein", "vegf_trap"),
    "sorafenib": (("BRAF", "RAF1", "KDR"), "small_molecule", "multikinase_inhibitor"),
    "sunitinib": (("KDR", "PDGFRB", "KIT"), "small_molecule", "multikinase_inhibitor"),
    "lenvatinib": (("KDR", "FGFR1", "RET"), "small_molecule", "multikinase_inhibitor"),
    "cabozantinib": (("MET", "KDR"), "small_molecule", "multikinase_inhibitor"),
    "regorafenib": (("KDR", "KIT", "RET"), "small_molecule", "multikinase_inhibitor"),
    "axitinib": (("KDR", "FLT1", "FLT4"), "small_molecule", "vegfr_inhibitor"),
    "pazopanib": (("KDR", "PDGFRA", "KIT"), "small_molecule", "multikinase_inhibitor"),
    "imatinib": (("ABL1", "KIT", "PDGFRA"), "small_molecule", "multikinase_inhibitor"),
    "dasatinib": (("ABL1", "SRC"), "small_molecule", "multikinase_inhibitor"),
    "nilotinib": (("ABL1",), "small_molecule", "abl_inhibitor"),
    "ponatinib": (("ABL1",), "small_molecule", "abl_inhibitor"),
    "bosutinib": (("ABL1", "SRC"), "small_molecule", "abl_inhibitor"),
    # Lymphoid antigens / ADCs
    "rituximab": (("MS4A1",), "antibody", "cd20_antibody"),
    "obinutuzumab": (("MS4A1",), "antibody", "cd20_antibody"),
    "daratumumab": (("CD38",), "antibody", "cd38_antibody"),
    "sacituzumab govitecan": (("TACSTD2",), "adc", "trop2_adc"),
    "enfortumab vedotin": (("NECTIN4",), "adc", "nectin4_adc"),
    "brentuximab vedotin": (("TNFRSF8",), "adc", "cd30_adc"),
    "polatuzumab vedotin": (("CD79B",), "adc", "cd79b_adc"),
    "tisotumab vedotin": (("F3",), "adc", "tissue_factor_adc"),
    "mirvetuximab soravtansine": (("FOLR1",), "adc", "folr1_adc"),
    # Hormonal
    "enzalutamide": (("AR",), "small_molecule", "ar_antagonist"),
    "apalutamide": (("AR",), "small_molecule", "ar_antagonist"),
    "darolutamide": (("AR",), "small_molecule", "ar_antagonist"),
    "abiraterone": (("CYP17A1",), "small_molecule", "cyp17_inhibitor"),
    "tamoxifen": (("ESR1",), "small_molecule", "serm"),
    "fulvestrant": (("ESR1",), "small_molecule", "serd"),
    "elacestrant": (("ESR1",), "small_molecule", "serd"),
    "letrozole": (("CYP19A1",), "small_molecule", "aromatase_inhibitor"),
    "anastrozole": (("CYP19A1",), "small_molecule", "aromatase_inhibitor"),
    "exemestane": (("CYP19A1",), "small_molecule", "aromatase_inhibitor"),
    # Other targeted
    "bortezomib": (("PSMB5",), "small_molecule", "proteasome_inhibitor"),
    "carfilzomib": (("PSMB5",), "small_molecule", "proteasome_inhibitor"),
    "ixazomib": (("PSMB5",), "small_molecule", "proteasome_inhibitor"),
    "lenalidomide": (("CRBN",), "small_molecule", "imid"),
    "pomalidomide": (("CRBN",), "small_molecule", "imid"),
    "thalidomide": (("CRBN",), "small_molecule", "imid"),
    "vismodegib": (("SMO",), "small_molecule", "hedgehog_inhibitor"),
    "sonidegib": (("SMO",), "small_molecule", "hedgehog_inhibitor"),
    "ivosidenib": (("IDH1",), "small_molecule", "idh_inhibitor"),
    "enasidenib": (("IDH2",), "small_molecule", "idh_inhibitor"),
    "gilteritinib": (("FLT3",), "small_molecule", "flt3_inhibitor"),
    "midostaurin": (("FLT3", "KIT"), "small_molecule", "flt3_inhibitor"),
    "erdafitinib": (("FGFR1", "FGFR2", "FGFR3", "FGFR4"), "small_molecule", "fgfr_inhibitor"),
    "pemigatinib": (("FGFR1", "FGFR2", "FGFR3"), "small_molecule", "fgfr_inhibitor"),
    # Cytotoxics (pseudo-target "DNA" has no gene-level features)
    "capecitabine": (("TYMS",), "small_molecule", "antimetabolite"),
    "fluorouracil": (("TYMS",), "small_molecule", "antimetabolite"),
    "pemetrexed": (("TYMS", "DHFR", "GART"), "small_molecule", "antimetabolite"),
    "methotrexate": (("DHFR",), "small_molecule", "antimetabolite"),
    "gemcitabine": (("RRM1",), "small_molecule", "antimetabolite"),
    "cytarabine": (("POLA1",), "small_molecule", "antimetabolite"),
    "cisplatin": (("DNA",), "small_molecule", "dna_crosslinker"),
    "carboplatin": (("DNA",), "small_molecule", "dna_crosslinker"),
    "oxaliplatin": (("DNA",), "small_molecule", "dna_crosslinker"),
    "temozolomide": (("DNA",), "small_molecule", "alkylating_agent"),
    "cyclophosphamide": (("DNA",), "small_molecule", "alkylating_agent"),
    "ifosfamide": (("DNA",), "small_molecule", "alkylating_agent"),
    "bendamustine": (("DNA",), "small_molecule", "alkylating_agent"),
    "paclitaxel": (("TUBB",), "small_molecule", "tubulin_inhibitor"),
    "nab-paclitaxel": (("TUBB",), "small_molecule", "tubulin_inhibitor"),
    "docetaxel": (("TUBB",), "small_molecule", "tubulin_inhibitor"),
    "eribulin": (("TUBB",), "small_molecule", "tubulin_inhibitor"),
    "vinorelbine": (("TUBB",), "small_molecule", "tubulin_inhibitor"),
    "vincristine": (("TUBB",), "small_molecule", "tubulin_inhibitor"),
    "doxorubicin": (("TOP2A",), "small_molecule", "topoisomerase_inhibitor"),
    "epirubicin": (("TOP2A",), "small_molecule", "topoisomerase_inhibitor"),
    "etoposide": (("TOP2A",), "small_molecule", "topoisomerase_inhibitor"),
    "irinotecan": (("TOP1",), "small_molecule", "topoisomerase_inhibitor"),
    "topotecan": (("TOP1",), "small_molecule", "topoisomerase_inhibitor"),
}

BRAND_ALIASES = {
    "keytruda": "pembrolizumab", "opdivo": "nivolumab", "tecentriq": "atezolizumab",
    "imfinzi": "durvalumab", "bavencio": "avelumab", "yervoy": "ipilimumab",
    "lynparza": "olaparib", "zejula": "niraparib", "tagrisso": "osimertinib",
    "tarceva": "erlotinib", "iressa": "gefitinib", "herceptin": "trastuzumab",
    "perjeta": "pertuzumab", "enhertu": "trastuzumab deruxtecan", "t-dxd": "trastuzumab deruxtecan",
    "kadcyla": "ado-trastuzumab emtansine", "t-dm1": "ado-trastuzumab emtansine",
    "avastin": "bevacizumab", "erbitux": "cetuximab", "vectibix": "panitumumab",
    "ibrance": "palbociclib", "kisqali": "ribociclib", "verzenio": "abemaciclib",
    "imbruvica": "ibrutinib", "venclexta": "venetoclax", "revlimid": "lenalidomide",
    "velcade": "bortezomib", "xtandi": "enzalutamide", "zytiga": "abiraterone",
    "rituxan": "rituximab", "mabthera": "rituximab", "abraxane": "nab-paclitaxel",
    "5-fu": "fluorouracil", "5-fluorouracil": "fluorouracil", "xeloda": "capecitabine",
    "gemzar": "gemcitabine", "taxol": "paclitaxel", "taxotere": "docetaxel", "alimta": "pemetrexed",
}

# Standard-of-care partners: de-prioritised when choosing the investigational agent.
BACKBONE_AGENTS = {
    "cisplatin", "carboplatin", "oxaliplatin", "paclitaxel", "nab-paclitaxel", "docetaxel",
    "gemcitabine", "fluorouracil", "capecitabine", "irinotecan", "topotecan", "etoposide",
    "doxorubicin", "epirubicin", "cyclophosphamide", "ifosfamide", "pemetrexed", "vinorelbine",
    "vincristine", "cytarabine", "temozolomide", "methotrexate", "bendamustine",
    "leucovorin", "levoleucovorin", "folinic acid", "dexamethasone", "prednisone", "prednisolone",
    "rituximab", "trastuzumab", "bevacizumab",
}
NON_DRUG_TERMS = ("placebo", "best supportive care", "standard of care", "observation",
                  "saline", "vehicle", "physician's choice", "investigator's choice")
CHECKPOINT_TARGETS = {"PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT"}
HORMONAL_TARGETS = {"ESR1", "ESR2", "AR", "CYP17A1", "CYP19A1", "GNRHR", "PGR", "SRD5A1", "SRD5A2"}
CYTOTOXIC_MOAS = {"antimetabolite", "dna_crosslinker", "alkylating_agent",
                  "tubulin_inhibitor", "topoisomerase_inhibitor"}
CYTOTOXIC_MECH_RE = (r"dna (inhibitor|cross|alkylat|intercal)|tubulin|topoisomerase|thymidylate synthase|"
                     r"ribonucleotide reductase|dna polymerase|dihydrofolate reductase")
ADC_NAME_PARTS = ("vedotin", "deruxtecan", "govitecan", "emtansine", "mafodotin", "ozogamicin",
                  "soravtansine", "tesirine", "duocarmazine", "mertansine", "ravtansine")
