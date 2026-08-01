import os

# =========================================================
# Base Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# =========================================================
# Dataset Paths (input triplet JSONL directories)
# =========================================================
DATASET_PATHS = {
    "GM_CIHT":  os.path.join(DATASET_DIR, "0_GM-CIHT"),
    "DDI":      os.path.join(DATASET_DIR, "1_DDI"),
    "CHEMPROT": os.path.join(DATASET_DIR, "2_chemprot"),
}

# =========================================================
# KG Output Directory
# =========================================================
KG_OUTPUT_DIR = os.path.join(BASE_DIR, "kg", "outputs")
os.makedirs(KG_OUTPUT_DIR, exist_ok=True)

# =========================================================
# Per-Dataset KG Output Paths
# Each dataset gets its own .pkl (fast binary) + .json (human-readable)
# =========================================================
KG_GRAPH_FILES = {
    ds_name: os.path.join(KG_OUTPUT_DIR, f"kg_{ds_name}.pkl")
    for ds_name in DATASET_PATHS
}

KG_JSON_FILES = {
    ds_name: os.path.join(KG_OUTPUT_DIR, f"kg_{ds_name}.json")
    for ds_name in DATASET_PATHS
}

# =========================================================
# Merged KG Output Paths (union of all datasets)
# =========================================================
KG_MERGED_GRAPH_FILE = os.path.join(KG_OUTPUT_DIR, "kg_MERGED.pkl")
KG_MERGED_JSON_FILE  = os.path.join(KG_OUTPUT_DIR, "kg_MERGED.json")

# =========================================================
# Legacy single-file paths (kept for backward compat)
# =========================================================
KG_GRAPH_FILE = KG_MERGED_GRAPH_FILE
KG_JSON_FILE  = KG_MERGED_JSON_FILE

# =========================================================
# Allowed Relations per Dataset
# Chỉ các triple có PREDICATE nằm trong whitelist này
# mới được đưa vào KG. Các relation ngoài whitelist bị lọc.
# =========================================================
DATASET_ALLOWED_RELATIONS = {
    # CHEMPROT: Chemical-Protein Interactions
    # 5 quan hệ chính của task NER/RE hóa học-protein
    "CHEMPROT": {
        "ACTIVATOR",       # Hóa chất kích hoạt protein
        "DOWNREGULATOR",   # Hóa chất ức chế biểu hiện protein
        "ANTAGONIST",      # Hóa chất đối kháng protein
        "AGONIST",         # Hóa chất đồng vận protein
        "PRODUCT-OF",      # Protein là sản phẩm của hóa chất

    },
    # DDI: Drug-Drug Interactions
    # Predicates in converted data are lowercase: mechanism, effect, advise, int
    # build_graph normalizes to UPPER for comparison, so whitelist uses UPPER
    "DDI": {
        "MECHANISM",
        "EFFECT",
        "ADVISE",
        "INT",
        "NONE",   # ← dấu phẩy bị thiếu trước đây → "INT""NONE" = "INTNONE" (bug!)
    },
    # GM_CIHT: General biomedical relations (actual predicates in dataset)
    "GM_CIHT": {
        "INTERACTS_WITH",
        "INHIBITS",
        "PRODUCES",
        "STIMULATES",
        "TREATS",
        "AFFECTS",
        "USES",
        "PROCESS_OF",
        "COMPLICATES",
        "PREDISPOSES",
        "DIAGNOSES",
        "PREVENTS",
        "AUGMENTS",
        "MANIFESTATION_OF",
        # --- Previously missing 8 (silently filtered, now added) ---
        "COEXISTS_WITH",       # A co-occurs / is comorbid with B
        "ASSOCIATED_WITH",     # A is associated with / correlates with B
        "ADMINISTERED_TO",     # Drug/procedure A is administered to patient/organism B
        "PRECEDES",            # Event A precedes / leads to event B (temporal)
        "CAUSES",              # A causes / induces B (causal, stronger than AFFECTS)
        "DOES_NOT_TREAT",      # Negative: A does NOT treat B (from neg. RCT results)
        # NOTE: only 20 distinct types were confirmed present in the scanned data;
        # 2 slots reserved for any tail-frequency types found in future data.
    },
}
