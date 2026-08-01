import os
import json
import pickle
import networkx as nx
from kg.config import DATASET_PATHS, KG_GRAPH_FILES, KG_JSON_FILES, KG_MERGED_GRAPH_FILE, KG_MERGED_JSON_FILE, DATASET_ALLOWED_RELATIONS


# =========================================================
# Step 1: Load triplets from JSONL files
# =========================================================
def load_triplets(dataset_dir: str) -> list:
    """
    Load triplet data from all JSONL files in a dataset directory.
    Returns a list of dicts:
        [{'SUBJECT_TEXT': ..., 'PREDICATE': ..., 'OBJECT_TEXT': ..., 'SENTENCE': ...}]
    """
    triplets = []
    if not os.path.exists(dataset_dir):
        print(f"  [WARN] Directory not found: {dataset_dir}")
        return triplets

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jsonl"): #and "_test_" not in filename:
            filepath = os.path.join(dataset_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        triplets.append(data)
                    except Exception as e:
                        print(f"  [ERROR] Parsing line in {filename}: {e}")
    return triplets


# =========================================================
# Step 2: Build a single KG for ONE dataset
# =========================================================
def build_graph_for_dataset(ds_name: str, ds_path: str) -> nx.MultiDiGraph:
    """
    Builds a NetworkX MultiDiGraph from the triplets of a single dataset.
    Only includes triples whose PREDICATE is in DATASET_ALLOWED_RELATIONS[ds_name].

    MultiDiGraph is chosen because:
      - Directed edges  : relation direction is semantically meaningful
      - Multi-edges     : two entities can share multiple distinct predicates
    """
    G = nx.MultiDiGraph()
    G.graph["dataset"] = ds_name

    # Whitelist filter — lấy set cho dataset này (nếu không có thì không lọc)
    allowed = DATASET_ALLOWED_RELATIONS.get(ds_name, None)
    if allowed:
        allowed_upper = {r.upper() for r in allowed}   # normalize to uppercase
        print(f"[{ds_name}] Relation filter: {sorted(allowed_upper)}")
    else:
        allowed_upper = None
        print(f"[{ds_name}] No relation filter — including ALL predicates.")

    print(f"\n[{ds_name}] Loading triplets from: {ds_path}")
    triplets = load_triplets(ds_path)
    print(f"[{ds_name}] Raw triplets read: {len(triplets)}")

    skipped_missing = 0
    skipped_filtered = 0
    for t in triplets:
        subj = t.get("SUBJECT_TEXT", "").strip().lower()
        obj  = t.get("OBJECT_TEXT",  "").strip().lower()
        pred = t.get("PREDICATE",    "").strip()
        sent = t.get("SENTENCE",     "")

        if not (subj and obj and pred):
            skipped_missing += 1
            continue

        # Relation whitelist filter
        if allowed_upper and pred.upper() not in allowed_upper:
            skipped_filtered += 1
            continue

        # Add nodes
        if not G.has_node(subj):
            G.add_node(subj, type="entity")
        if not G.has_node(obj):
            G.add_node(obj, type="entity")

        # Add directed edge with rich metadata
        G.add_edge(subj, obj,
                   predicate=pred,
                   context=sent,
                   source=ds_name)

    print(f"[{ds_name}] Skipped (missing fields): {skipped_missing}")
    print(f"[{ds_name}] Skipped (filtered by relation whitelist): {skipped_filtered}")
    print(f"[{ds_name}] Graph => Nodes: {G.number_of_nodes():,} | "
          f"Edges: {G.number_of_edges():,}")
    return G


# =========================================================
# Step 3: Save a single KG to disk
# =========================================================
def save_graph(G: nx.MultiDiGraph, ds_name: str, graph_file: str, json_file: str):
    """
    Saves a graph to:
      - .pkl  : fast binary format for KGRetriever / MultiHopReasoner
      - .json : human-readable node-link JSON for manual inspection in VS Code
    """
    print(f"  -> Saving pickle : {graph_file}")
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)

    print(f"  -> Saving JSON   : {json_file}")
    json_data = nx.node_link_data(G)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


# =========================================================
# Step 4: Build merged KG from a dict of pre-built graphs
# =========================================================
def build_merged_graph(graphs: dict) -> nx.MultiDiGraph:
    """
    Merges all per-dataset graphs into one unified MultiDiGraph using nx.compose_all().
    Shared entity nodes will be merged; multi-source edges are preserved with
    their individual 'source' attribute intact.
    """
    print("\n[MERGED] Composing all dataset graphs into merged KG...")
    merged = nx.compose_all(graphs.values())
    merged.graph["dataset"] = "MERGED"
    print(f"[MERGED] Graph => Nodes: {merged.number_of_nodes():,} | "
          f"Edges: {merged.number_of_edges():,}")
    return merged


# =========================================================
# Main pipeline
# =========================================================
if __name__ == "__main__":
    all_graphs = {}

    # --- Build per-dataset KGs ---
    for ds_name, ds_path in DATASET_PATHS.items():
        G = build_graph_for_dataset(ds_name, ds_path)
        save_graph(G, ds_name,
                   graph_file=KG_GRAPH_FILES[ds_name],
                   json_file=KG_JSON_FILES[ds_name])
        all_graphs[ds_name] = G

    # --- Build merged KG ---
    G_merged = build_merged_graph(all_graphs)
    save_graph(G_merged, "MERGED",
               graph_file=KG_MERGED_GRAPH_FILE,
               json_file=KG_MERGED_JSON_FILE)

    # --- Summary ---
    print("\n" + "="*60)
    print("  Knowledge Graph Construction — COMPLETE")
    print("="*60)
    print(f"  {'Dataset':<12} {'Nodes':>8} {'Edges':>10}")
    print("-"*60)
    for ds_name, G in all_graphs.items():
        print(f"  {ds_name:<12} {G.number_of_nodes():>8,} {G.number_of_edges():>10,}")
    print("-"*60)
    print(f"  {'MERGED':<12} {G_merged.number_of_nodes():>8,} {G_merged.number_of_edges():>10,}")
    print("="*60)
    print(f"\n  Output directory: {__import__('kg.config', fromlist=['KG_OUTPUT_DIR']).KG_OUTPUT_DIR}")
