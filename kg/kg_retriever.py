import pickle
import networkx as nx
from typing import List, Dict, Any, Optional
from kg.config import KG_GRAPH_FILES, KG_MERGED_GRAPH_FILE


class KGRetriever:
    """
    Module corresponding to the 'Direct KG Evidence' branch in the pipeline.
    Fetches 1-hop (direct) connections from a specified Knowledge Graph.

    Supports:
      - Per-dataset KGs  : pass dataset_name = "GM_CIHT" | "DDI" | "CHEMPROT"
      - Merged KG        : pass dataset_name = "MERGED"  (default)
      - Custom graph path: pass graph_path explicitly to override
    """

    AVAILABLE_DATASETS = list(KG_GRAPH_FILES.keys()) + ["MERGED"]

    def __init__(self,
                 dataset_name: str = "MERGED",
                 graph_path: Optional[str] = None):
        """
        Args:
            dataset_name : Which pre-built KG to load.
                           One of: "GM_CIHT", "DDI", "CHEMPROT", "MERGED".
            graph_path   : Override path. If provided, ignores dataset_name.
        """
        if graph_path:
            self.graph_path = graph_path
        elif dataset_name == "MERGED":
            self.graph_path = KG_MERGED_GRAPH_FILE
        elif dataset_name in KG_GRAPH_FILES:
            self.graph_path = KG_GRAPH_FILES[dataset_name]
        else:
            raise ValueError(
                f"Unknown dataset_name '{dataset_name}'. "
                f"Choose from: {self.AVAILABLE_DATASETS}"
            )

        self.dataset_name = dataset_name
        self.G = self._load_graph()

    def _load_graph(self) -> nx.MultiDiGraph:
        """Load the pre-built Knowledge Graph from pickle."""
        try:
            with open(self.graph_path, 'rb') as f:
                G = pickle.load(f)
            print(f"[KGRetriever] Loaded '{self.dataset_name}' KG — "
                  f"Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")
            return G
        except FileNotFoundError:
            print(f"[KGRetriever][WARN] Graph file not found at: {self.graph_path}")
            print("  Please run:  python -m kg.build_graph  from the project root first.")
            return nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Entity Extraction (placeholder — to be replaced with medical NER)
    # ------------------------------------------------------------------
    # Các từ generic xuất hiện trong câu sinh học nhưng KHÔNG phải entity biomedical
    # (drug/protein/gene). Khớp chúng tạo ra evidence hoàn toàn irrelevant.
    _GENERIC_STOPWORDS = {
        # Sinh vật / tế bào
        "cells", "cell", "rats", "mice", "mouse", "human", "patients", "patient",
        "animals", "subjects", "volunteers", "rabbits", "dogs", "monkeys",
        # Quy trình / điều trị
        "treatment", "administration", "injection", "infusion", "exposure",
        "dose", "doses", "placebo", "control", "controls", "therapy",
        "transplantation", "resection", "surgery",
        # Sinh lý / thực thể quá chung
        "stimulation", "inhibition", "activation", "expression", "production",
        "synthesis", "release", "secretion", "binding", "activity", "effect",
        "effects", "levels", "concentration", "water", "plasma", "serum",
        "blood", "urine", "tissue", "tissues", "culture", "cultures",
        # Trạng thái bệnh quá chung
        "diabetic", "cancer", "tumor", "tumors", "disease", "disorder",
        "agent", "agents", "fatness", "peptides",
    }

    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract potential entities from the input query.

        Filters:
          1. Stop-word list: skip generic biomedical words (cells, rats, treatment...)
          2. Minimum length: entity must be >= 4 chars
          3. Word-boundary: entity must appear as a whole word in the query

        TODO: Replace with a proper Medical NER model (scispaCy / BioBERT-NER)
              for production-quality entity extraction.
        """
        import re
        query_lower = query.lower()
        matched = []
        for node in self.G.nodes():
            # Filter 1: stop-words — block generic biomedical terms
            if node.lower() in self._GENERIC_STOPWORDS:
                continue
            # Filter 2: minimum length — skip very short tokens
            if len(node) < 4:
                continue
            # Filter 3: must appear as whole word (word boundary on both sides)
            pattern = r'(?<![a-z0-9])' + re.escape(node) + r'(?![a-z0-9])'
            if re.search(pattern, query_lower):
                matched.append(node)
        if not matched:
            print("[KGRetriever][WARN] No entities matched in query (naive NER).")
        return matched



    # ------------------------------------------------------------------
    # 1-hop Direct Evidence Retrieval
    # ------------------------------------------------------------------
    def retrieve_direct_evidence(self, query: str) -> List[Dict[str, Any]]:
        """
        Takes an input query, extracts entities, and retrieves all 1-hop
        neighbors (incoming + outgoing) as Direct KG Evidence.

        Returns:
            List of evidence dicts, each containing:
            {subject, predicate, object, context, source, hop, direction}
        """
        entities = self.extract_entities_from_query(query)
        evidence = []
        seen_triples = set()   # De-duplicate: tránh cùng 1 triple xuất hiện 6+ lần

        for entity in entities:
            entity = entity.strip().lower()
            if not self.G.has_node(entity):
                continue

            # Outgoing edges: entity -> neighbor
            for neighbor in self.G.successors(entity):
                for _, props in self.G.get_edge_data(entity, neighbor).items():
                    triple_key = (entity, props.get("predicate"), neighbor)
                    if triple_key in seen_triples:
                        continue
                    seen_triples.add(triple_key)
                    evidence.append({
                        "subject":   entity,
                        "predicate": props.get("predicate"),
                        "object":    neighbor,
                        "context":   props.get("context"),
                        "source":    props.get("source"),
                        "hop":       1,
                        "direction": "outgoing",
                    })

            # Incoming edges: neighbor -> entity
            for neighbor in self.G.predecessors(entity):
                for _, props in self.G.get_edge_data(neighbor, entity).items():
                    triple_key = (neighbor, props.get("predicate"), entity)
                    if triple_key in seen_triples:
                        continue
                    seen_triples.add(triple_key)
                    evidence.append({
                        "subject":   neighbor,
                        "predicate": props.get("predicate"),
                        "object":    entity,
                        "context":   props.get("context"),
                        "source":    props.get("source"),
                        "hop":       1,
                        "direction": "incoming",
                    })

        return evidence


    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, int]:
        return {
            "dataset":    self.dataset_name,
            "num_nodes":  self.G.number_of_nodes(),
            "num_edges":  self.G.number_of_edges(),
        }


# =========================================================
# Quick smoke test
# =========================================================
if __name__ == "__main__":
    for ds in KGRetriever.AVAILABLE_DATASETS:
        r = KGRetriever(dataset_name=ds)
        print(r.get_stats())
