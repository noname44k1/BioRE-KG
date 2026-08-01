import networkx as nx
from typing import List, Dict, Any
from kg.kg_retriever import KGRetriever


class MultiHopReasoner:
    """
    Module corresponding to the 'Multi-hop Reasoning' branch in the pipeline.
    Finds deep connections (2-hop, 3-hop) between entities in the KG and
    identifies latent relationships or hub nodes to enrich the RAG evidence.

    Works with any KG loaded by a KGRetriever instance (per-dataset or merged).
    """

    def __init__(self, retriever: KGRetriever):
        self.retriever = retriever
        self.G = self.retriever.G      # Reuse the already-loaded graph

    # ------------------------------------------------------------------
    # Path Finding
    # ------------------------------------------------------------------
    def find_paths(self,
                   source_entity: str,
                   target_entity: str,
                   max_depth: int = 2) -> List[List[Dict[str, Any]]]:
        """
        Find all simple directed paths between two entities up to max_depth hops.

        Args:
            source_entity : Start node (case-insensitive).
            target_entity : End node (case-insensitive).
            max_depth     : Maximum number of hops / edges in the path.

        Returns:
            List of paths. Each path is a list of edge dicts:
            [{subject, predicate, object, context, source}, ...]

        Note:
            nx.all_simple_paths can be expensive on very dense subgraphs.
            For large KGs, consider limiting with a pre-filter or BFS cutoff.
        """
        source_entity = source_entity.strip().lower()
        target_entity = target_entity.strip().lower()

        if not (self.G.has_node(source_entity) and self.G.has_node(target_entity)):
            return []

        all_paths = []
        try:
            raw_paths = nx.all_simple_paths(
                self.G,
                source=source_entity,
                target=target_entity,
                cutoff=max_depth
            )

            for path_nodes in raw_paths:
                path_edges = []
                for i in range(len(path_nodes) - 1):
                    subj = path_nodes[i]
                    obj  = path_nodes[i + 1]

                    # MultiDiGraph can have multiple edges; take the first one.
                    # TODO: could branch into separate paths per multi-edge.
                    edge_data_dict = self.G.get_edge_data(subj, obj)
                    edge_id        = list(edge_data_dict.keys())[0]
                    props          = edge_data_dict[edge_id]

                    path_edges.append({
                        "subject":   subj,
                        "predicate": props.get("predicate"),
                        "object":    obj,
                        "context":   props.get("context"),
                        "source":    props.get("source"),
                    })
                all_paths.append(path_edges)

        except nx.NetworkXNoPath:
            pass

        return all_paths

    # ------------------------------------------------------------------
    # High-level Reasoning Entry Point
    # ------------------------------------------------------------------
    def reason_over_evidence(self,
                             query: str,
                             context_entities: List[str] = None,
                             max_depth: int = 2) -> List[List[Dict[str, Any]]]:
        """
        Extends direct KG evidence by performing multi-hop reasoning.

        Args:
            query            : Input query string.
            context_entities : Entity list from the RAG/retrieval branch
                               (used as target nodes for path finding).
            max_depth        : Maximum hop depth for path search.

        Returns:
            List of paths (each path is a list of edge dicts).

        TODO: Add path scoring, graph attention, or LLM-based path filtering.
        """
        print(f"[MultiHopReasoner] dataset='{self.retriever.dataset_name}' | max_depth={max_depth}")
        query_entities  = self.retriever.extract_entities_from_query(query)
        reasoned_paths  = []

        if context_entities and query_entities:
            for q_ent in query_entities:
                for c_ent in context_entities:
                    if q_ent != c_ent:
                        paths = self.find_paths(q_ent, c_ent, max_depth=max_depth)
                        reasoned_paths.extend(paths)

        print(f"[MultiHopReasoner] Paths found: {len(reasoned_paths)}")
        return reasoned_paths


# =========================================================
# Quick smoke test
# =========================================================
if __name__ == "__main__":
    from kg.kg_retriever import KGRetriever
    # retriever = KGRetriever(dataset_name="DDI")
    # reasoner  = MultiHopReasoner(retriever)
    # paths = reasoner.find_paths("aspirin", "headache", max_depth=2)
    pass
