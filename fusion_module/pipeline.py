"""
pipeline.py
───────────
BioHybridPipeline — kết nối toàn bộ:
  KG Branch (KGRetriever — 1-hop direct only)
  + Fusion Module (Weighted Score Fusion)

Multi-hop reasoning đã được loại bỏ để giảm nhiễu.

Chạy: python -m fusion_module.pipeline
"""
from __future__ import annotations

import logging
from typing import List, Any

from kg.kg_retriever import KGRetriever
from fusion_module.fusion import FusionModule
from fusion_module.config import (
    SOURCE_WEIGHT, SIM_WEIGHT, KG_SOURCE_WEIGHT, RAG_SOURCE_WEIGHT,
    TOP_K_AFTER_RERANK,
)

logger = logging.getLogger(__name__)


class BioHybridPipeline:
    """End-to-end pipeline: KG 1-hop Retrieval + Weighted Fusion."""

    def __init__(
        self,
        kg_dataset:    str   = "MERGED",
        top_k:         int   = TOP_K_AFTER_RERANK,
        source_weight: float = SOURCE_WEIGHT,
        sim_weight:    float = SIM_WEIGHT,
        kg_src_score:  float = KG_SOURCE_WEIGHT,
        rag_src_score: float = RAG_SOURCE_WEIGHT,
    ):
        """
        Args:
            kg_dataset    : KG to load — "GM_CIHT" | "DDI" | "CHEMPROT" | "MERGED"
            top_k         : Number of evidence items to keep after scoring.
            source_weight : Weight for source-type component (α). Default 0.4.
            sim_weight    : Weight for cosine similarity component (β). Default 0.6.
            kg_src_score  : Source score for KG direct evidence. Default 1.0.
            rag_src_score : Source score for RAG chunks. Default 0.75.
        """
        logger.info(
            f"[BioHybridPipeline] Init | KG={kg_dataset} | top_k={top_k} | "
            f"α={source_weight} | β={sim_weight}"
        )

        # KG Branch — 1-hop only
        self.retriever = KGRetriever(dataset_name=kg_dataset)

        # Weighted Fusion Module
        self.fusion = FusionModule(
            top_k         = top_k,
            source_weight = source_weight,
            sim_weight    = sim_weight,
            kg_src_score  = kg_src_score,
            rag_src_score = rag_src_score,
        )

    # ──────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────
    def run(
        self,
        query:      str,
        rag_chunks: List[Any] = None,
    ) -> str:
        """
        Chạy toàn bộ pipeline cho một câu query.

        Args:
            query      : Câu văn / query y sinh học.
            rag_chunks : (Tuỳ chọn) Evidence từ RAG Branch.
                         Format: [{"chunk": str, "score": float, "relation": str}, ...]

        Returns:
            Fused context string đã sẵn sàng để gắn vào LLM triple extraction.
        """
        rag_chunks = rag_chunks or []

        logger.info("[BioHybridPipeline] Step 1: Direct KG evidence (1-hop)")
        kg_direct = self.retriever.retrieve_direct_evidence(query)
        logger.info(f"  → {len(kg_direct)} direct evidence items")

        logger.info("[BioHybridPipeline] Step 2: Weighted Fusion")
        return self.fusion.fuse(
            query      = query,
            kg_direct  = kg_direct,
            rag_chunks = rag_chunks,
        )

    def get_fused_context(
        self,
        query:      str,
        rag_chunks: List[Any] = None,
    ) -> str:
        """Alias cho run(). Dùng để debug."""
        return self.run(query, rag_chunks)


# ══════════════════════════════════════════════════════════
# Smoke test — không cần GPU hay Ollama
# Chạy: python -m fusion_module.pipeline
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pipeline = BioHybridPipeline(kg_dataset="MERGED")

    query     = "aspirin inhibits cox-2 enzyme activity"
    rag_input = [
        {"chunk": "aspirin reduces platelet aggregation",     "score": 0.91, "relation": "INHIBITS"},
        {"chunk": "cox-2 causes inflammation in the joints",  "score": 0.85, "relation": "CAUSES"},
        {"chunk": "ibuprofen blocks prostaglandin synthesis",  "score": 0.78, "relation": "INHIBITS"},
    ]

    SEP = "=" * 65
    print(f"\n{SEP}")
    print(" SMOKE TEST — BioHybridPipeline (Weighted Fusion)")
    print(SEP)

    context = pipeline.get_fused_context(query, rag_chunks=rag_input)
    print("\n--- Fused Context ---")
    print(context)
    print(f"\n{SEP}")
    print(" SMOKE TEST PASSED")
    print(SEP)
