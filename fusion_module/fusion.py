"""
fusion.py
─────────
FusionModule — Weighted Score Fusion (KG + RAG).

Workflow:
  1. Nhận KG evidence (1-hop direct only) và RAG chunks.
  2. Chuẩn hoá về unified format qua evidence_formatter.
  3. Pre-filter (giới hạn số lượng theo MAX_* trong config).
  4. Gán ID tuần tự: E1, E2, ...
  5. Tính weighted score: α * source_score + β * cosine_sim(text, query).
  6. Giữ lại top-K, build fused context string.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

from fusion_module.config import (
    MAX_KG_EVIDENCE_INPUT, MAX_RAG_EVIDENCE_INPUT,
    FINAL_PROMPT_TEMPLATE, TOP_K_AFTER_RERANK,
    SOURCE_WEIGHT, SIM_WEIGHT, KG_SOURCE_WEIGHT, RAG_SOURCE_WEIGHT,
)
from fusion_module import evidence_formatter as ef
from fusion_module import scorer

logger = logging.getLogger(__name__)


class FusionModule:
    """
    Weighted Score Fusion — KG 1-hop Direct + RAG.

    Score formula:
        final_score(e) = SOURCE_WEIGHT * source_score(e) + SIM_WEIGHT * cosine_sim(e.text, query)

    source_score:
        KG direct evidence  → KG_SOURCE_WEIGHT  (default 1.0)
        RAG chunk evidence  → RAG_SOURCE_WEIGHT (default 0.75)
    """

    def __init__(
        self,
        top_k:         int   = TOP_K_AFTER_RERANK,
        source_weight: float = SOURCE_WEIGHT,
        sim_weight:    float = SIM_WEIGHT,
        kg_src_score:  float = KG_SOURCE_WEIGHT,
        rag_src_score: float = RAG_SOURCE_WEIGHT,
    ):
        """
        Args:
            top_k         : Number of evidence items to keep after scoring.
            source_weight : Weight for source-type component (α).
            sim_weight    : Weight for cosine similarity component (β).
            kg_src_score  : Source score for KG direct evidence.
            rag_src_score : Source score for RAG chunks.
        """
        self.top_k         = top_k
        self.source_weight = source_weight
        self.sim_weight    = sim_weight
        self.kg_src_score  = kg_src_score
        self.rag_src_score = rag_src_score

        logger.info(
            f"[FusionModule] Weighted Fusion | top_k={top_k} | "
            f"α(source)={source_weight} | β(sim)={sim_weight} | "
            f"KG_src={kg_src_score} | RAG_src={rag_src_score}"
        )

    # ──────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────
    def fuse(
        self,
        query:      str,
        kg_direct:  List[Dict[str, Any]] = None,
        rag_chunks: List[Any]            = None,
    ) -> str:
        """
        Full pipeline: collect → format → pre-filter → weighted score → context string.

        Args:
            query      : Input sentence / query.
            kg_direct  : Output of KGRetriever.retrieve_direct_evidence()
            rag_chunks : Output of RAG Branch
                         (list of dicts: {"chunk": str, "score": float, "relation": str})

        Returns:
            Fused context string to insert into the final LLM prompt.
        """
        kg_direct  = kg_direct  or []
        rag_chunks = rag_chunks or []

        # Step 1: Chuẩn hoá sang unified format
        unified_kg  = ef.format_kg_direct(kg_direct)
        unified_rag = ef.format_rag(rag_chunks)

        # Step 2: Pre-filter
        kg_limited   = unified_kg [:MAX_KG_EVIDENCE_INPUT]
        rag_limited  = unified_rag[:MAX_RAG_EVIDENCE_INPUT]
        all_evidence = kg_limited + rag_limited

        logger.info(
            f"[FusionModule] Evidence → KG direct={len(unified_kg)} | "
            f"RAG={len(unified_rag)} | Total after pre-filter={len(all_evidence)}"
        )

        if not all_evidence:
            logger.warning("[FusionModule] No evidence available.")
            return "(No evidence available)"

        # Step 3: Gán ID
        all_evidence = ef.assign_ids(all_evidence)

        # Step 4: Weighted scoring + top-K
        ranked = scorer.weighted_score_evidence(
            all_evidence, query,
            source_weight = self.source_weight,
            sim_weight    = self.sim_weight,
            kg_src_score  = self.kg_src_score,
            rag_src_score = self.rag_src_score,
        )
        ranked = scorer.top_k(ranked, self.top_k)

        logger.info(f"[FusionModule] After scoring: {len(ranked)} items selected.")

        # Step 5: Build context string
        return ef.build_final_context(ranked)

    # ──────────────────────────────────────────────────────
    # Build final inference prompt
    # ──────────────────────────────────────────────────────
    def build_prompt(self, query: str, fused_context: str) -> str:
        """Tạo prompt LLM inference từ context đã fused và query."""
        return FINAL_PROMPT_TEMPLATE.format(
            fused_context = fused_context,
            query         = query,
        )

    def fuse_and_build_prompt(
        self,
        query:      str,
        kg_direct:  List[Dict[str, Any]] = None,
        rag_chunks: List[Any]            = None,
    ) -> str:
        """Shortcut: fuse() + build_prompt() trong một lần gọi."""
        context = self.fuse(query, kg_direct, rag_chunks)
        return self.build_prompt(query, context)
