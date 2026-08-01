"""
scorer.py
─────────
Weighted evidence scorer for BioHybridKG Fusion Module.

Score formula:
    final_score(e) = SOURCE_WEIGHT * source_score(e) + SIM_WEIGHT * cosine_sim(e.text, query)

source_score:
    KG direct evidence  → KG_SOURCE_WEIGHT  (1-hop triples, highest trust)
    RAG chunk evidence  → RAG_SOURCE_WEIGHT (text chunks)
"""
from __future__ import annotations
from typing import List, Dict, Any
import math
import re

from fusion_module.config import (
    SOURCE_WEIGHT, SIM_WEIGHT,
    KG_SOURCE_WEIGHT, RAG_SOURCE_WEIGHT,
    TOP_K_AFTER_RERANK,
)
from fusion_module.evidence_formatter import TYPE_KG_DIRECT, TYPE_RAG


# ──────────────────────────────────────────────────────────
# Utility: lightweight TF-IDF cosine similarity
# ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf_vector(tokens: List[str], vocab: Dict[str, int]) -> Dict[int, float]:
    tf: Dict[int, float] = {}
    for t in tokens:
        idx = vocab.get(t)
        if idx is not None:
            tf[idx] = tf.get(idx, 0) + 1
    norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
    return {k: v / norm for k, v in tf.items()}


def cosine_sim_text(text_a: str, text_b: str) -> float:
    """Lightweight cosine similarity via TF vectors. No external dependencies."""
    toks_a = _tokenize(text_a)
    toks_b = _tokenize(text_b)
    vocab  = {t: i for i, t in enumerate(set(toks_a) | set(toks_b))}
    if not vocab:
        return 0.0
    va  = _tf_vector(toks_a, vocab)
    vb  = _tf_vector(toks_b, vocab)
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in set(va) | set(vb))
    return float(dot)


# ──────────────────────────────────────────────────────────
# Main: Weighted Evidence Scorer
# ──────────────────────────────────────────────────────────

def weighted_score_evidence(
    evidence_list: List[Dict[str, Any]],
    query: str,
    source_weight: float = SOURCE_WEIGHT,
    sim_weight:    float = SIM_WEIGHT,
    kg_src_score:  float = KG_SOURCE_WEIGHT,
    rag_src_score: float = RAG_SOURCE_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    Assign a weighted relevance score to each evidence item.

    Score formula:
        final_score = source_weight * source_score + sim_weight * cosine_sim(text, query)

    source_score:
        TYPE_KG_DIRECT → kg_src_score  (default 1.0)
        TYPE_RAG       → rag_src_score (default 0.75)

    Args:
        evidence_list : Unified evidence list (with "type", "text" fields).
        query         : The input query sentence.
        source_weight : Weight for source-type component (α).
        sim_weight    : Weight for text similarity component (β).
        kg_src_score  : Source score for KG direct evidence.
        rag_src_score : Source score for RAG chunks.

    Returns:
        Same list with "score" field added, sorted descending by score.
    """
    for item in evidence_list:
        ev_type   = item.get("type", "")
        text      = item.get("text", "")

        # Source score based on evidence type
        if ev_type == TYPE_KG_DIRECT:
            src_score = kg_src_score
        else:  # RAG or unknown
            src_score = rag_src_score

        # Cosine similarity between evidence text and query
        sim = cosine_sim_text(text, query)

        item["score"] = source_weight * src_score + sim_weight * sim

    return sorted(evidence_list, key=lambda x: x.get("score", 0.0), reverse=True)


def top_k(evidence_list: List[Dict[str, Any]], k: int = TOP_K_AFTER_RERANK) -> List[Dict[str, Any]]:
    """Return the top-k items by score (assumes weighted_score_evidence already called)."""
    return evidence_list[:k]

