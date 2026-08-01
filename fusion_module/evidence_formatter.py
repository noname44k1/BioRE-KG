"""
evidence_formatter.py
─────────────────────
Converts raw evidence dicts from KGRetriever / MultiHopReasoner / RAG Branch
into clean, numbered text blocks for display + LLM consumption.
"""
from typing import List, Dict, Any


# ──────────────────────────────────────────────────────────
# Evidence type constants
# ──────────────────────────────────────────────────────────
TYPE_KG_DIRECT   = "kg_direct"
TYPE_KG_MULTIHOP = "kg_multihop"
TYPE_RAG         = "rag"


def format_kg_direct(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalise 1-hop KG evidence dicts into the unified evidence format.

    Input (from KGRetriever.retrieve_direct_evidence):
        [{"subject": ..., "predicate": ..., "object": ...,
          "context": ..., "source": ..., "hop": 1, "direction": ...}]

    Output (unified):
        [{"id": "E1", "type": "kg_direct", "text": ..., "meta": {...}}]
    """
    unified = []
    for item in evidence_list:
        subj = item.get("subject", "")
        pred = item.get("predicate", "")
        obj  = item.get("object", "")
        ctx  = item.get("context", "")
        src  = item.get("source", "")
        hop  = item.get("hop", 1)
        direction = item.get("direction", "outgoing")

        triple_str = f"{subj} --{pred}--> {obj}"
        text = f"[KG 1-hop | {direction} | src={src}] {triple_str}"

        unified.append({
            "type": TYPE_KG_DIRECT,
            "text": text,
            "triple": (subj, pred, obj),
            "hop":    hop,
            "source": src,
            "raw":    item,
        })
    return unified


def format_kg_multihop(paths: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Normalise multi-hop paths into the unified evidence format.

    Input (from MultiHopReasoner.find_paths / reason_over_evidence):
        [ [{"subject":..,"predicate":..,"object":..,"context":..,"source":..}, ...], ...]

    Output (unified):
        [{"id": "E5", "type": "kg_multihop", "text": ..., "meta": {...}}]
    """
    unified = []
    for path in paths:
        if not path:
            continue
        hops = len(path)
        steps = []
        sources = set()
        for edge in path:
            steps.append(f"{edge.get('subject','')} --{edge.get('predicate','')}--> {edge.get('object','')}")
            sources.add(edge.get("source", ""))
        path_str = "  →  ".join(steps)
        text = f"[KG {hops}-hop | src={','.join(sources)}] {path_str}"

        unified.append({
            "type":   TYPE_KG_MULTIHOP,
            "text":   text,
            "path":   path,
            "hop":    hops,
            "source": list(sources),
            "raw":    path,
        })
    return unified


def format_rag(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalise RAG chunks into the unified evidence format.

    Expected input format (flexible):
        [{"chunk": "text", "relation": "INHIBITS", "score": 0.92}, ...]
      OR
        [{"text": "text", "score": 0.88}, ...]
      OR plain strings.

    Output (unified):
        [{"id": "E3", "type": "rag", "text": ..., "meta": {...}}]
    """
    unified = []
    for item in chunks:
        if isinstance(item, str):
            chunk_text = item
            score = None
            relation = None
        else:
            chunk_text = item.get("chunk") or item.get("text") or str(item)
            score      = item.get("score")
            relation   = item.get("relation")

        score_str    = f" | sim={score:.3f}" if score is not None else ""
        relation_str = f" | rel={relation}" if relation else ""
        text = f"[RAG chunk{score_str}{relation_str}] {chunk_text}"

        unified.append({
            "type":     TYPE_RAG,
            "text":     text,
            "chunk":    chunk_text,
            "score":    score,
            "relation": relation,
            "raw":      item,
        })
    return unified


def assign_ids(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign sequential IDs E1, E2, ... to a unified evidence list."""
    for i, item in enumerate(evidence_list):
        item["id"] = f"E{i + 1}"
    return evidence_list


def build_evidence_block(evidence_list: List[Dict[str, Any]]) -> str:
    """
    Build the numbered evidence block string for the LLM reranker prompt.

    Example output:
        [E1] [KG 1-hop | outgoing | src=DDI] aspirin --INHIBITS--> cox-2
          Context: aspirin inhibits cox-2 enzyme...
        [E2] [RAG chunk | sim=0.92 | rel=INHIBITS] colchicine binds to microtubule
    """
    lines = []
    for item in evidence_list:
        eid  = item.get("id", "?")
        text = item.get("text", "")
        lines.append(f"[{eid}] {text}")
    return "\n".join(lines)


def build_final_context(ranked_evidence: List[Dict[str, Any]]) -> str:
    """
    Build the final fused context string from ranked/selected evidence.
    This is what goes into the LLM inference prompt.
    """
    kg_direct   = [e for e in ranked_evidence if e["type"] == TYPE_KG_DIRECT]
    kg_multihop = [e for e in ranked_evidence if e["type"] == TYPE_KG_MULTIHOP]
    rag         = [e for e in ranked_evidence if e["type"] == TYPE_RAG]

    sections = []

    if kg_direct:
        lines = ["=== KG Direct Evidence ==="]
        for e in kg_direct:
            lines.append(e["text"])
        sections.append("\n".join(lines))

    if kg_multihop:
        lines = ["=== KG Multi-hop Paths ==="]
        for e in kg_multihop:
            lines.append(e["text"])
        sections.append("\n".join(lines))

    if rag:
        lines = ["=== RAG Text Evidence ==="]
        for e in rag:
            lines.append(e["text"])
        sections.append("\n".join(lines))

    return "\n\n".join(sections) if sections else "(No evidence found)"
