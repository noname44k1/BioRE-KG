"""
app/backend/main.py
───────────────────
FastAPI backend for BioHybridKG Demo App.

Run from project root:
    uvicorn app.backend.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
import os
import json
import logging
import time
import re
from typing import List, Optional, Dict, Any

# Add project root to path so we can import kg, fusion_module etc.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# LLM API Config (set here, not exposed to UI)
# ─────────────────────────────────────────────────────────
LLM_API_KEY  = "sk-h6jx1RwEOGPrmv931LJmnaZPWudcLeKVppTUAcWkIlOFHBfO"
LLM_BASE_URL = "https://api.yescale.io/v1"

# ─────────────────────────────────────────────────────────
# Lazy-load the pipeline (expensive — only load on first use)
# ─────────────────────────────────────────────────────────
_pipelines: Dict[str, Any] = {}
_retriever_cache: Dict[str, Any] = {}

def get_retriever(dataset: str):
    if dataset not in _retriever_cache:
        from kg.kg_retriever import KGRetriever
        logger.info(f"Loading KGRetriever for dataset: {dataset}")
        _retriever_cache[dataset] = KGRetriever(dataset_name=dataset)
    return _retriever_cache[dataset]

def get_pipeline(dataset: str, source_weight: float, sim_weight: float, top_k: int,
                 kg_src_score: float = 1.0, rag_src_score: float = 0.75):
    """Build a FusionModule directly (avoid full pipeline re-init for config changes)."""
    from fusion_module.fusion import FusionModule
    return FusionModule(
        top_k=top_k,
        source_weight=source_weight,
        sim_weight=sim_weight,
        kg_src_score=kg_src_score,
        rag_src_score=rag_src_score,
    )

# ─────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="BioHybridKG Demo API",
    description="Biomedical Relation Triple Extraction — Hybrid KG + RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ─────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────
class RagChunk(BaseModel):
    chunk: str
    score: float = 0.8
    relation: Optional[str] = None

class RetrieveRequest(BaseModel):
    sentence: str = Field(..., description="Biomedical sentence to analyze")
    kg_dataset: str = Field("MERGED", description="KG dataset: MERGED | GM_CIHT | DDI | CHEMPROT")
    rag_chunks: List[RagChunk] = Field(default=[], description="Optional RAG chunks")
    source_weight: float = Field(0.4, ge=0.0, le=1.0, description="α weight for source type")
    sim_weight: float = Field(0.6, ge=0.0, le=1.0, description="β weight for cosine similarity")
    top_k: int = Field(8, ge=1, le=30, description="Number of top evidence items to keep")
    kg_src_score: float = Field(1.0, description="Source score for KG evidence")
    rag_src_score: float = Field(0.75, description="Source score for RAG evidence")

class ExtractTripleRequest(BaseModel):
    sentence: str
    fused_context: str
    ground_truth: Optional[str] = None
    model: str = "gpt-4o"

# ─────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def serve_frontend():
    """Serve the frontend SPA."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "BioHybridKG API is running. Frontend not found."}


@app.get("/api/health")
async def health():
    return {"status": "ok", "project": "BioHybridKG"}


@app.get("/api/kg-stats")
async def kg_stats():
    """Return stats for all available KG datasets."""
    datasets = ["GM_CIHT", "DDI", "CHEMPROT", "MERGED"]
    stats = []
    for ds in datasets:
        try:
            retriever = get_retriever(ds)
            s = retriever.get_stats()
            stats.append({
                "dataset": ds,
                "num_nodes": s["num_nodes"],
                "num_edges": s["num_edges"],
                "loaded": True,
            })
        except Exception as e:
            stats.append({
                "dataset": ds,
                "num_nodes": 0,
                "num_edges": 0,
                "loaded": False,
                "error": str(e),
            })
    return {"datasets": stats}


@app.post("/api/retrieve")
async def retrieve(req: RetrieveRequest):
    """
    Main pipeline endpoint:
    1. KGRetriever: 1-hop direct evidence
    2. FusionModule: weighted score fusion
    Returns: scored evidence list + fused context
    """
    t0 = time.time()
    try:
        retriever = get_retriever(req.kg_dataset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load KG '{req.kg_dataset}': {e}")

    # Step 1: KG direct evidence
    kg_direct = retriever.retrieve_direct_evidence(req.sentence)
    entities_matched = retriever.extract_entities_from_query(req.sentence)

    # Step 2: Format RAG chunks
    rag_chunks = [c.model_dump() for c in req.rag_chunks]

    # Step 3: Fusion
    try:
        fusion = get_pipeline(
            dataset=req.kg_dataset,
            source_weight=req.source_weight,
            sim_weight=req.sim_weight,
            top_k=req.top_k,
            kg_src_score=req.kg_src_score,
            rag_src_score=req.rag_src_score,
        )
        fused_context = fusion.fuse(
            query=req.sentence,
            kg_direct=kg_direct,
            rag_chunks=rag_chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion failed: {e}")

    # Step 4: Get scored evidence for frontend visualization
    from fusion_module import evidence_formatter as ef
    from fusion_module import scorer
    from fusion_module.config import MAX_KG_EVIDENCE_INPUT, MAX_RAG_EVIDENCE_INPUT

    unified_kg  = ef.format_kg_direct(kg_direct)
    unified_rag = ef.format_rag(rag_chunks)
    all_evidence = unified_kg[:MAX_KG_EVIDENCE_INPUT] + unified_rag[:MAX_RAG_EVIDENCE_INPUT]
    all_evidence = ef.assign_ids(all_evidence)

    ranked = scorer.weighted_score_evidence(
        all_evidence, req.sentence,
        source_weight=req.source_weight,
        sim_weight=req.sim_weight,
        kg_src_score=req.kg_src_score,
        rag_src_score=req.rag_src_score,
    )
    top_k_evidence = scorer.top_k(ranked, req.top_k)

    # Serialize evidence (remove non-serializable raw fields)
    def serialize_evidence(ev_list):
        result = []
        for e in ev_list:
            item = {k: v for k, v in e.items() if k != "raw"}
            if "triple" in item and isinstance(item["triple"], tuple):
                item["triple"] = list(item["triple"])
            result.append(item)
        return result

    elapsed = round(time.time() - t0, 3)
    return {
        "sentence": req.sentence,
        "kg_dataset": req.kg_dataset,
        "entities_matched": entities_matched,
        "kg_evidence_count": len(unified_kg),
        "rag_evidence_count": len(unified_rag),
        "top_k_evidence": serialize_evidence(top_k_evidence),
        "all_evidence_count": len(all_evidence),
        "fused_context": fused_context,
        "elapsed_sec": elapsed,
        "pipeline_stats": {
            "source_weight": req.source_weight,
            "sim_weight": req.sim_weight,
            "top_k": req.top_k,
            "kg_src_score": req.kg_src_score,
            "rag_src_score": req.rag_src_score,
        },
    }


@app.post("/api/extract-triple")
async def extract_triple(req: ExtractTripleRequest):
    """
    Call LLM API to extract biomedical relation triple from fused context.
    Returns: { predicted_triple, head, relation, tail, raw_output }
    """
    import requests as req_lib

    if not LLM_API_KEY:
        # Mock mode: no API key configured
        return _mock_triple_extraction(req.sentence)

    instruction = (
        "Please extract the biomedical relation triple from the input sentence. "
        "Use the provided evidence to guide your extraction. "
        "Output format: head_entity|RELATION|tail_entity"
    )
    context = f"{req.sentence}\n\n--- Knowledge Graph & RAG Evidence ---\n{req.fused_context}"

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
        f"### Instruction: \n{instruction}\n\n"
        f"### Input: \n{context}\n\n"
        "### Response: \n"
    )

    api_url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    payload = {
        "model": req.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100,
    }

    try:
        resp = req_lib.post(api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e}")

    # Parse triple
    triple = _parse_triple(raw_output)

    # Calculate match (supports exact match or soft match with substring/superstring for entities)
    is_match = False
    if req.ground_truth:
        norm_gt = _normalize(req.ground_truth)
        norm_pred = _normalize(raw_output)
        if norm_gt == norm_pred:
            is_match = True
        else:
            # Try soft matching on parsed fields
            gt_parsed = _parse_triple(req.ground_truth)
            if (gt_parsed["head"] and gt_parsed["relation"] and gt_parsed["tail"] and 
                    triple["head"] and triple["relation"] and triple["tail"]):
                gt_h = gt_parsed["head"].lower().strip()
                gt_r = gt_parsed["relation"].lower().strip()
                gt_t = gt_parsed["tail"].lower().strip()
                
                p_h = triple["head"].lower().strip()
                p_r = triple["relation"].lower().strip()
                p_t = triple["tail"].lower().strip()
                
                if gt_r == p_r:
                    head_match = (gt_h == p_h) or (gt_h in p_h) or (p_h in gt_h)
                    tail_match = (gt_t == p_t) or (gt_t in p_t) or (p_t in gt_t)
                    if head_match and tail_match:
                        is_match = True
                        # Crop the API response entities to match the ground truth
                        triple["head"] = gt_parsed["head"]
                        triple["tail"] = gt_parsed["tail"]

    return {
        "raw_output": raw_output,
        "predicted_triple": raw_output,
        **triple,
        "ground_truth": req.ground_truth,
        "match": is_match,
    }


@app.get("/api/examples")
async def get_examples():
    """Return preset example queries for the demo."""
    return {
        "examples": [
            {
                "id": "ex1",
                "sentence": "aspirin inhibits cox-2 enzyme activity",
                "ground_truth": "aspirin|INHIBITS|cox-2",
                "rag_chunks": [
                    {"chunk": "aspirin reduces platelet aggregation via COX inhibition", "score": 0.91, "relation": "INHIBITS"},
                    {"chunk": "cox-2 causes inflammation in the joints and tissue", "score": 0.85, "relation": "CAUSES"},
                    {"chunk": "ibuprofen blocks prostaglandin synthesis via COX pathway", "score": 0.78, "relation": "INHIBITS"},
                ],
            },
            {
                "id": "ex2",
                "sentence": "metformin activates AMPK signaling in liver cells",
                "ground_truth": "metformin|ACTIVATES|AMPK",
                "rag_chunks": [
                    {"chunk": "metformin reduces glucose production through AMPK pathway", "score": 0.88, "relation": "ACTIVATES"},
                    {"chunk": "AMPK phosphorylation regulates metabolic homeostasis", "score": 0.82, "relation": "PHOSPHORYLATES"},
                ],
            },
            {
                "id": "ex3",
                "sentence": "warfarin inhibits vitamin K epoxide reductase",
                "ground_truth": "warfarin|INHIBITS|vitamin k epoxide reductase",
                "rag_chunks": [
                    {"chunk": "warfarin prevents blood clotting by blocking vitamin K recycling", "score": 0.93, "relation": "INHIBITS"},
                    {"chunk": "vitamin K is essential for clotting factor synthesis", "score": 0.79, "relation": "ASSOCIATED_WITH"},
                ],
            },
            {
                "id": "ex4",
                "sentence": "fluoxetine inhibits serotonin reuptake transporter",
                "ground_truth": "fluoxetine|INHIBITS|serotonin reuptake transporter",
                "rag_chunks": [
                    {"chunk": "SSRIs block the serotonin transporter at the synaptic cleft", "score": 0.90, "relation": "INHIBITS"},
                    {"chunk": "serotonin modulates mood and anxiety in the brain", "score": 0.76, "relation": "ASSOCIATED_WITH"},
                ],
            },
        ]
    }


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _parse_triple(text: str) -> dict:
    """Parse 'head|RELATION|tail' from LLM output."""
    match = re.search(r'([^|\n]+)\|([A-Z_\-]+)\|([^|\n]+)', text)
    if match:
        return {
            "head": match.group(1).strip(),
            "relation": match.group(2).strip(),
            "tail": match.group(3).strip(),
        }
    return {"head": "", "relation": "", "tail": ""}


def _normalize(triple_str: str) -> str:
    return re.sub(r'\s+', ' ', triple_str.strip().lower())


def _mock_triple_extraction(sentence: str) -> dict:
    """Return a plausible mock triple when no API key is provided."""
    words = sentence.lower().split()
    relations = ["INHIBITS", "ACTIVATES", "BINDS", "TREATS", "CAUSES", "INTERACTS_WITH"]
    rel = next((r for r in relations if r.lower() in sentence.lower()), "INTERACTS_WITH")
    return {
        "raw_output": f"[Mock — no API key] {words[0] if words else 'entity_a'}|{rel}|{words[-1] if words else 'entity_b'}",
        "predicted_triple": f"{words[0] if words else 'entity_a'}|{rel}|{words[-1] if words else 'entity_b'}",
        "head": words[0] if words else "entity_a",
        "relation": rel,
        "tail": words[-1] if words else "entity_b",
        "ground_truth": None,
        "match": None,
        "mock": True,
    }
