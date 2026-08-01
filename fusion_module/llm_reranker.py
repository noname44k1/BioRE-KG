"""
llm_reranker.py
───────────────
LLM-based Listwise Reranking (Strategy 3).

Hỗ trợ 2 backend:
  - "local"  : HuggingFace Transformers chạy trên máy (GPU/CPU)
  - "ollama" : Gọi Ollama API server đang chạy local

Cách chạy:
─────────────────────────────────────────────────────────────
[Backend: local]
  1. Đặt đường dẫn model trong config.py:
       LLM_MODEL_PATH = r"D:\\models\\MedLLaMA_13B"   # hoặc để None để dùng HF Hub
  2. Khởi tạo:
       reranker = LLMReranker(backend="local")

[Backend: ollama]
  1. Cài và chạy Ollama: https://ollama.com/download
  2. Pull model:  ollama pull llama2
  3. Đặt model name trong config.py:
       OLLAMA_MODEL    = "llama2"          # hoặc "mistral", "phi3", "medllama2", ...
       OLLAMA_BASE_URL = "http://localhost:11434"
  4. Khởi tạo:
       reranker = LLMReranker(backend="ollama")
─────────────────────────────────────────────────────────────

Approach: RankGPT-style Listwise Reranking.
  1. Present all evidence items (KG + RAG) with IDs to the LLM.
  2. LLM returns IDs sorted by relevance: "E3, E1, E5, E2, ..."
  3. Parse → reorder evidence → return top-K.
  4. Fallback to cosine+hop scoring if LLM call fails or parse fails.
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Any, Optional

from fusion_module.config import (
    LLM_MODEL_PATH, LLM_HF_REPO,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    RERANKER_MAX_NEW_TOKENS, RERANKER_TEMPERATURE, RERANKER_DO_SAMPLE,
    RERANK_SYSTEM_PROMPT, RERANK_USER_TEMPLATE,
    TOP_K_AFTER_RERANK,
)
from fusion_module.evidence_formatter import build_evidence_block
from fusion_module import scorer as fallback_scorer

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# Backend 1: Local HuggingFace Transformers
# ══════════════════════════════════════════════════════════

class _LocalHFBackend:
    """
    Runs inference using a local HuggingFace causal LM.
    Model is loaded lazily on first use.

    Cách dùng:
        # Trong config.py đặt:
        LLM_MODEL_PATH = r"D:\\models\\MedLLaMA_13B"
        # Sau đó:
        reranker = LLMReranker(backend="local")
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model     = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            logger.info(f"[LocalHF] Loading model: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True,   # 8-bit quantisation to save VRAM
            )
            self._model.eval()
            logger.info("[LocalHF] Model loaded (8-bit quantised).")
        except Exception as e:
            logger.error(f"[LocalHF] Failed to load model: {e}")
            raise

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens = RERANKER_MAX_NEW_TOKENS,
                temperature    = RERANKER_TEMPERATURE if RERANKER_DO_SAMPLE else 1.0,
                do_sample      = RERANKER_DO_SAMPLE,
                pad_token_id   = self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════
# Backend 2: Ollama API
# ══════════════════════════════════════════════════════════

class _OllamaBackend:
    """
    Calls a running Ollama server for inference.

    Cách dùng:
        1. Cài Ollama: https://ollama.com/download
        2. Chạy:  ollama serve              (tự động khi mở app trên Windows/Mac)
        3. Pull:  ollama pull llama2        (hoặc phi3, mistral, medllama2, ...)
        4. Đặt trong config.py:
               OLLAMA_MODEL    = "llama2"
               OLLAMA_BASE_URL = "http://localhost:11434"
        5. Khởi tạo:
               reranker = LLMReranker(backend="ollama")
    """

    def __init__(self, base_url: str, model: str):
        # Strip trailing slash
        self.base_url = base_url.rstrip("/")
        self.model    = model

    def generate(self, prompt: str) -> str:
        try:
            import urllib.request
            import json

            payload = json.dumps({
                "model":  self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": RERANKER_TEMPERATURE,
                    "num_predict": RERANKER_MAX_NEW_TOKENS,
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                url     = f"{self.base_url}/api/generate",
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            return data.get("response", "").strip()

        except Exception as e:
            logger.error(f"[Ollama] Request failed: {e}")
            return ""


# ══════════════════════════════════════════════════════════
# Ranking Output Parser
# ══════════════════════════════════════════════════════════

def _parse_ranking(llm_output: str, all_ids: List[str]) -> List[str]:
    """
    Parse LLM output like "E3, E1, E5, E2" → ordered list of valid IDs.
    - Ignores unknown IDs.
    - Appends any IDs the LLM forgot at the end (ensures completeness).
    """
    found  = re.findall(r"E\d+", llm_output, re.IGNORECASE)
    found  = [f.upper() for f in found]

    seen   = set()
    ranked = []
    for eid in found:
        if eid in all_ids and eid not in seen:
            ranked.append(eid)
            seen.add(eid)

    # Append anything the LLM missed
    for eid in all_ids:
        if eid not in seen:
            ranked.append(eid)

    return ranked


# ══════════════════════════════════════════════════════════
# Main Reranker Class
# ══════════════════════════════════════════════════════════

class LLMReranker:
    """
    Strategy 3: LLM Listwise Reranking.

    Cách dùng:

        # --- Local HuggingFace model ---
        # (Đặt LLM_MODEL_PATH trong config.py trước)
        reranker = LLMReranker(backend="local")

        # --- Ollama ---
        # (Đảm bảo `ollama serve` đang chạy và model đã pull)
        reranker = LLMReranker(backend="ollama")
        # Hoặc override model/URL:
        reranker = LLMReranker(backend="ollama",
                               ollama_model="mistral",
                               ollama_url="http://localhost:11434")

        # --- Rerank ---
        top_evidence = reranker.rerank(evidence_list, query="...")
    """

    VALID_BACKENDS = {"local", "ollama"}

    def __init__(
        self,
        backend:      str = "ollama",
        ollama_model: Optional[str] = None,
        ollama_url:   Optional[str] = None,
        top_k:        int = TOP_K_AFTER_RERANK,
    ):
        """
        Args:
            backend      : "local"  → HuggingFace Transformers (local GPU/CPU)
                           "ollama" → Ollama API server (http://localhost:11434)
            ollama_model : Override OLLAMA_MODEL from config (optional).
            ollama_url   : Override OLLAMA_BASE_URL from config (optional).
            top_k        : Number of top evidence items to return after reranking.
        """
        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. "
                f"Choose from: {self.VALID_BACKENDS}"
            )

        self.top_k = top_k

        if backend == "local":
            model_path = LLM_MODEL_PATH or LLM_HF_REPO
            self._llm  = _LocalHFBackend(model_path=model_path)
            logger.info(f"[LLMReranker] Backend=local | model={model_path} | top_k={top_k}")

        elif backend == "ollama":
            model      = ollama_model or OLLAMA_MODEL
            url        = ollama_url   or OLLAMA_BASE_URL
            self._llm  = _OllamaBackend(base_url=url, model=model)
            logger.info(f"[LLMReranker] Backend=ollama | model={model} | url={url} | top_k={top_k}")

    # ──────────────────────────────────────────────────────
    # Core rerank
    # ──────────────────────────────────────────────────────
    def rerank(
        self,
        evidence_list: List[Dict[str, Any]],
        query:         str,
    ) -> List[Dict[str, Any]]:
        """
        Rerank evidence by relevance to query using the LLM.

        Args:
            evidence_list : Unified evidence dicts (must have "id" field assigned).
            query         : Input query / sentence.

        Returns:
            Top-K evidence items sorted by LLM-determined relevance.
            Falls back to cosine+hop scoring if LLM call fails.
        """
        if not evidence_list:
            return []

        all_ids        = [e["id"] for e in evidence_list]
        evidence_block = build_evidence_block(evidence_list)
        prompt         = RERANK_USER_TEMPLATE.format(
            query          = query,
            evidence_block = evidence_block,
        )

        logger.info(f"[LLMReranker] Reranking {len(evidence_list)} items...")

        try:
            llm_output = self._llm.generate(prompt)
            logger.debug(f"[LLMReranker] Raw output: {llm_output!r}")
        except Exception as e:
            logger.warning(f"[LLMReranker] LLM call failed ({e}). Using fallback scorer.")
            return self._fallback(evidence_list, query)

        ranked_ids = _parse_ranking(llm_output, all_ids)
        logger.info(f"[LLMReranker] Parsed ranking: {ranked_ids}")

        if not ranked_ids:
            logger.warning("[LLMReranker] Parse failed. Using fallback scorer.")
            return self._fallback(evidence_list, query)

        id_to_item = {e["id"]: e for e in evidence_list}
        ranked     = [id_to_item[eid] for eid in ranked_ids if eid in id_to_item]
        for rank, item in enumerate(ranked):
            item["llm_rank"] = rank + 1

        return ranked[: self.top_k]

    # ──────────────────────────────────────────────────────
    # Fallback
    # ──────────────────────────────────────────────────────
    def _fallback(
        self,
        evidence_list: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Cosine TF-IDF + hop penalty fallback when LLM is unavailable."""
        logger.info("[LLMReranker] Fallback: cosine + hop scoring.")
        scored = fallback_scorer.score_evidence(evidence_list, query)
        return fallback_scorer.top_k(scored, self.top_k)
