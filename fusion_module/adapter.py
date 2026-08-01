"""
adapter.py
──────────
Bước 8 Mới: Thay thế chuck_triplet_progress_test.py trong pipeline.

INPUT : test_chuck_5_triplet_llama2_13b_right_2.json   ← output của RAG pipeline (Bước 7)
           Mỗi dòng: {"instruction": ..., "context": sentence, "response": "h|R|t"}
           -> instruction chứa KNN examples + RAG candidate chunks (pipe-separated)
           -> context chứa câu văn gốc

OUTPUT: fused_test_input.json                           ← input cho Bước 9 (inference)
           Mỗi dòng: {"instruction": augmented_instruction, "context": sentence, "response": "h|R|t"}
           -> instruction = KNN examples + RAG candidate chunks + KG evidence block
           -> context GIỮ NGUYÊN (original sentence)
           -> response GIỮ NGUYÊN

Workflow:
  rag_jsonl (from Step 7)
       ↓ đọc từng record
       ├─ extract sentence từ "context" field
       ├─ extract RAG chunks từ "instruction" field (sau "Examples:")
       │     format: "Context: {chunk} Response: {relation}"
       ├─ KGRetriever → 1-hop direct evidence (không có context sentence, chỉ triple)
       ├─ WeightedFusion → top-K evidence theo α*source_score + β*cosine_sim
       └─ Ghi record mới với augmented context

Cách chạy:
  python -m fusion_module.adapter                    # test 3 sample đầu
  python -m fusion_module.adapter --full             # toàn bộ test set
  python -m fusion_module.adapter --full --kg-dataset CHEMPROT
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Optional, List, Dict, Any

from fusion_module.config import INFERENCE_INSTRUCTION, INFERENCE_CONTEXT_TEMPLATE, INFERENCE_INSTRUCTION_TEMPLATE
from fusion_module.pipeline import BioHybridPipeline

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# Default paths (relative to project root)
# ══════════════════════════════════════════════════════════
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RAG_OUTPUT = os.path.join(
    _ROOT, "2_relation_data_to_triple_train_data",
    "test_chuck_5_triplet_llama2_13b_right_2.json"
)
DEFAULT_FUSED_OUTPUT = os.path.join(
    _ROOT, "4_generation_triple_model",
    "fused_test_input.json"
)


# ══════════════════════════════════════════════════════════
# RAG chunk parser
# ══════════════════════════════════════════════════════════

def _extract_rag_chunks(instruction: str) -> List[Dict[str, Any]]:
    """
    Parse các RAG chunks từ field "instruction".

    Instruction chứa KNN few-shot examples theo format:
      "...Context: {chunk_text} Response: {head}|{RELATION}|{tail}..."
    hoặc candidate chunks từ step 5:
      "...Context: {chunk_text} Response: {RELATION_NAME}..."

    Cách cũ (split |) bị vỡ vì | cũng xuất hiện trong "head|RELATION|tail".
    Cách mới dùng regex để match đúng block "Context:...Response:...".
    """
    import re
    chunks = []

    # Tìm tất cả block "Context: {text} Response: {response}"
    # Dùng lookahead để biết khi nào block kết thúc (trước Context: tiếp theo hoặc cuối chuỗi)
    pattern = re.compile(
        r'Context:\s*(.*?)\s+Response:\s*([^\n]+?)(?=\s+Context:|\s*$)',
        re.DOTALL
    )

    for m in pattern.finditer(instruction):
        chunk_text = m.group(1).strip()
        response   = m.group(2).strip()

        # Bỏ qua chunk quá ngắn (noise)
        if len(chunk_text) < 5:
            continue

        # Xác định relation từ response
        # Format A: "head|RELATION|tail"  → lấy phần giữa
        # Format B: "RELATION_NAME"       → lấy nguyên
        parts = response.split("|")
        if len(parts) == 3:
            relation = parts[1].strip()   # head|RELATION|tail
        elif len(parts) == 1:
            relation = parts[0].strip()   # RELATION_NAME
        else:
            relation = response           # fallback

        chunks.append({
            "chunk":    chunk_text,
            "relation": relation,
            "score":    None,
        })

    return chunks



# ══════════════════════════════════════════════════════════
# Main Adapter Class
# ══════════════════════════════════════════════════════════

class FusionAdapter:
    """
    Adapter kết nối RAG pipeline output với KG Fusion Module.

    Thay thế Bước 8 (chuck_triplet_progress_test.py) trong pipeline.
    """

    def __init__(
        self,
        kg_dataset:    str = "MERGED",
        top_k:         int = 8,
        source_weight: float = 0.4,
        sim_weight:    float = 0.6,
    ):
        self.pipeline = BioHybridPipeline(
            kg_dataset    = kg_dataset,
            top_k         = top_k,
            source_weight = source_weight,
            sim_weight    = sim_weight,
        )

    # ──────────────────────────────────────────────────────
    # Build fused context for ONE record
    # ──────────────────────────────────────────────────────
    def _build_fused_context(
        self,
        sentence:   str,
        rag_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Chạy KG retrieval + fusion cho một câu và trả về fused context string.

        Args:
            sentence   : Câu văn gốc (từ field "context").
            rag_chunks : RAG chunks đã parse từ field "instruction".

        Returns:
            Fused context string gồm KG evidence + RAG evidence đã rerank.
        """
        return self.pipeline.get_fused_context(
            query      = sentence,
            rag_chunks = rag_chunks,
        )

    # ──────────────────────────────────────────────────────
    # Main pipeline method: RAG output → Fused output
    # ──────────────────────────────────────────────────────
    def build_from_rag_output(
        self,
        rag_jsonl:    str,
        output_jsonl: str,
        max_samples:  Optional[int] = None,
    ) -> None:
        """
        Đọc file RAG pipeline output (Bước 7), thêm KG evidence, ghi file mới cho inference.

        Input format (mỗi dòng):
            {
              "instruction": "...KNN examples + RAG candidate chunks...",
              "context":     "original sentence",
              "response":    "head|RELATION|tail"
            }

        Output format (mỗi dòng):
            {
              "instruction": "...KNN examples + ...\\n\\n--- KG Evidence ---\\n..." ← AUGMENTED
              "context":     "original sentence"                                     ← GIỮ NGUYÊN
              "response":    "head|RELATION|tail"                                ← GIỮ NGUYÊN
              "category":    "triplet_extraction"                                ← THÊM MỚI
            }

        Args:
            rag_jsonl    : Đường dẫn đến test_chuck_5_triplet_llama2_13b_right_2.json
            output_jsonl : Đường dẫn output cho inference (fused_test_input.json)
            max_samples  : Giới hạn số sample (None = tất cả)
        """
        if not os.path.exists(rag_jsonl):
            raise FileNotFoundError(
                f"RAG output file không tồn tại: {rag_jsonl}\n"
                f"Hãy chạy Bước 7 (chuck_triplet_progress_test.py) trước."
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

        total = written = skipped = 0
        logger.info(f"[FusionAdapter] Đọc RAG output: {rag_jsonl}")
        logger.info(f"[FusionAdapter] Ghi fused output: {output_jsonl}")

        with open(rag_jsonl, "r", encoding="utf-8") as fin, \
             open(output_jsonl, "w", encoding="utf-8") as fout:

            for raw_line in fin:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                total += 1

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    logger.warning(f"  [SKIP #{total}] JSON parse error: {e}")
                    skipped += 1
                    continue

                # ── Lấy fields từ record gốc ─────────────────────
                instruction  = record.get("instruction", "")
                sentence     = record.get("context", "").strip()
                ground_truth = record.get("response", "")

                if not sentence:
                    logger.warning(f"  [SKIP #{total}] context field rỗng")
                    skipped += 1
                    continue

                # ── Parse RAG chunks từ instruction ───────────────
                # Bỏ qua việc trích xuất RAG chunks vì instruction đã chứa sẵn
                rag_chunks = []
                logger.debug(f"  [#{total}] Skipping RAG chunk parsing as they are already included")

                # ── KG Retrieval + Fusion ─────────────────────────
                try:
                    fused_context = self._build_fused_context(sentence, rag_chunks)
                except Exception as e:
                    logger.warning(f"  [SKIP #{total}] Fusion error: {e}")
                    skipped += 1
                    continue

                # ── Tạo augmented instruction field ───────────────────
                # instruction = original instruction + KG evidence block
                # context GIỮ NGUYÊN là original sentence
                augmented_instruction = INFERENCE_INSTRUCTION_TEMPLATE.format(
                    instruction   = instruction,
                    fused_context = fused_context,
                )

                # ── Ghi record mới ────────────────────────────────
                new_record = {
                    "instruction": augmented_instruction, # AUGMENTED với KG & RAG
                    "context":     sentence,              # GIỮ NGUYÊN
                    "response":    ground_truth,          # GIỮ NGUYÊN
                    "category":    record.get("category", "triplet_extraction"),
                }
                fout.write(json.dumps(new_record, ensure_ascii=False))
                fout.write("\n")
                written += 1

                if written % 100 == 0:
                    logger.info(f"  Đã xử lý {written} samples...")

                if max_samples and written >= max_samples:
                    logger.info(f"  Đạt giới hạn max_samples={max_samples}, dừng.")
                    break

        logger.info(
            f"[FusionAdapter] Hoàn thành. "
            f"Tổng đọc={total} | Ghi={written} | Bỏ qua={skipped}"
        )
        logger.info(f"[FusionAdapter] Output: {output_jsonl}")


# ══════════════════════════════════════════════════════════
# CLI entry-point
# Chạy: python -m fusion_module.adapter           (3 sample, no LLM)
#        python -m fusion_module.adapter --full    (toàn bộ, no LLM)
#        python -m fusion_module.adapter --full --backend ollama
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="BioHybridKG Fusion Adapter")
    parser.add_argument("--rag-input",  default=DEFAULT_RAG_OUTPUT,
                        help="Path to RAG output JSONL (Step 7)")
    parser.add_argument("--output",     default=DEFAULT_FUSED_OUTPUT,
                        help="Path to write fused output JSONL")
    parser.add_argument("--kg-dataset", default="MERGED",
                        choices=["GM_CIHT", "DDI", "CHEMPROT", "MERGED"])
    parser.add_argument("--top-k",      default=2, type=int,
                        help="Number of evidence items to keep (default=2)")
    parser.add_argument("--full",       action="store_true",
                        help="Process all samples (default: only 3 for test)")
    args = parser.parse_args()

    adapter = FusionAdapter(
        kg_dataset = args.kg_dataset,
        top_k      = args.top_k,
    )

    max_samples = None if args.full else 3

    # Nếu file RAG input không có, dùng mock để test
    if not os.path.exists(args.rag_input):
        logger.warning(f"RAG input không tồn tại: {args.rag_input}")
        logger.warning("Chạy với MOCK DATA (3 record giả) để test pipeline...")

        import tempfile, json
        mock_records = [
            {
                "instruction": (
                    "Please extract the relation triple from the context.\n"
                    "Examples: 1. Context: aspirin blocks pain Response: aspirin|INHIBITS|pain\n"
                    "|Context: aspirin reduces cox-2 activity Response: INHIBITS"
                    "|Context: ibuprofen inhibits prostaglandin Response: INHIBITS"
                ),
                "context": "Aspirin is known to inhibit the enzyme cox-2.",
                "response": "aspirin|INHIBITS|cox-2",
            },
            {
                "instruction": (
                    "Please extract the relation triple from the context.\n"
                    "|Context: methotrexate blocks dihydrofolate Response: INHIBITS"
                    "|Context: folic acid is substrate of enzyme Response: SUBSTRATE"
                ),
                "context": "Methotrexate acts as an inhibitor of dihydrofolate reductase.",
                "response": "methotrexate|INHIBITS|dihydrofolate reductase",
            },
            {
                "instruction": (
                    "Please extract the relation triple from the context.\n"
                    "|Context: colchicine binds tubulin protein Response: INHIBITS"
                    "|Context: microtubule polymerization is inhibited Response: INHIBITS"
                ),
                "context": "Colchicine binds to tubulin and inhibits microtubule polymerization.",
                "response": "colchicine|INHIBITS|microtubule polymerization",
            },
        ]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False, encoding="utf-8")
        for r in mock_records:
            tmp.write(json.dumps(r) + "\n")
        tmp.close()
        args.rag_input = tmp.name

    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f" FusionAdapter — Bước 8 Mới (Weighted Fusion)")
    print(f" KG: {args.kg_dataset} | top_k: {args.top_k} | Full: {args.full}")
    print(SEP)

    adapter.build_from_rag_output(
        rag_jsonl    = args.rag_input,
        output_jsonl = args.output,
        max_samples  = max_samples,
    )

    # In 3 record đầu để kiểm tra
    print(f"\n--- Sample output records ({args.output}) ---")
    try:
        with open(args.output, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                rec = json.loads(line)
                print(f"\n[Record {i+1}]")
                print(f"  response   : {rec['response']}")
                print(f"  context    :\n{rec['context'][:400]}")
                print()
    except FileNotFoundError:
        print("(output file không tồn tại)")
    print(SEP)
    print(" DONE")
    print(SEP)
