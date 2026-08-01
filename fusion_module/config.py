import os

# =========================================================
# Base Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================================================
# Weighted Fusion — Score Weights
# =========================================================
# final_score(e) = SOURCE_WEIGHT * source_score(e) + SIM_WEIGHT * cosine_sim(e, query)
#
# source_score:
#   KG direct evidence  → KG_SOURCE_WEIGHT  (1-hop triples, highest priority)
#   RAG chunk evidence  → RAG_SOURCE_WEIGHT (text chunks from retriever)
#
SOURCE_WEIGHT     = 0.4   # weight for source type (KG vs RAG)
SIM_WEIGHT        = 0.6   # weight for cosine similarity with query

KG_SOURCE_WEIGHT  = 1.0   # KG 1-hop direct edges — highest trust
RAG_SOURCE_WEIGHT = 0.75  # RAG chunks — slightly lower than KG

# =========================================================
# Fusion Configuration
# =========================================================
# Maximum evidence items before scoring (pre-filter)
MAX_KG_EVIDENCE_INPUT  = 20   # KG 1-hop direct edges
MAX_RAG_EVIDENCE_INPUT = 10   # RAG chunks from instruction field

# How many top evidence items to keep AFTER scoring
TOP_K_AFTER_RERANK = 8

# =========================================================
# Prompt Templates — LLM Inference
# =========================================================
#
# Pipeline flow cho inference (chuck5_generation_8000.py):
#
#   make_inference(instruction, context) builds:
#     "Below is an instruction...
#      ### Instruction: \n{instruction}\n\n
#      ### Input: \n{context}\n\n
#      ### Response: \n"
#
#   eval.py parses predicted output:
#     line["predicted"].split("\n\n")[3].split("\n")[1]  → "head|relation|tail"
#
# Chiến lược:
#   - "instruction" field: AUGMENTED = original_instruction + KG+RAG evidence
#   - "context"    field: GIỮ NGUYÊN (original sentence)
#   - "response"   field: GIỮ NGUYÊN ground truth "head|RELATION|tail"
#
INFERENCE_CONTEXT_TEMPLATE = "{sentence} --- Knowledge Graph & RAG Evidence --- {fused_context}"
INFERENCE_INSTRUCTION_TEMPLATE = "{instruction}\n\n--- Knowledge Graph Evidence ---\n{fused_context}"

# Instruction mô tả task
INFERENCE_INSTRUCTION = (
    "Please extract the biomedical relation triple from the input sentence. "
    "Use the provided evidence to guide your extraction. "
    "Output format: head_entity|RELATION|tail_entity"
)

# Backward compat alias
FINAL_PROMPT_TEMPLATE = INFERENCE_CONTEXT_TEMPLATE
