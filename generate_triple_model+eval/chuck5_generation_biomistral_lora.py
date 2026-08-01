"""
chuck5_generation_biomistral_lora.py
──────────────────────────────────────
Generation script dùng BioMistral-7B + LoRA adapter đã fine-tune.
Dựa trên chuck5_generation_8000.py (LLaMA-7B), điều chỉnh cho BioMistral-7B.

Thay đổi so với gốc:
  1. base_model trỏ đến BioMistral-7B
  2. lora_weights trỏ đến checkpoint train trên BioMistral
  3. Bỏ LlamaTokenizer → AutoTokenizer
  4. pad_token = eos_token (không add [PAD] token mới)
  5. Thêm do_sample=False, pad_token_id để generation ổn định
  6. Resume checkpoint + flush (từ chuck5_generation_api.py)

⚠️  LoRA weights từ LLaMA-7B KHÔNG tương thích — phải dùng adapter
    được train bởi trainer_biomistral.py (trong thư mục này).
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

number_    = "8000"
checkpoint = "checkpoint-%s" % number_

# ── BioMistral-7B base model ─────────────────────────────────
BASE_MODEL = "/home/thanhlv/code/NTMP/BioMistral-7B/"

# ── LoRA adapter train trên BioMistral (từ trainer_biomistral.py) ──
LORA_WEIGHTS = (
    "/home/thanhlv/code/NTMP/Thesis_HybridRAG/3_trainning_triple_model/"
    "TE_biomistral_7b_ourmethod_chuck5_right_0"
    "/" + checkpoint
)

# ── Input / Output ───────────────────────────────────────────────
input_test_file_name = "fused_test_input.json"
save_file_name       = "chuck_biomistral_lora_triplet_%s.json" % number_

# ── Inference settings ───────────────────────────────────────────
MAX_NEW_TOKENS   = 50
MAX_INPUT_LENGTH = 1500


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"[INFO] Loading tokenizer from: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

# Mistral không có pad_token riêng → dùng eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"[INFO] Loading base model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto",
)

print(f"[INFO] Loading LoRA adapter: {LORA_WEIGHTS}")
model = PeftModel.from_pretrained(
    model,
    LORA_WEIGHTS,
    torch_dtype=torch.float16,
)

model.eval()
print("[INFO] Model ready.\n")


# ═══════════════════════════════════════════════════════════════
# CHECKPOINT HELPER (resume khi script bị interrupt)
# ═══════════════════════════════════════════════════════════════

def count_completed_records(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

def make_inference(instruction: str, context: str = None) -> str:
    """
    Sinh triple từ instruction + context dùng BioMistral-7B + LoRA.
    Dùng Alpaca prompt format (giống file gốc để kết quả so sánh được).
    """
    if context:
        prompt = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context.\n\n"
            f"### Instruction: \n{instruction}\n\n"
            f"### Input: \n{context}\n\n"
            "### Response: \n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction: \n{instruction}\n\n"
            "### Response: \n"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    ).to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                         # greedy – deterministic
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[INFO] Backend      : BioMistral-7B + LoRA (local)")
    print(f"[INFO] Base model   : {BASE_MODEL}")
    print(f"[INFO] LoRA weights : {LORA_WEIGHTS}")
    print(f"[INFO] Input        : {input_test_file_name}")
    print(f"[INFO] Output       : {save_file_name}")
    print(f"[INFO] Max new tokens: {MAX_NEW_TOKENS}")

    # ── Resume từ checkpoint ─────────────────────────────────────
    skip = count_completed_records(save_file_name)
    if skip > 0:
        print(f"[INFO] Resuming from record {skip + 1} (skipping {skip} already done)")
    else:
        print(f"[INFO] Starting fresh")

    i       = 0
    written = 0

    with open(save_file_name, "a", encoding="utf-8") as fw:
        with open(input_test_file_name, "r", encoding="utf-8") as fr:
            for line in fr:
                line = json.loads(line.strip())
                i += 1

                if i <= skip:
                    continue

                instruction  = line["instruction"]
                sentence     = line["context"]
                ground_truth = line["response"]

                predicted = make_inference(instruction, sentence)

                written += 1
                print(f"[OK] Record {i} done  (session total: {written})")

                dic_ = {
                    "sentence":     sentence,
                    "ground_truth": ground_truth,
                    "predicted":    predicted,
                }

                fw.write(json.dumps(dic_, ensure_ascii=False))
                fw.write("\n")
                fw.flush()

    total_done = skip + written
    print(f"\n[DONE] Session wrote {written} new records.")
    print(f"[DONE] Total records in {save_file_name}: {total_done}")
