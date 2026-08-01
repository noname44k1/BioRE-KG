import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset


# ═══════════════════════════════════════════════════════════════
# Prompt formatting (Alpaca format – giống gốc)
# ═══════════════════════════════════════════════════════════════

def formatting_func(example):
    if example.get("context", "") != "":
        input_prompt = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input: \n"
            f"{example['context']}\n\n"
            "### Response: \n"
            f"{example['response']}"
        )
    else:
        input_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:\n"
            f"{example['response']}"
        )
    return {"text": input_prompt}


def prepare_data(path):
    data = load_dataset("json", data_files=path)
    formatted_data = data.map(formatting_func)
    return formatted_data["train"]


# ═══════════════════════════════════════════════════════════════
# CONFIG – chỉnh đường dẫn phù hợp với máy chủ
# ═══════════════════════════════════════════════════════════════

train_path = "train_chunk_triplet_final.json"
test_path  = "test_chunk_triplet_final.json"

train = prepare_data(train_path)
test  = prepare_data(test_path)


# Change to your model path
model_id = ""




# ═══════════════════════════════════════════════════════════════
# LoRA Config
# ═══════════════════════════════════════════════════════════════
# Giữ nguyên r=64, lora_alpha=32 như gốc.
# target_modules: BioMistral dùng Mistral attention, tên layer giống nhau
# (q_proj, k_proj, v_proj, o_proj) nhưng shape k/v KHÁC LLaMA do GQA.
# SFTTrainer + PEFT tự xử lý đúng shape khi train từ đầu.

qlora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    # BioMistral/Mistral-7B attention layer names:
    # peft mới (>= 0.4) KHÔNG tự detect → phải chỉ rõ.
    # q_proj + v_proj: standard QLoRA (tiết kiệm VRAM nhất)
    # Thêm k_proj, o_proj nếu muốn coverage rộng hơn
    target_modules=["q_proj", "v_proj"],
)


# ═══════════════════════════════════════════════════════════════
# Quantization Config (4-bit NF4, giống gốc)
# ═══════════════════════════════════════════════════════════════

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ═══════════════════════════════════════════════════════════════
# Load Model & Tokenizer
# ═══════════════════════════════════════════════════════════════

print(f"[INFO] Loading BioMistral-7B from: {model_id}")

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

# AutoTokenizer thay vì LlamaTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# BioMistral/Mistral không có [PAD] → dùng eos_token làm pad_token
# (KHÔNG dùng add_special_tokens({'pad_token': '[PAD]'}) như LLaMA
#  vì sẽ resize embedding và làm thay đổi model architecture)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# SFTTrainer yêu cầu padding_side='right' cho causal LM training (tránh overflow half-precision)
tokenizer.padding_side = "right"

print(f"[INFO] Tokenizer pad_token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")


# ═══════════════════════════════════════════════════════════════
# SFT Training
# ═══════════════════════════════════════════════════════════════

from trl import SFTTrainer, SFTConfig

supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=train,
    eval_dataset=test,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        max_steps=10000,
        save_steps=1000,
        max_grad_norm=0.1,
        warmup_ratio=0.03,
        output_dir="TE_biomistral_7b_ourmethod_chuck5_right_0",
        optim="adamw_torch",   # 32-bit Adam chuẩn – tránh bug 8-bit optimizer của bitsandbytes < 0.41.1
        fp16=True,
        dataset_text_field="text",   # ← chuyển vào SFTConfig (không dùng deprecated arg)
        max_seq_length=1024,         # ← chuyển vào SFTConfig
    ),
    tokenizer=tokenizer,
    peft_config=qlora_config,
)

supervised_finetuning_trainer.train()
