# from datasets import load_dataset
import torch
import transformers
# pyrefly: ignore [missing-import]
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import LlamaTokenizer
from trl import SFTTrainer
from datasets import load_dataset


def formatting_func(example):
  if example.get("context", "") != "":
      input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response: \n"
      f"{example['response']}")

  else:
    input_prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Response:\n"
      f"{example['response']}")

  return {"text" : input_prompt}

def prepare_data(path):
    data = load_dataset("json", data_files=path)
    formatted_data = data.map(formatting_func)
    # print( formatted_data["train"])
    return formatted_data["train"]



train_path="train_chunk_triplet_final.json"
test_path="test_chunk_triplet_final.json"
train = prepare_data(train_path)
test = prepare_data(test_path)


# Change to your model path
model_id = ""


qlora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,torch_dtype=torch.float16, device_map='auto',
)



tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "right"  # SFTTrainer yêu cầu padding_side='right' để tránh overflow và chạy nhanh hơn



supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=train,
    eval_dataset=test,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        max_steps=10000,
        save_steps=1000,
        max_grad_norm=0.1,
        warmup_ratio=0.03,
        output_dir="TE_llama2_7b_ourmethod_chuck5_right_0",
        optim="paged_adamw_8bit",
        fp16=True,
        gradient_checkpointing=False,  # peft 0.4.0 tự bật → chậm 2x, tắt để về tốc độ gốc
    ),
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="text",
    max_seq_length=512
)

supervised_finetuning_trainer.train()
