from peft import get_peft_model
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,LlamaTokenizer
from transformers import AutoTokenizer
from peft import PeftModel
import sentencepiece
import accelerate
import json

number_="8000"

checkpoint="checkpoint-%s"%number_

# === FUSION PIPELINE INPUT (Bước 8 output từ fusion_module.adapter) ===
# Tạo file này bằng: python -m fusion_module.adapter --full [--backend ollama]
input_test_file_name="fused_test_input.json"


# === Fallback: RAG-only (không có KG), bỏ comment dòng dưới nếu muốn so sánh ===
#input_test_file_name="../2_relation_data_to_triple_train_data/test_chuck_5_triplet_llama2_13b_right_2.json"

save_file_name="chuck_mg_triplet_sim_abaltion_%s.json"%number_



lora_weights="/home/thanhlv/code/NTMP/Thesis_HybridRAG/3_trainning_triple_model/TE_llama2_7b_ourmethod_chuck5_right_0"+"/"+checkpoint  #FTOpenLM-just_ourdata   SFTOpenLM-with_ourdata_lolly   SFTOpenLM-Dolly15k


# lora_config = LoraConfig.from_pretrained(saved_path)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model="/home/thanhlv/code/NTMP/llama-7b-hf/"

tokenizer=AutoTokenizer.from_pretrained(base_model)  #, config=config, cache_dir="./llamacache"


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map='auto')

# model = get_peft_model(model, lora_config)
model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )



def make_inference(instruction, context = None):
  if context:
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
  else:
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: \n{instruction}\n\n### Response: \n"
  inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False,max_length=1500).to("cuda:0")
  # outputs = base_model.generate(**inputs, max_new_tokens=100)
  # display(Markdown((tokenizer.decode(outputs[0], skip_special_tokens=True))))
  # model.zero()
  model.eval()
  with torch.no_grad():
      outputs = model.generate(**inputs, max_new_tokens=50)
      results=(tokenizer.decode(outputs[0], skip_special_tokens=True))
      return results
      # print(results)
      # print("---- NON-INSTRUCT-TUNED-MODEL ----")



if __name__=="__main__":
 
    fw=open(save_file_name,"w")
    i=0
    with open(input_test_file_name,"r",encoding="utf-8") as fr:  #path+"test_chuck_final_ICL_t2.json"
      for line in fr.readlines():
        line=json.loads(line.strip())
        instruction=line["instruction"]
        sentence=line["context"]
        ground_truth=line["response"]
        predicted=make_inference(instruction,sentence)
        i=i+1
        print(i)
        
        Dic_={}
        Dic_["sentence"]=sentence
        Dic_["ground_truth"]=ground_truth
        Dic_["predicted"]=predicted

        fw.write(json.dumps(Dic_))
        fw.flush()
        fw.write("\n")



