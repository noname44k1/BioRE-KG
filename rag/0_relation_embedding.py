
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np

# Change to your model path
model_id = '' 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, torch_dtype=torch.float16, device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_id)


all_vectors=[]
with open("relation_with_defination.txt", "r") as fr:
    i = -1
    for line in fr.readlines():
        i = i + 1
        line = line.strip().split("\t")[1]

        input_ids = tokenizer.encode(line, return_tensors='pt').to("cuda")
        output = model(input_ids, output_hidden_states=True)  # <= set output_hidden_states to True
        hidden_states = output.hidden_states  # all the hidden_states are collected in this tuple
        input_embeddings = hidden_states[0]  # get the input embeddings
        input_embeddings=torch.mean(input_embeddings.squeeze(0),dim=0)

        final_embeddings=input_embeddings.detach().cpu().numpy()
        all_vectors.append(final_embeddings)

np.save("relation_description.npy", np.array(all_vectors))

