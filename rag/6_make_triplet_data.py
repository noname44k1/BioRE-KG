import json

def convert_topn_to_right_one(input_file, output_file, relation_tuple_str):
    """
    Reads the output of 3_make_topn_for_train_test_5.py and generates 
    the ..._llama2_13b_right_one.json format directly, bypassing the need for an LLM.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                data = json.loads(line)
                
                # Format the top-n chunks into Examples string
                examples_list = []
                topn = data.get("topn_sim_chuck", {})
                
                # Take up to 5 examples
                for i, (chunk, rel) in enumerate(topn.items()):
                    if i >= 5: break
                    examples_list.append(f"Context: {chunk} . Response: {rel}")
                
                examples_str = " ".join(examples_list)
                
                # Base instruction template
                instruction = (
                    f"You are an excellent linguist. The task is to predict the most relevant relation, "
                    f"which must be in {relation_tuple_str}, for the given sentence.  "
                    f"Examples:  {examples_str}"
                )
                
                # The context in the original right_one.json often had spaces around entities, 
                # but the simplest way is to just use the original sentence.
                # If we want to strictly mimic it, we could replace the head/tail with padded versions,
                # but simply passing the sentence is sufficient since the pipeline only extracts the instruction anyway!
                context = data.get("sentence", "")
                
                out_obj = {
                    "instruction": instruction,
                    "context": context,
                    "response": data.get("relation", ""),
                    "category": "relation_extraction"
                }
                
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                
        print(f"Successfully created: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please ensure you have run stage 0 completely.")

if __name__ == "__main__":
    # DDI relations tuple
    ddi_tuple = "('mechanism', 'effect', 'advise', 'int', 'None')"
    
    # Paths (adjust if your topn files are in a different folder)
    train_in = "train_data_topn_chunks.json"
    train_out = "train_chunk_s_r_triplet.json"
    
    test_in = "test_data_topn_chunks.json"
    test_out = "test_chunk_s_r_triplet.json"
    
    convert_topn_to_right_one(train_in, train_out, ddi_tuple)
    convert_topn_to_right_one(test_in, test_out, ddi_tuple)
