"""
chuck5_generation_api.py
────────────────────────
Phiên bản API của chuck5_generation_8000.py.
Gọi OpenAI-compatible API để sinh relation triple.
"""

import json
import requests
import time


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

# Chỉ để base URL, KHÔNG thêm /chat/completions ở đây
OPENAI_BASE_URL = "https://api.yescale.io/v1"

# Bạn tự điền API key ở đây
OPENAI_API_KEY = ""

OPENAI_MODEL = "gpt-4o"
#OPENAI_MAX_TOKENS = 80


# ═══════════════════════════════════════════════════════════════
# INPUT / OUTPUT FILES
# ═══════════════════════════════════════════════════════════════

number_ = "8000"

#input_test_file_name = "D:/Research 2025-2/Thesis_BioHybridKG/4_generation_triple_model/test_chuck_5_triplet_llama2_13b_right_2.json"
input_test_file_name = "fused_test_input_1_MERGED.json"

#save_file_name = "api_chuck_1_DDI_triplet_%s.json" % number_
#save_file_name = "api_chuck_3_MERGED_triplet_%s.json" % number_

save_file_name = "api_chuck_1m_triplet_%s.json" % number_

# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

def make_inference(instruction, context=None):
    """
    Gọi OpenAI-compatible API để sinh text, dùng prompt format Alpaca.
    """

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Please fill in your API key.")

    if context:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
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

    api_url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        #"max_tokens": OPENAI_MAX_TOKENS,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            #timeout=60,
        )

        if response.status_code != 200:
            print("\n[ERROR] API request failed")
            print(f"[ERROR] URL        : {api_url}")
            print(f"[ERROR] HTTP Code  : {response.status_code}")
            print(f"[ERROR] Body       : {response.text}\n")
            response.raise_for_status()

        data = response.json()

        return prompt + data["choices"][0]["message"]["content"].strip()

    #except requests.exceptions.Timeout:
     #   print("\n[ERROR] API request timeout")
     #   print(f"[ERROR] URL: {api_url}\n")
     #   raise

    except requests.exceptions.RequestException as e:
        print("\n[ERROR] Request failed")
        print(f"[ERROR] URL   : {api_url}")
        print(f"[ERROR] Error : {e}\n")
        raise

    except KeyError as e:
        print("\n[ERROR] Unexpected API response format")
        print(f"[ERROR] Missing key: {e}")
        print(f"[ERROR] Raw response: {data}\n")
        raise

    except json.JSONDecodeError:
        print("\n[ERROR] API response is not valid JSON")
        print(f"[ERROR] Raw response: {response.text}\n")
        raise


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[INFO] Backend : OpenAI-compatible API")
    print(f"[INFO] URL     : {OPENAI_BASE_URL}")
    print(f"[INFO] Model   : {OPENAI_MODEL}")
    print(f"[INFO] Input   : {input_test_file_name}")
    print(f"[INFO] Output  : {save_file_name}")

    import os

    # Load existing progress if file exists
    existing_records = []
    if os.path.exists(save_file_name):
        try:
            with open(save_file_name, "r", encoding="utf-8") as fr_check:
                for line in fr_check:
                    line_str = line.strip()
                    if line_str:
                        try:
                            existing_records.append(json.loads(line_str))
                        except json.JSONDecodeError:
                            print(f"[WARNING] Skipping a malformed line in existing output file.")
            print(f"[INFO] Found existing progress. Loaded {len(existing_records)} records.")
        except Exception as e:
            print(f"[WARNING] Error reading existing file {save_file_name}: {e}. Starting from scratch.")
            existing_records = []

    num_completed = len(existing_records)
    i = num_completed

    with open(save_file_name, "w", encoding="utf-8") as fw:
        # Rewrite existing records to ensure a clean file status
        for rec in existing_records:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fw.flush()

        with open(input_test_file_name, "r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                if idx < num_completed:
                    continue

                line_data = json.loads(line.strip())

                instruction = line_data["instruction"]
                sentence = line_data["context"]
                ground_truth = line_data["response"]

                predicted = None
                while True:
                    try:
                        predicted = make_inference(instruction, sentence)
                        break
                    except KeyboardInterrupt:
                        print("\n[INFO] Interrupted by user. Exiting...")
                        raise
                    except Exception as e:
                        print(f"\n[WARNING] Error at record {idx + 1}: {e}")
                        print("[INFO] Waiting 3 seconds before retrying...")
                        time.sleep(3)

                i += 1
                print(i)

                dic_ = {
                    "sentence": sentence,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                }

                fw.write(json.dumps(dic_, ensure_ascii=False))
                fw.write("\n")
                fw.flush()

    print(f"[DONE] Wrote {i} records (total) to {save_file_name}")