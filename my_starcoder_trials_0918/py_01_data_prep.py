import json
from datasets import load_dataset

# the `cache_dir` argument is optional
dataset = load_dataset("bigcode/starcoderdata", 
                       data_dir="julia",
                       split="train", 
                       cache_dir="./my_starcoder_trials_0918/cache")

with open("/mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/raw.jsonl", "w+") as f:
    for document in dataset:
        document = {
            "id": document["id"], 
            "text": document["content"]
        }
        f.write(json.dumps(document) + "\n")

print("done")