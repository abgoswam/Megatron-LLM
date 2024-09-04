from datasets import load_dataset
from transformers import LlamaTokenizer
import json

# Load the LLaMA2 tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Load the dataset
dataset = load_dataset("togethercomputer/RedPajama-Data-V2", split="train[:1000]")

# Function to count tokens
def count_tokens(example):
    example['token_count'] = len(tokenizer.encode(example['raw_content']))
    return example

# Apply token counting
dataset = dataset.map(count_tokens)

# Filter out examples with less than 4000 tokens
filtered_dataset = dataset.filter(lambda x: x['token_count'] >= 4000)

# Save the filtered dataset to a JSONL file
with open("filtered_dataset.jsonl", "w") as f:
    for example in filtered_dataset:
        json.dump({"text": example['raw_content']}, f)
        f.write("\n")

print("Filtered dataset saved to filtered_dataset.jsonl")
