from datasets import load_dataset
from transformers import LlamaTokenizer
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

dataset = load_dataset("glue", "mrpc", split="train")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

dataset = dataset.map(encode, batched=True)

dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

print(dataset[0])

dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

print("done")

# # Load the LLaMA2 tokenizer
# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# # Load the dataset
# dataset = load_dataset("togethercomputer/RedPajama-Data-V2", "default", split="train")

# # Function to count tokens
# def count_tokens(example):
#     example['token_count'] = len(tokenizer.encode(example['raw_content']))
#     return example

# # Apply token counting
# dataset = dataset.map(count_tokens)

# # Filter out examples with less than 4000 tokens
# filtered_dataset = dataset.filter(lambda x: x['token_count'] >= 4000)

# # Save the filtered dataset to a JSONL file
# with open("filtered_dataset.jsonl", "w") as f:
#     for example in filtered_dataset:
#         json.dump({"text": example['raw_content']}, f)
#         f.write("\n")

# print("Filtered dataset saved to filtered_dataset.jsonl")
