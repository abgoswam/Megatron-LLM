from datasets import load_dataset
from transformers import LlamaTokenizer
import json

from datasets import load_dataset

def gopher_rules_pass(sample) -> bool:
    """ function returns True if the sample complies with Gopher rules """
    signals = json.loads(sample["quality_signals"])

    # rule 1: number of words must be greater than 4K
    word_count = signals["rps_doc_word_count"][0][2]
    if word_count < 4_000:
        return False

    # rule 2: mean word length between 3 and 10
    mean_word_length = signals["rps_doc_mean_word_length"][0][2]
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # rule 2: symbol to word ratio below 0.1
    symbol_word_ratio = signals["rps_doc_symbol_to_word_ratio"][0][2]
    if  symbol_word_ratio > 0.1:
        return False

    # rule 3: 90% of lines need to start without a bullet point
    n_lines = signals["ccnet_nlines"][0][2]
    n_lines_bulletpoint_start = sum(map(lambda ln: ln[2], signals["rps_lines_start_with_bulletpoint"]))
    if n_lines_bulletpoint_start / n_lines > 0.9:
        return False

    # rule 4: the ratio between characters in the most frequent 2-gram and the total number 
    # of characters must be below 0.2
    top_2_gram_frac = signals["rps_doc_frac_chars_top_2gram"][0][2]
    if top_2_gram_frac > 0.2:
        return False

    # rule 5: ...


    return True

print("loading...")
ds = load_dataset(
    "togethercomputer/RedPajama-Data-V2", 
    name="default", 
    split="train",
    snapshots=["2023-14"],
    languages=["en"])

#  with streaming
# ds = load_dataset(
#     "togethercomputer/RedPajama-Data-V2", 
#     name="default", 
#     split="train",
#     snapshots=["2023-14"],
#     languages=["en"],
#     streaming=True)
# for sample in ds:
#     if not gopher_rules_pass(sample):
#         continue
#     print(sample)
#     break


print("gopher filtering...")
# filtered_dataset = list(filter(gopher_rules_pass, ds["train"]))
# print("dataset details...")
# print("------"*30)
# print(len(filtered_dataset))
# print("------"*30)
# print(filtered_dataset[0])

filtered_dataset = ds.filter(gopher_rules_pass)
print("dataset details...")
print("------"*30)
print(filtered_dataset.num_rows)
print("------"*30)
print(filtered_dataset)

# Define the old and new column names
old_column_name = 'raw_content'  # Replace with the column you want to rename
new_column_name = 'text'  # Replace with the new column name

# Rename the column and keep only the renamed column
renamed_dataset = filtered_dataset.map(
    lambda x: {new_column_name: x[old_column_name]}, 
    remove_columns=filtered_dataset.column_names)

print(renamed_dataset)

def find_lengths_of_max_min_word_strings(strings):
    # Split each string into words and count the number of words
    word_counts = [(s, len(s.split())) for s in strings]
    
    # Find the string with the maximum number of words
    max_word_string = max(word_counts, key=lambda x: x[1])[0]
    
    # Find the string with the minimum number of words
    min_word_string = min(word_counts, key=lambda x: x[1])[0]
    
    # Get the lengths of the strings with max and min words
    max_length = len(max_word_string)
    min_length = len(min_word_string)
    
    return max_length, min_length

max_length, min_length = find_lengths_of_max_min_word_strings(renamed_dataset["text"])
print(f"Length of string with max words: {max_length}")
print(f"Length of string with min words: {min_length}")

print("saving dataset...")
renamed_dataset.to_json(f"my_red_pajama_v1.jsonl")

print("done")
