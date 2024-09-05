from datasets import load_dataset
from transformers import LlamaTokenizer
import json
from tqdm import tqdm
from datasets import load_dataset

def write_dict_array_to_jsonl(data, filename: str) -> None:
    """Write an array of dictionaries to a jsonl file."""
    with open(filename, "w", encoding='utf-8') as f:
        for d in tqdm(data, desc="writing to file"):
            json_string = json.dumps(d, ensure_ascii=False)
            f.write(json_string + "\n")

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

# no streaming
# ds = load_dataset(
#     "togethercomputer/RedPajama-Data-V2", 
#     name="default", 
#     split="train",
#     snapshots=["2023-14"],
#     languages=["en"])

#  with streaming
ds = load_dataset(
    "togethercomputer/RedPajama-Data-V2", 
    name="default", 
    split="train",
    snapshots=["2023-14"],
    languages=["en"],
    streaming=True)

data_samples = []
id = 0

for sample in tqdm(ds, desc="processing data"):
    if not gopher_rules_pass(sample):
        continue

    # print(sample)

    d = {
        "id": id,
        "text": sample["raw_content"]
    }
    
    data_samples.append(d)
    id += 1

    if id %1000 == 0:
        print(f"id: {id}")

    if id > 10000:
        break

write_dict_array_to_jsonl(data_samples, "./my_red_pajama/data_10k.jsonl")

print("done")
