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

print("loading..")
ds_iterator = load_dataset(
    "togethercomputer/RedPajama-Data-V2",
    snapshots=["2023-14"],
    languages=["en"],
    name="default",
    streaming=True
)

filtered_dataset = []

for sample in ds_iterator["train"]:
    if not gopher_rules_pass(sample):
        continue

    filtered_dataset.append(sample)
    print(filtered_dataset)
    break

print("done")