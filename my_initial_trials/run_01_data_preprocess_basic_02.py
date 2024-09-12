from datasets import load_dataset
from transformers import LlamaTokenizer
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from datasets import load_dataset_builder

ds_builder = load_dataset_builder("rotten_tomatoes")
print(ds_builder.info.description)
print(ds_builder.info.features)
print(ds_builder.info.dataset_size)
print(ds_builder.info.size_in_bytes)

ds_builder = load_dataset_builder("togethercomputer/RedPajama-Data-1T-Sample")
print(ds_builder.info.description)
print(ds_builder.info.features)
print(ds_builder.info.dataset_size)
print(ds_builder.info.size_in_bytes)

ds_builder = load_dataset_builder("togethercomputer/RedPajama-Data-V2", "default")
print(ds_builder.info.description)
print(ds_builder.info.features)
print(ds_builder.info.dataset_size)
print(ds_builder.info.size_in_bytes)

ds_builder = load_dataset_builder("togethercomputer/RedPajama-Data-V2", "sample-1T")
print(ds_builder.info.description)
print(ds_builder.info.features)
print(ds_builder.info.dataset_size)
print(ds_builder.info.size_in_bytes)

ds_builder = load_dataset_builder("togethercomputer/RedPajama-Data-V2", "sample-10B")
print(ds_builder.info.description)
print(ds_builder.info.features)
print(ds_builder.info.dataset_size)
print(ds_builder.info.size_in_bytes)

print("done")
