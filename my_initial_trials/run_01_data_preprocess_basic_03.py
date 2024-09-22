from datasets import load_dataset
from transformers import LlamaTokenizer
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from datasets import load_dataset_builder

import requests
from pprint import pprint

API_TOKEN = "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"

headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://datasets-server.huggingface.co/size?dataset=ibm/duorc"
# API_URL = "https://datasets-server.huggingface.co/size?dataset=cornell-movie-review-data/rotten_tomatoes"
API_URL = "https://datasets-server.huggingface.co/size?dataset=togethercomputer/RedPajama-Data-V2"

def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()

data = query()
pprint(data)

print("done")
