#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

# python tools/preprocess_data.py --input=/path/to/raw.jsonl \
# 	--output_prefix=/path/to/tokenized/starcoder \
# 	--tokenizer_type=SentencePieceTokenizer \
# 	--vocab_file=/path/to/tokenizer.model \
# 	--chunk_size=32 \
# 	--workers=16 \
# 	--no_new_tokens

# python tools/preprocess_data.py \
#         --input=/path/to/data.json \
#         --output_prefix=wiki-train \
#         --dataset_impl=mmap \
#         --tokenizer_type=SentencePieceTokenizer \
#         --vocab_file=/path/to/tokenizer.model \
#         --workers=2 \
#         --chunk_size=32

# bash ./my_phi3_trials_0910/run_02_data_preprocess_phi3.sh 16384 && bash ./my_phi3_trials_0910/run_02_data_preprocess_phi3.sh 32768 && bash ./my_phi3_trials_0910/run_02_data_preprocess_phi3.sh 65536 && bash ./my_phi3_trials_0910/run_02_data_preprocess_phi3.sh 131072

# token_count=${1}

# https://epfllm.github.io/Megatron-LLM/guide/tokenization.html
python tools/preprocess_data.py \
        --input=./my_red_pajama_4_phi31_trials_0916/data_10k.jsonl \
        --output_prefix=my_red_pajama_4_phi31 \
        --vocab_file=./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct/tokenizer.model \
        --tokenizer_type=SentencePieceTokenizer \
        --workers=2 \
        --dataset_impl=mmap \
        --chunk_size=32
