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

# token_count=8192

# https://epfllm.github.io/Megatron-LLM/guide/tokenization.html
python tools/preprocess_data.py \
        --input=/mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/raw.jsonl \
        --output_prefix=my_starcoder_phi3 \
        --vocab_file=/tmp/amlt-code-download/abgoswam-epf2/my_starcoder_trials_0918/ckpts/Phi-3-mini-4k-instruct/tokenizer.model \
        --tokenizer_type=SentencePieceTokenizer \
        --workers=2 \
        --dataset_impl=mmap \
        --chunk_size=32 \
        --no_new_tokens
