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

token_count=8192

# https://epfllm.github.io/Megatron-LLM/guide/tokenization.html
python tools/preprocess_data.py \
        --input=./my_repro_0908/my_long_corpus_repro_data/my_long_corpus_${token_count}.jsonl \
        --output_prefix=my_long_corpus_${token_count}_mistral \
        --vocab_file=./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b/tokenizer.model \
        --tokenizer_type=SentencePieceTokenizer \
        --workers=2 \
        --dataset_impl=mmap \
        --chunk_size=32
