# python3 tools/preprocess_data.py --input ./my_long_corpus_mistral/my_long_corpus_128.jsonl 
#     --output_prefix wiki-train 
#     --dataset_impl mmap 
#     --tokenizer_type FalconTokenizer 
#     --workers 2 
#     --chunk_size 32
#     --append_eod

# https://github.com/microsoft/Megatron-LM/pull/7/files#diff-e7f94618c1e8ba2205553d2195908271f3cee263632060576534dc1d3a937659
# token_count=1048576

# python /workspace/megatron/tools/preprocess_data.py \
#        --input my_long_corpus_${token_count}.jsonl \
#        --output-prefix my_long_corpus_${token_count}_gpt2 \
#        --vocab-file gpt2-vocab.json \
#        --tokenizer-type GPT2BPETokenizer \
#        --merge-file gpt2-merges.txt \
#        --workers 32

token_count=${1}
# token_count=128
# token_count=4096
# token_count=8192
# token_count=16384
# token_count=32768
# token_count=65536
# token_count=131072
# token_count=262144
# token_count=524288
# token_count=1048576

# bash run_02_tokenize.sh 262144 && bash run_02_tokenize.sh 524288 && bash run_02_tokenize.sh 1048576

# https://epfllm.github.io/Megatron-LLM/guide/tokenization.html
python tools/preprocess_data.py \
        --input=./my_long_corpus_mistral/my_long_corpus_${token_count}.jsonl \
        --output_prefix=my_long_corpus_${token_count}_mistral \
        --vocab_file=./weights_conversion/out_mistral_7b/tokenizer.model \
        --tokenizer_type=SentencePieceTokenizer \
        --workers=2 \
        --dataset_impl=mmap \
        --chunk_size=32
