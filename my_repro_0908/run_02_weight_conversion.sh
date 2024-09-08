# llama2
python hf_to_megatron.py \
    --size 7 \
    --out temp_out_llama2_7b \
    --model-path meta-llama/Llama-2-7b-hf \
    --cache-dir temp_cache_llama2_7b \
    llama2

# mistral
python hf_to_megatron.py \
    --size 7 \
    --out out_mistral_7b \
    --model-path mistralai/Mistral-7B-v0.1 \
    --cache-dir cache_mistral_7b \
    mistral
