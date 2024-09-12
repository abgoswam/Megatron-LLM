# python weights_conversion/hf_to_megatron.py \
#     llama2 \
#     --size=7 \
# 	--out=/path/to/megatron/weights/ \
#     --cache-dir=/path/to/llama-2-7b/

# # llama2
# python hf_to_megatron.py \
#     --size 7 \
#     --out temp_out_llama2_7b \
#     --model-path meta-llama/Llama-2-7b-hf \
#     --cache-dir temp_cache_llama2_7b \
#     llama2

# mistral
python ./weights_conversion/hf_to_megatron.py \
    --size 3 \
    --out ./my_phi3_trials_0910/ckpts/out_phi3 \
    --model-path microsoft/Phi-3-mini-4k-instruct \
    --cache-dir ./my_phi3_trials_0910/ckpts/cache_phi3 \
    phi3