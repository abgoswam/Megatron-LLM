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
    --size 7 \
    --model-path mistralai/Mistral-7B-v0.1 \
    --out /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_mistral_7b_orig3 \
    mistral
