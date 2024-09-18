tp=8

# https://github.com/epfLLM/Megatron-LLM/issues/63
python tools/checkpoint_util.py \
	--target_tensor_parallel_size ${tp} \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_mistral_7b_orig2 \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_mistral_7b_orig2_tp${tp} \
	--model_type mistral \
	--true_vocab_size 32000 \
	--bf16