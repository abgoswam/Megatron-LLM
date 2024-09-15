tp=8

# https://github.com/epfLLM/Megatron-LLM/issues/63
python tools/checkpoint_util.py \
	--target_tensor_parallel_size ${tp} \
	--target_pipeline_parallel_size 1 \
	--load_dir ./my_phi3_trials_0910/ckpts/out_phi3/ \
	--save_dir ./my_phi3_trials_0910/ckpts/out_phi3_tp${tp}/ \
	--model_type phi3 \
	--true_vocab_size 32064 \
	--bf16