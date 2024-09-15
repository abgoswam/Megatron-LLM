tp=8

# https://github.com/epfLLM/Megatron-LLM/issues/63
python tools/checkpoint_util.py \
	--target_tensor_parallel_size ${tp} \
	--target_pipeline_parallel_size 1 \
	--load_dir ./my_phi3_trials_0910/ckpts/out_mistral_7b/ \
	--save_dir ./my_phi3_trials_0910/ckpts/out_mistral_7b_tp${tp}/ \
	--model_type mistral \
	--true_vocab_size 32000 \
	--bf16