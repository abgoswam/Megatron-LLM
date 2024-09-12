python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir ./weights_conversion/out_mistral_7b_save/ \
	--save_dir ./weights_conversion/out_mistral_7b_reshard \
	--model_type mistral \
	--true_vocab_size 32000 \
	--bf16