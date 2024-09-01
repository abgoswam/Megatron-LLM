python tools/checkpoint_util.py \
	--target_tensor_parallel_size 2 \
	--target_pipeline_parallel_size 1 \
	--load_dir /path/to/megatron/weights/ \
	--save_dir /path/to/sharded/weights/ \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16