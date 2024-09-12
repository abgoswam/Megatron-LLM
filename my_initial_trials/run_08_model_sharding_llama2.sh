tp=${1}

python tools/checkpoint_util.py \
	--target_tensor_parallel_size ${tp} \
	--target_pipeline_parallel_size 1 \
	--load_dir ./weights_conversion/out_llama2_7b/ \
	--save_dir ./weights_conversion/out_llama2_7b_tp${tp}/ \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16