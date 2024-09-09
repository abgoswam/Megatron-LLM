tp=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size ${tp} \
	--target_pipeline_parallel_size 1 \
	--load_dir ./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b/ \
	--save_dir ./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_tp${tp}/ \
	--model_type mistral \
	--true_vocab_size 32000 \
	--bf16