# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir ./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_tp${TP}_save/ \
	--save_dir ./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_reshard/ \
	--model_type mistral \
	--true_vocab_size 32000 \
	--bf16