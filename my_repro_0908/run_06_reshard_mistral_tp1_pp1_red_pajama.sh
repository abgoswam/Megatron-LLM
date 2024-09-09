# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/out_mistral_7b_tp${TP}_red_pajama \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/out_mistral_7b_reshard_red_pajama \
	--model_type mistral \
	--true_vocab_size 32000 \
	--bf16