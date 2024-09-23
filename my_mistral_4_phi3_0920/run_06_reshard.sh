# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_combo2_tp${TP}_save_starcoder \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_combo2_reshard7500_save_starcoder \
	--model_type llama2 \
	--true_vocab_size 32064 \
	--bf16