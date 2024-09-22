# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_phi35_orig1_tp${TP}_save_redpajama1 \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_phi35_orig1_reshardckpt10000_save_redpajama1 \
	--model_type mistral \
	--true_vocab_size 32064 \
	--bf16