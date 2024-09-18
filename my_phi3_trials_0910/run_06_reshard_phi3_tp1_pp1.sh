# resharded model back to tp=1, pp=1

tp=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi3_trials_0917/ckpts/out_phi3_orig2_tp${tp} \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi3_trials_0917/ckpts/out_phi3_orig2_reshard \
	--model_type phi3 \
	--true_vocab_size 32064 \
	--bf16