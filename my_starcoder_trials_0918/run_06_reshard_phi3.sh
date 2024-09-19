# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_phi3_orig2_tp8_save_LR0/ \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_phi3_orig2_reshard500_save_LR0/ \
	--model_type mistral \
	--true_vocab_size 32064 \
	--bf16