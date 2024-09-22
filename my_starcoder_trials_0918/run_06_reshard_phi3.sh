# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0920/ckpts/out_phi3_orig1_tp8_redpajama \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0920/ckpts/out_phi3_orig1_reshard10000_redpajama \
	--model_type mistral \
	--true_vocab_size 32064 \
	--bf16

# python tools/checkpoint_util.py \
# 	--target_tensor_parallel_size 1 \
# 	--target_pipeline_parallel_size 1 \
# 	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0920/ckpts/out_phi3_orig1_tp8_save \
# 	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0920/ckpts/out_phi3_orig1_reshard10000_save/ \
# 	--model_type mistral \
# 	--true_vocab_size 32064 \
# 	--bf16