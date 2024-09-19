# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_mistral_7b_orig2_tp8_save_starcoder1/ \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_mistral_7b_orig2_reshard5000_save_starcoder1/ \
	--model_type mistral \
	--true_vocab_size 32000 \
	--bf16