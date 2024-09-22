TP=8

# https://github.com/epfLLM/Megatron-LLM/issues/63
python tools/checkpoint_util.py \
	--target_tensor_parallel_size ${tp} \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_phi35_orig1 \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_phi35_orig1_tp${TP} \
	--model_type mistral \
	--true_vocab_size 32064 \
	--bf16