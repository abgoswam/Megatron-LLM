TP=8

# https://github.com/epfLLM/Megatron-LLM/issues/63
python tools/checkpoint_util.py \
	--target_tensor_parallel_size ${TP} \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_combo2 \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_combo2_tp${TP} \
	--model_type llama2 \
	--true_vocab_size 32064 \
	--bf16