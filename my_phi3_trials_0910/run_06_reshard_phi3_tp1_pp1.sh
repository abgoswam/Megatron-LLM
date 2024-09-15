# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir ./my_phi3_trials_0910/ckpts/out_phi3_tp${TP}_save/ \
	--save_dir ./my_phi3_trials_0910/ckpts/out_phi3_reshard/ \
	--model_type phi3 \
	--true_vocab_size 32064 \
	--bf16