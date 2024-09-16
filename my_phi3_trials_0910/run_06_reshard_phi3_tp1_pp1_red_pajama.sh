# resharded model back to tp=1, pp=1

TP=8

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/out_phi31_red_pajama/ \
	--save_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/out_phi31_red_pajama_ckpt1500_reshard/ \
	--model_type phi3 \
	--true_vocab_size 32064 \
	--bf16