# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name llama2 \
	--load /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_combo2 \
	--data_path /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/data_verify_correctness_combo2/test/my_redpajama1_using_combo2_text_document \
	--tokenizer_type SentencePieceTokenizer \
    --vocab_extra_ids 64 \
	--vocab_file /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Llama-2-7b-Phi3-hf/tokenizer.model \
	--huggingface_cache /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Llama-2-7b-Phi3-hf/ \
	--huggingface_device=cuda:1 \
	$COMMON_ARGS $LLAMA_ARGS 