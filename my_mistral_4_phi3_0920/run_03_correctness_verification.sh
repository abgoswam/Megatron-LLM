# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name mistral \
	--model_size 3 \
	--load /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_phi35_orig1 \
	--data_path=./my_repro_0908/my_long_corpus_repro_data/my_long_corpus_8192_mistral_text_document \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Phi-3.5-pretrain//tokenizer.model \
    --vocab_extra_ids 64 \
	--huggingface_cache /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Phi-3.5-pretrain/ \
	--huggingface_device=cuda:1 \
	$COMMON_ARGS $LLAMA_ARGS 