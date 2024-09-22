# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name=mistral \
	--model_size=7 \
	--load=./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b/ \
	--data_path=./my_repro_0908/my_long_corpus_repro_data/my_long_corpus_8192_mistral_text_document \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b/tokenizer.model \
	--huggingface_cache=./my_repro_0908/my_repro_ckpts/repro_cache_2_mistral_7b \
	--huggingface_device=cuda:1 \
	--split 95,5,0 \
	$COMMON_ARGS $LLAMA_ARGS 
