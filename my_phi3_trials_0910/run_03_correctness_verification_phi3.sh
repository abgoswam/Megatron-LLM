# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name=phi3 \
	--model_size=3 \
	--load=./my_phi3_trials_0910/ckpts/out_phi3/ \
	--data_path=./my_long_corpus_4_phi3_trials_0910/my_long_corpus_4_phi3_8192_text_document \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct/tokenizer.model \
	--huggingface_cache=./my_phi3_trials_0910/ckpts/cache_phi3/ \
	--huggingface_device=cuda:1 \
	--split 95,5,0 \
	$COMMON_ARGS $LLAMA_ARGS 
