# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name=mistral \
	--model_size=7 \
	--load=./weights_conversion/out_mistral_7b \
	--data_path=./my_long_corpus_mistral/my_long_corpus_4096_mistral_text_document \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=./weights_conversion/out_mistral_7b/tokenizer.model \
	--huggingface_cache=./weights_conversion/cache_mistral_7b/ \
	--huggingface_device=cuda:1 \
	$COMMON_ARGS $LLAMA_ARGS 

# torchrun $DISTRIBUTED_ARGS verify_correctness.py \
# 	--model_name=llama2 \
# 	--model_size=7 \
# 	--load=/path/to/megatron/weights/ \
# 	--data_path=/path/to/tokenized/starcoder_text_document \  # without the .idx or .bin extension
# 	--tokenizer_type=SentencePieceTokenizer \
# 	--vocab_file=/path/to/megatron/weights/tokenizer.model \
# 	--huggingface_cache=/path/to/meta/llama-2-7b/ \
# 	--huggingface_device=cuda:1 \
# 	$COMMON_ARGS $LLAMA_ARGS  # dont include LLAMA_ARGS if using Falcon

# torchrun $DISTRIBUTED_ARGS verify_correctness.py \
# 	--model_name=mistral \
# 	--model_size=7 \
# 	--load=./weights_conversion/out_mistral_7b \
# 	--data_path=./my_long_corpus_mistral/my_long_corpus_128_mistral_text_document \  # without the .idx or .bin extension
# 	--tokenizer_type=SentencePieceTokenizer \
# 	--vocab_file=./weights_conversion/out_mistral_7b \
# 	--huggingface_cache=./weights_conversion/cache_mistral_7b \
# 	--huggingface_device=cuda:1 \
# 	$COMMON_ARGS $LLAMA_ARGS  # dont include LLAMA_ARGS if using Falcon

