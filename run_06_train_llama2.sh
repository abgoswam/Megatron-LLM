LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 500 --lr_decay_style cosine --lr_warmup_iters 50 --lr 3e-4 --min_lr 1e-6"
DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"

# torchrun $DISTRIBUTED_ARGS finetune.py \
# 	--tensor_model_parallel_size 2 \
# 	--pipeline_model_parallel_size 1 \
# 	--load /path/to/sharded/weights/ \
# 	--save /path/to/sharded/weights/ \
# 	--tensorboard_dir /path/to/sharded/weights/tensorboard/ \
# 	--data_path /path/to/tokenized/starcoder_text_document \  # without the .idx or .bin extension
# 	--model_name llama2 \
# 	--tokenizer_type SentencePieceTokenizer \
# 	--vocab_file=/path/to/megatron/weights/tokenizer.model \
# 	--bf16 \
# 	--use_flash_attn \
# 	--micro_batch_size 1 \
# 	--global_batch_size 1000 \
# 	--sequence_parallel \
# 	--recompute_granularity selective \
# 	--use_checkpoint_args \
# 	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS

COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"

torchrun $DISTRIBUTED_ARGS finetune.py \
	--tensor_model_parallel_size 1 \
	--pipeline_model_parallel_size 1 \
	--load ./weights_conversion/out_llama2_7b/ \
	--save ./weights_conversion/out_llama2_7b_save/ \
	--tensorboard_dir ./weights_conversion/out_llama2_7b_save/tensorboard/ \
	--data_path ./my_long_corpus_llama2/my_long_corpus_4096_llama2_text_document \
	--model_name llama2 \
	--tokenizer_type SentencePieceTokenizer \
    --vocab_file=./weights_conversion/out_llama2_7b/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 1 \
	--global_batch_size 1000 \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS