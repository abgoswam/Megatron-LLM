# Fail if no positional parameters are passed
set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=${1}
GBZ=32
GPUS_PER_NODE=8

# Your script continues here
echo "TP: $TP, GBZ: $GBZ"

LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 500 --lr_decay_style cosine --lr_warmup_iters 50 --lr 3e-4 --min_lr 1e-6"
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"

# MY_ARGS_1="--tensor_model_parallel_size 2 --pipeline_model_parallel_size 1"
# MY_ARGS_2="--seq_length 1024 --max_position_embeddings 1024"

# https://github.com/epfLLM/Megatron-LLM/issues/70
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"

torchrun $DISTRIBUTED_ARGS finetune.py \
	--load ./weights_conversion/out_llama2_7b_tp${TP}/ \
	--save ./weights_conversion/out_llama2_7b_save/ \
	--tensorboard_dir ./weights_conversion/out_llama2_7b_save/tensorboard/ \
	--data_path ./my_long_corpus_llama2/my_long_corpus_4096_llama2_text_document \
	--model_name llama2 \
	--tokenizer_type SentencePieceTokenizer \
    --vocab_file=./weights_conversion/out_llama2_7b/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 1 \
	--global_batch_size ${GBZ} \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS
