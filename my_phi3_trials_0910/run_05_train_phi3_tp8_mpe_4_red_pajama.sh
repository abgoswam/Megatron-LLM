#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1

S=${1}
TP=8
GBZ=4
GPUS_PER_NODE=8


# Your script continues here
echo "TP: $TP, GBZ: $GBZ"

LOG_ARGS="--log_interval 1 --save_interval 500 --eval_interval 10000"
TRAIN_ARGS="--train_iters 10000 --lr_decay_style cosine --lr_warmup_iters 50 --lr 3e-4 --min_lr 1e-6"
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"

# MY_ARGS_1="--tensor_model_parallel_size 2 --pipeline_model_parallel_size 1"
MY_ARGS_2="--seq_length ${S} --max_position_embeddings ${S}"

# https://github.com/epfLLM/Megatron-LLM/issues/70
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"

torchrun $DISTRIBUTED_ARGS finetune.py \
	--load ./my_phi3_trials_0910/ckpts/out_phi3_tp${TP}/ \
	--save  /mnt/synthdatastore/agoswami/models_04_postlaborday/out_phi31_red_pajama \
	--tensorboard_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/out_phi31_red_pajama/tensorboard/ \
	--data_path ./my_red_pajama_4_phi31_trials_0916/my_red_pajama_4_phi31_text_document \
	--model_name phi3 \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file=./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 1 \
	--global_batch_size ${GBZ} \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	--finetune \
	--split 95,5,0 \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS $MY_ARGS_2