#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=8
GBZ=4
GPUS_PER_NODE=8

# Your script continues here
echo "TP: $TP, GBZ: $GBZ"

LOG_ARGS="--log_interval 1 --save_interval 500 --eval_interval 100000"
TRAIN_ARGS="--train_iters 10000 --lr_decay_style cosine --lr_warmup_iters 50 --lr 3e-4 --min_lr 1e-6"
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"

# https://github.com/epfLLM/Megatron-LLM/issues/70
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name llama2 \
	--load /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_combo2_tp${TP} \
	--save /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_combo2_tp${TP}_save_starcoder \
	--tensorboard_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_combo2_tp${TP}_save_starcoder/tensorboard/ \
	--data_path /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/data_verify_correctness_combo2/my_starcoder4_using_combo2_text_document \
    --tokenizer_type SentencePieceTokenizer \
    --vocab_extra_ids 64 \
	--vocab_file /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Llama-2-7b-Phi3-hf/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 1 \
	--global_batch_size ${GBZ} \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	--split 970,30,0 \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS