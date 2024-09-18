#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

python weights_conversion/megatron_to_hf.py \
    --input_dir=/mnt/synthdatastore/agoswami/models_04_postlaborday/out_phi31_red_pajama_ckpt7500_reshard/ \
	--output_dir=/mnt/synthdatastore/agoswami/models_04_postlaborday/out_phi31_red_pajama_ckpt7500_reshard_hf_fixed/ \
    --model phi3 \
    --vocab_extra_ids_list "<|endoftext|>" \
    --override_special_tokens "pad=<|endoftext|>"