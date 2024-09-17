#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

python weights_conversion/megatron_to_hf.py \
    --input_dir=/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi3_trials_0910/ckpts/out_phi3_orig2 \
	--output_dir=/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi3_trials_0910/ckpts/out_phi3_orig2_hf \
    --model phi3