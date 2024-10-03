#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

deepspeed ./my_mistral_4_phi3_0920/py_eval_passkey_ds.py \
    --max_length 131072 \
    --max_position_embeddings 131072 \
    --model_path meta-llama/Llama-3.1-8B \
    --output_dir /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/passk/Llama-3.1-8B-deepspeed