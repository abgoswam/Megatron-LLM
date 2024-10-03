
#!/bin/bash

set -x #This command turns on debugging by making the shell print each command before executing it.
set -e #This command tells the shell to exit immediately if any command it runs exits with a non-zero status (which usually indicates an error)
set -u

# model_path=/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Llama-2-7b-Phi3-hf
# model_path=/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts_helpful_rooster/Phi-3.5-pretrain-llama2/
model_path="/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts_sharing_reptile/out_hf_ckpt/"

output_path=.

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \
    --batch_size 8 \
    --num_fewshot 5