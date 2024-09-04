# job: https://ml.azure.com/runs/busy_pin_zs7rqpx4h3?wsid=/subscriptions/584bfbd1-d24e-4f7b-81cf-c953a75c45e5/resourcegroups/genai-postraining-RG/workspaces/genai-posttraining-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47&monitoringView=%7B%22options%22:%7B%22GpuMemoryUtilizationMegabytes%22:%5B%22GpuMemoryUtilizationMegabytes/node-0/4%22,%22GpuMemoryUtilizationMegabytes/node-0/5%22,%22GpuMemoryUtilizationMegabytes/node-0/7%22,%22GpuMemoryUtilizationMegabytes/node-0/0%22,%22GpuMemoryUtilizationMegabytes/node-0/2%22,%22GpuMemoryUtilizationMegabytes/node-0/3%22,%22GpuMemoryUtilizationMegabytes/node-0/1%22,%22GpuMemoryUtilizationMegabytes/node-0/6%22%5D%7D%7D#

# deepspeed eval_passkey.py \
    # --use_flash_attention ${{inputs.use_flash_attention}} \
    # --max_length ${{inputs.max_length}} \
    # --trials ${{inputs.trials}} \
    # --seed ${{inputs.seed}} \
    # --max_position_embeddings ${{inputs.max_position_embeddings}} \
    # --model_path ${{inputs.model_path}} \
    # --output_dir ${{outputs.output_dir}}

# llama2
# python run_15_eval_passkey.py \
#     --max_length 131072 \
#     --max_position_embeddings 131072 \
#     --model_path meta-llama/Llama-2-7b-hf \
#     --output_dir ./my_long_context_eval_passkey_mistral

# mistral
python run_15_eval_passkey.py \
    --max_length 131072 \
    --max_position_embeddings 131072 \
    --model_path mistralai/Mistral-7B-v0.1 \
    --output_dir ./my_long_context_eval_passkey_mistral
