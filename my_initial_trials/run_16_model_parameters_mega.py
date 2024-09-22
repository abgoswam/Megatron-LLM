import torch

# Replace with the path to your Megatron-LM checkpoint
# checkpoint_path = './weights_conversion/out_llama2_7b/release/mp_rank_00/model_optim_rng.pt'
checkpoint_path = './weights_conversion/out_mistral_7b/release/mp_rank_00/model_optim_rng.pt'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Access the state dictionary
model_state_dict = checkpoint['model']['language_model']['transformer']

# Print the layers and their sizes
for name, param in model_state_dict.items():
    print(f"Layer: {name}, Size: {param.size()}")
