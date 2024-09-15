import torch
import sys

sys.path.append('/tmp/amlt-code-download/abgoswam_epf')
print('\n'.join(sys.path))

# Replace with the path to your Megatron-LM checkpoint
# checkpoint_path = './weights_conversion/out_llama2_7b/release/mp_rank_00/model_optim_rng.pt'
# checkpoint_path = './weights_conversion/out_mistral_7b/release/mp_rank_00/model_optim_rng.pt'
# checkpoint_path = '/tmp/amlt-code-download/abgoswam_epf/my_phi3_trials_0910/ckpts/out_phi3/release/mp_rank_00/model_optim_rng.pt'
# # checkpoint_path = '/tmp/amlt-code-download/abgoswam_epf/my_phi3_trials_0910/ckpts/out_mistral_7b/release/mp_rank_00/model_optim_rng.pt'

# # Load the checkpoint
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# # Access the state dictionary
# embed_state_dict = checkpoint['model']['language_model']['embedding']

# # Print the layers and their sizes
# for name, param in embed_state_dict.items():
#     print(f"Layer: {name}, Size: {param.size()}")

# # Access the state dictionary
# model_state_dict = checkpoint['model']['language_model']['transformer']

# # Print the layers and their sizes
# for name, param in model_state_dict.items():
#     print(f"Layer: {name}, Size: {param.size()}")

# ==================== resharded ckpt seem sto have a slightly different model architecture ===================
checkpoint_path = '/tmp/amlt-code-download/abgoswam_epf/my_phi3_trials_0910/ckpts/out_phi3_reshard/iter_0000020/mp_rank_00/model_optim_rng.pt'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Access the state dictionary
embed_state_dict = checkpoint['model']['language_model']['embedding']

# Print the layers and their sizes
for name, param in embed_state_dict.items():
    print(f"Layer: {name}, Size: {param['weight'].size()}")

# Access the state dictionary
model_state_dict = checkpoint['model']['language_model']['encoder']

# Print the layers and their sizes
for name, param in model_state_dict.items():
    print(f"Layer: {name}, Size: {param.size()}")
