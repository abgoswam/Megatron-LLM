import torch

# model has "silu" change
loaded = torch.load(
    "/mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_mistral_7b_orig2/release/mp_rank_00/model_optim_rng.pt", 
    map_location="cpu")

print(loaded["args"])


# original model
loaded = torch.load(
    "/mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_phi3_orig2/release/mp_rank_00/model_optim_rng.pt", 
    map_location="cpu")

print(loaded["args"])

print("done")