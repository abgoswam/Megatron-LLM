from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: Load the model and tokenizer
# model_name = "./Llama-2-7b-hf"  # Example with LLaMA-2 (adjust if using a different variant)
# model_name = "./Meta-Llama-3-8B"  
# model_name = "./Meta-Llama-3.1-8B"
# model_name = "./Mistral-7B-v0.1"
# model_name = "./Mistral-7B-v0.3"
# model_name = "microsoft/Phi-3-mini-4k-instruct"
# model_name = "./out_phi3_orig2_reshard5000_hf_MistralForCausalLM"
# model_name = "./out_phi3_orig2_reshard5000_hf_Phi3ForCausalLM"
# model_name = "./out_phi3_orig1_reshard3000_save_hf/"
# model_name = "./out_phi3_orig1_reshard3000_redpajama_hf/"
# model_name = "./Phi-3_1-mling-mini-pretrain-1_0-3_8b-phase3-140000-full-mistral-20240712/"
# model_name = "./Phi-3.5-pretrain"
# model_name = "/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Phi-3.5-pretrain/"
# model_name = "/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_phi35_orig1_reshardckpt10000_save_starcoder1_hf"
model_name = "/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_phi35_orig1_reshardckpt10000_save_redpajama1_hf"


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\nModel layers and their shapes:")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}")

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 2: Tokenize a simple sentence
input_texts = [
    "tell me a fun fact:",
    "whats your favourite programming language",
    "write some code:",
    "function add(x, y)",
    "Swiss-born novelist and poet (1887–1961)",
    "Swiss-born novelist and poet (1887–1961)  was a member of the Royal",
    "Swiss-born novelist and poet (1887–1961)  was a member of the Royal Navy", 
]

snip1 = """
using StatsBase

function Statistics.mean(A::KeyedArray, wv::AbstractWeights; dims=:, kwargs...)
    dims === Colon() && return mean(parent(A), wv; kwargs...)
    numerical_dims = NamedDims.dim(A, dims)
"""

input_texts.append(snip1)

# Input prompt for text generation
for sentence in input_texts:
    # Step 2: Tokenize a simple sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print("Tokenized input:", inputs)

    # Step 4: Use the above sentence to generate responses from the model
    # Generate response
    output = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_length=128, 
        num_return_sequences=1)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\nGenerated Response:")
    print(generated_text)
    print("="*30)


