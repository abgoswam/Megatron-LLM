from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: Load the model and tokenizer
# model_name = "./Llama-2-7b-hf"  # Example with LLaMA-2 (adjust if using a different variant)
# model_name = "./Meta-Llama-3-8B"  
# model_name = "./Meta-Llama-3.1-8B"
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "microsoft/Phi-3-mini-4k-instruct"
# model_name = "./Mistral-7B-v0.3"
# model_name = "./Mistral-7B-Instruct-v0.2"
# model_name = "./Phi-3-mini-4k-instruct"
# model_name = "./Phi-3.5-mini-instruct"
# model_name = "./out_mistral_7b_orig2_reshard500_hf"
# model_name = "./out_mistral_7b_orig2_reshard5000_hf"
# model_name = "./out_phi3_orig2_reshard5000_hf_MistralForCausalLM"


# Load the tokenizer and model
hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

print(hf_tokenizer.special_tokens_map)