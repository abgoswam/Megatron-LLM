from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: Load the model and tokenizer
# model_name_or_path = "./Llama-2-7b-hf"  # Example with LLaMA-2 (adjust if using a different variant)
# model_name_or_path = "./Meta-Llama-3-8B"  
# model_name_or_path = "./Meta-Llama-3.1-8B"
# model_name_or_path = "./Mistral-7B-v0.1"
# model_name_or_path = "./Mistral-7B-v0.3"
# model_name_or_path = "./Mistral-7B-Instruct-v0.2"
# model_name_or_path = "./Phi-3-mini-4k-instruct"
# model_name_or_path = "./Phi-3.5-mini-instruct"
# model_name_or_path = "./weights_conversion/out_llama2_7b_hf"
# model_name_or_path = "microsoft/Phi-3-small-8k-instruct"
model_name_or_path = "microsoft/Phi-3-mini-4k-instruct"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

# Step 2: Tokenize a simple sentence
sentence = "How are you today?"
inputs = tokenizer(sentence, return_tensors="pt")
print("Tokenized input:", inputs)

# Step 3: Display the model layers along with their sizes and shapes
print("\nModel layers and their shapes:")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}")

# Step 4: Use the above sentence to generate responses from the model
# Generate response
output = model.generate(
    inputs["input_ids"], 
    attention_mask=inputs["attention_mask"],
    max_length=50, 
    num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Response:")
print(generated_text)
