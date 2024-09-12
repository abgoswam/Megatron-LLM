from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify the model name or path
model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Tokenize a simple sentence
sentence = "How are you today?"
inputs = tokenizer(sentence, return_tensors="pt")

# Move inputs to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}
print("Tokenized input:", inputs)

# Display the model layers along with their sizes and shapes
print("\nModel layers and their shapes:")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}")

# Generate response
output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,
    num_return_sequences=1
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Response:")
print(generated_text)
