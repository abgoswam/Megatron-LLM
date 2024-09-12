from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model from Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"  # Change this to your specific LLaMA2 model
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Reload the model with inferred device map for parallel inference
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)


# Move the model to GPUs (if not automatically done)
model = model.to("cuda")

# Input prompt for text generation
input_text = "Once upon a time in a distant land,"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# Generate text using the model
output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Text:\n", generated_text)
