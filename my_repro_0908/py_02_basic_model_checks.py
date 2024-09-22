from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model from Hugging Face
# model_name = "meta-llama/Llama-2-7b-hf"  # Change this to your specific LLaMA2 model

# model_name_or_path = "mistralai/Mistral-7B-v0.1"
model_name_or_path = "/mnt/synthdatastore/agoswami/models_04_postlaborday/out_mistral_7b_hf_red_pajama"
# model_name_or_path = "./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_hf"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


# Reload the model with inferred device map for parallel inference
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)


# Move the model to GPUs (if not automatically done)
model = model.to("cuda")

input_texts = [
    "Swiss-born novelist and poet (1887–1961)",
    "Swiss-born novelist and poet (1887–1961)  was a member of the Royal",
    "Swiss-born novelist and poet (1887–1961)  was a member of the Royal Navy"
]

# Input prompt for text generation
for input_text in input_texts:
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    # Generate text using the model
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("Generated Text:\n", generated_text)
    print("="*30)
