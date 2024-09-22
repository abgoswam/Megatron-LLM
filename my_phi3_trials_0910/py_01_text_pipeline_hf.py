import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "microsoft/Phi-3-mini-4k-instruct"
# model_name_or_path = "mistralai/Mistral-7B-v0.1"
# model_name_or_path = "/mnt/synthdatastore/agoswami/models_04_postlaborday/out_mistral_7b_hf_red_pajama"
# model_name_or_path = "./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_hf"
# model_name_or_path = "./my_repro_0908/my_repro_ckpts/Mistral-7B-v0.1"

pipeline = transformers.pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(model_name_or_path),
    tokenizer=AutoTokenizer.from_pretrained(model_name_or_path),
    torch_dtype=torch.bfloat16,
    device="cuda")

prompt = """#= a unction that returns the fibonacci number of its argument =#
unction fibonacci(n::Int)::Int
"""

prompt = "Insurrectionary Warfare\nCELT document E900002-006\nChapter 1."
prompt = "Starting with a parade of unarmed men and women to the Palace of the Czar, the flames of insurrection spread all over the land."

sequences = pipeline(
                prompt, 
                max_new_tokens=100, 
                do_sample=True, 
                top_k=20,
                num_return_sequences=1)

for sequence in sequences:
    print(sequence["generated_text"])

print("done")