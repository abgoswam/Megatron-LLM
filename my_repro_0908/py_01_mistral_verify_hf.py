import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

pipeline = transformers.pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained("./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_hf/"),
    tokenizer=AutoTokenizer.from_pretrained("./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_hf/"),
    torch_dtype=torch.bfloat16,
    device="cuda"
)

prompt = """#= a function that returns the fibonacci number of its argument =#
function fibonacci(n::Int)::Int
"""
sequences = pipeline(
                prompt, 
                max_new_tokens=100, 
                do_sample=True, 
                top_k=20,
                num_return_sequences=1)

for sequence in sequences:
    print(sequence["generated_text"])

print("done")