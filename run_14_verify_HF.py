import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

pipeline = transformers.pipeline(
    "text-generation",
    model=LlamaForCausalLM.from_pretrained("./weights_conversion/out_llama2_7b_hf/"),
    tokenizer=LlamaTokenizer.from_pretrained("./weights_conversion/out_llama2_7b_hf/"),
    torch_dtype=torch.bfloat16,
    device="cuda"
)

prompt = """#= a function that returns the fibonacci number of its argument =#
function fibonacci(n::Int)::Int
"""
sequences = pipeline(prompt, max_new_tokens=100, do_sample=True, top_k=20,
                     num_return_sequences=1)
for sequence in sequences:
    print(sequence["generated_text"])

print("done")