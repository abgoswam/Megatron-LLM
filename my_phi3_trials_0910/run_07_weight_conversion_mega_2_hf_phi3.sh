# TODO (agoswami): at this point we may need to pass in a tokenizer.model file
# logs show this:
# LlamaTokenizerFast loaded from huggingface
# vocab_file not set, assuming same tokenizer.model used by llama LlamaTokenizerFast

python weights_conversion/megatron_to_hf.py \
    --input_dir=./my_phi3_trials_0910/ckpts/out_phi3_reshard/ \
	--output_dir=./my_phi3_trials_0910/ckpts/out_phi3_hf/

# python weights_conversion/megatron_to_hf.py \
#     --input_dir=./my_phi3_trials_0910/ckpts/out_phi3_reshard/ \
# 	--output_dir=./my_phi3_trials_0910/ckpts/out_phi3_hf/ \
#     --vocab_file=./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct/tokenizer.model \
#     --override_special_tokens "bos=<s>" "eos=<|endoftext|>" "pad=<|endoftext|>" "unk=<unk>"

# python weights_conversion/megatron_to_hf.py \
#     --input_dir=./my_phi3_trials_0910/ckpts/out_phi3_reshard/ \
# 	--output_dir=./my_phi3_trials_0910/ckpts/out_phi3_hf/ \
#     --vocab_file=./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct/tokenizer.model