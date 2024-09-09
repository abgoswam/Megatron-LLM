# TODO (agoswami): at this point we may need to pass in a tokenizer.model file
# logs show this:
# LlamaTokenizerFast loaded from huggingface
# vocab_file not set, assuming same tokenizer.model used by llama LlamaTokenizerFast

python weights_conversion/megatron_to_hf.py \
    --input_dir=./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_reshard/ \
	--output_dir=./my_repro_0908/my_repro_ckpts/repro_out_mistral_7b_hf/