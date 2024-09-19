# TODO (agoswami): at this point we may need to pass in a tokenizer.model file
# logs show this:
# LlamaTokenizerFast loaded from huggingface
# vocab_file not set, assuming same tokenizer.model used by llama LlamaTokenizerFast

python weights_conversion/megatron_to_hf.py \
    --input_dir=/mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_phi3_orig2_reshard5000_save_starcoder1/ \
	--output_dir=/mnt/synthdatastore/agoswami/models_04_postlaborday/my_starcoder_trials_0918/ckpts/out_phi3_orig2_reshard5000_hf_Phi3ForCausalLM \
    --model phi3 \
    --vocab_extra_ids_list "<|endoftext|>" \
    --override_special_tokens "pad=<|endoftext|>" # need to re-insert because it got overwritten L686
