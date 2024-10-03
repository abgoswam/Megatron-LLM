python tools/preprocess_data.py \
        --input /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/data/cosmopedia_v2.jsonl \
        --output_prefix agoswami_cosmopedia_v2 \
        --vocab_file /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Llama-2-7b-Phi3-hf/tokenizer.model \
        --tokenizer_type SentencePieceTokenizer \
        --workers=2 \
        --dataset_impl=mmap \
        --chunk_size=32 \
        --no_new_tokens