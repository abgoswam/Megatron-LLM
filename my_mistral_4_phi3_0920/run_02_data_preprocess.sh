python tools/preprocess_data.py \
        --input /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/data/starcoder_julia.jsonl \
        --output_prefix my_starcoder_julia \
        --vocab_file /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Phi-3.5-pretrain/tokenizer.model \
        --tokenizer_type SentencePieceTokenizer \
        --workers=2 \
        --dataset_impl=mmap \
        --chunk_size=32 \
        --no_new_tokens