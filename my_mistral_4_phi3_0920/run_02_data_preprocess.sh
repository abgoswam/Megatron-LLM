python tools/preprocess_data.py \
        --input /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/data/red_pajama_10k.jsonl \
        --output_prefix my_redpajama1_using_combo2 \
        --vocab_file /mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Llama-2-7b-Phi3-hf/tokenizer.model \
        --tokenizer_type SentencePieceTokenizer \
        --workers=2 \
        --dataset_impl=mmap \
        --chunk_size=32 \
        --no_new_tokens