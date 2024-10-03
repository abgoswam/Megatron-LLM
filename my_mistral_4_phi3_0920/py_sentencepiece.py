import sys

sys.path.append('/tmp/amlt-code-download/abgoswam_epf')
print('\n'.join(sys.path))

from megatron.tokenizer.tokenizer import _SentencePieceTokenizer

# vocab_file = "/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/Phi-3.5-pretrain/tokenizer.model"
vocab_file = "/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/ckpts/out_mistral7b_orig1/tokenizer.model"

# mt_tokenizer = _SentencePieceTokenizer(vocab_file)
# mt_tokenizer = _SentencePieceTokenizer(vocab_file, new_tokens=False)
mt_tokenizer = _SentencePieceTokenizer(vocab_file, vocab_extra_ids_list="<|endoftext|>", new_tokens=False)

print(f"mt_tokenizer.vocab_size: {mt_tokenizer.vocab_size}")

print(f"cls: {mt_tokenizer.cls}")
print(f"sep: {mt_tokenizer.sep}")
print(f"eod: {mt_tokenizer.eod}")
print(f"mask: {mt_tokenizer.mask}")
print(f"pad: {mt_tokenizer.pad}")
print(f"bos: {mt_tokenizer.bos}")
print(f"eos: {mt_tokenizer.eos}")


ids1 = [
    mt_tokenizer.bos_token_id,
    mt_tokenizer.eos_token_id,
]
print(ids1)

ids2 = [
    mt_tokenizer.bos,
    mt_tokenizer.eos,
]
print(ids2)

assert ids1 == ids2

for id in ids1:
    print(id)
    print(f"{id}, {mt_tokenizer._tokenizer.id_to_piece(id)}")

print("done")