import sys

sys.path.append('/tmp/amlt-code-download/abgoswam_epf')
print('\n'.join(sys.path))

from megatron.tokenizer.tokenizer import _SentencePieceTokenizer

vocab_file = "/tmp/amlt-code-download/abgoswam_epf/my_starcoder_trials_0918/ckpts/Mistral-7B-v0.1/tokenizer.model"
mt_tokenizer = _SentencePieceTokenizer(vocab_file,
                                    new_tokens=False)

print(mt_tokenizer)

print(f"cls: {mt_tokenizer.cls}")
print(f"sep: {mt_tokenizer.sep}")
print(f"eod: {mt_tokenizer.eod}")
print(f"mask: {mt_tokenizer.mask}")
print(f"pad: {mt_tokenizer.pad}")
print(f"bos: {mt_tokenizer.bos}")
print(f"eos: {mt_tokenizer.eos}")


ids = [
    mt_tokenizer.bos_token_id,
    mt_tokenizer.eos_token_id,
]

for id in ids:
    print(id)
    print(f"{id}, {mt_tokenizer._tokenizer.id_to_piece(id)}")

print("done")

