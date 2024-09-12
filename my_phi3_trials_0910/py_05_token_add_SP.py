# Import the _SentencePieceTokenizer class (ensure it's accessible in your script).
# Replace 'path/to/your/vocab.model' with the path to your SentencePiece model file.

import sys
sys.path.append('/tmp/amlt-code-download/abgoswam_epf')
print('\n'.join(sys.path))

from megatron.tokenizer.tokenizer import _SentencePieceTokenizer

model_file = "./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct/tokenizer.model"  # Replace with the path to your SentencePiece model file.

# Instantiate the tokenizer with vocab_extra_ids set to 64.
tokenizer = _SentencePieceTokenizer(model_file=model_file, vocab_extra_ids=64, new_tokens=False)

# Verify the vocabulary size.
print(f"Vocabulary size: {tokenizer.vocab_size}")  

# tokenizer = _SentencePieceTokenizer(args.vocab_file, vocab_extra_ids=args.vocab_extra_ids, 
#                                     vocab_extra_ids_list=args.vocab_extra_ids_list, new_tokens=args.new_tokens)

# This should print 32064.

# tokenizer = _SentencePieceTokenizer(model_file=model_file, vocab_extra_ids=64)
