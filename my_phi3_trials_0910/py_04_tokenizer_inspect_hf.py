from transformers import AutoTokenizer, PreTrainedTokenizerFast
import sentencepiece as spm
from transformers import AutoTokenizer

# Specify the path to the tokenizer files
tokenizer_path = "./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct"

# Load the tokenizer using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# Test the tokenizer
text = "The Hugging Face tokenizer might include extra tokens for compatibility or other reasons that were added post-training."  # Expecting specific token IDs
# text = "<|endoftext|><|assistant|><|system|><|end|><|user|><s></s>"  # Expecting specific token IDs
tokens = tokenizer.tokenize(text)
print(tokens)

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct/tokenizer.model")
tokens_sp = sp.encode(text, out_type=str)  # Token strings
print(tokens_sp)

# Get the vocabulary size
vocab_size_hf = len(tokenizer.get_vocab())
print(f"Hugging Face Tokenizer Vocabulary Size: {vocab_size_hf}")

vocab_size_sp = sp.get_piece_size()
print(f"SentencePiece Tokenizer Vocabulary Size: {vocab_size_sp}")

# Debug =================
# Check if special tokens exist in the SentencePiece model
special_tokens = ["<|endoftext|>", "<|assistant|>", "<|system|>", "<|end|>", "<|user|>", "<s>", "</s>"]
for token in special_tokens:
    token_id = sp.piece_to_id(token)
    print(f"Token '{token}' found with ID: {token_id}" if token_id != sp.unk_id() else f"Token '{token}' not found.")

# Debug =================
# Check token IDs from Hugging Face tokenizer
for token in special_tokens:
    hf_token_id = tokenizer.convert_tokens_to_ids(token)
    sp_token_id = sp.piece_to_id(token)
    print(f"{token}: HF Token ID = {hf_token_id}, SP Token ID = {sp_token_id}")


assert tokens == tokens_sp