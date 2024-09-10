import sentencepiece as spm

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("./my_phi3_trials_0910/ckpts/Phi-3-mini-4k-instruct/tokenizer.model")

# Print the tokenizer type
# print(f"Model Type: {sp.model_proto().trainer_spec.model_type}")

# Inspect general details (though it may not directly show model type)
print(f"Number of tokens: {sp.get_piece_size()}")
print(f"First few tokens: {sp.id_to_piece(0)}, {sp.id_to_piece(1)}, {sp.id_to_piece(2)}")

# Sample text to tokenize
text = "hellohellohellohello"
text = "<|endoftext|><|assistant|><|system|><|end|><|user|>" # [32000, 32001, 32006, 32007, 32010]

# Convert text to tokens (IDs)
token_ids = sp.encode(text, out_type=int)  # Token IDs
print("Token IDs:", token_ids)

# Convert text to tokens (subword units)
tokens = sp.encode(text, out_type=str)  # Token strings
print("Tokens:", tokens)

# Verify if special tokens exist in the model
special_tokens = ["<|endoftext|>", "<|assistant|>"]

for token in special_tokens:
    token_id = sp.piece_to_id(token)
    if token_id != sp.unk_id():  # Check if the ID is not the unknown token ID
        print(f"Token '{token}' found with ID: {token_id}")
    else:
        print(f"Token '{token}' not found in the model.")

print("done")

