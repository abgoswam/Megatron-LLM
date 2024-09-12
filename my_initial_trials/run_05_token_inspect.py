import sentencepiece as spm

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("./weights_conversion/out_mistral_7b/tokenizer.model")

# Print the tokenizer type
# print(f"Model Type: {sp.model_proto().trainer_spec.model_type}")

# Inspect general details (though it may not directly show model type)
print(f"Number of tokens: {sp.get_piece_size()}")
print(f"First few tokens: {sp.id_to_piece(0)}, {sp.id_to_piece(1)}, {sp.id_to_piece(2)}")

# Sample text to tokenize
text = "hellohellohellohello"

# Convert text to tokens (IDs)
token_ids = sp.encode(text, out_type=int)  # Token IDs
print("Token IDs:", token_ids)

# Convert text to tokens (subword units)
tokens = sp.encode(text, out_type=str)  # Token strings
print("Tokens:", tokens)

print("done")

