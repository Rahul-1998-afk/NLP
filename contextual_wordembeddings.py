from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Example sentence
sentence = "The cat sat on the mat."

# Tokenize the input sentence and convert tokens to tensor
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Pass the input through the BERT model to get embeddings
with torch.no_grad():
    outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state

# Print the shape of the last hidden states tensor
print("Shape of last hidden states:", last_hidden_states.shape)

# Convert the embeddings to numpy array (for easier manipulation)
embeddings = last_hidden_states.squeeze().numpy()

# Tokenize the sentence to match embeddings to words
tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

# Print the tokens and their corresponding contextual embeddings
for token, embedding in zip(tokens, embeddings):
    print(f"Token: {token}")
    print(f"Embedding: {embedding[:10]}...")  # Print first 10 dimensions for brevity
    print()