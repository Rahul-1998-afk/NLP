import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Sample text data
text = [
    "The cat sat on the mat",
    "The dog barked at the cat",
    "The cat chased the mouse",
    "The dog chased the cat",
]

# Tokenize the sentences into words
tokenized_text = [word_tokenize(sentence.lower()) for sentence in text]

# Train a Word2Vec model on the tokenized text
model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, sg=0)

# Get the word embeddings for a specific word
cat_vector = model.wv['cat']

# Print the word embedding for 'cat'
print("Word Embedding for 'cat':")
print(cat_vector)

# Find words most similar to 'cat'
similar_words = model.wv.most_similar('cat', topn=5)
print("\nWords most similar to 'cat':")
print(similar_words)