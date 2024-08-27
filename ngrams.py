import nltk
from nltk.util import ngrams
from collections import Counter

# Sample text data
text = "The quick brown fox jumps over the lazy dog"

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Generate Unigrams (1-gram)
unigrams = list(ngrams(tokens, 1))
print("Unigrams:")
print(unigrams)

# Generate Bigrams (2-gram)
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Generate Trigrams (3-gram)
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Count frequency of each n-gram (for demonstration)
unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)
# Print frequencies (optional)
print("\nUnigram Frequencies:")
print(unigram_freq)
print("\nBigram Frequencies:")
print(bigram_freq)
print("\nTrigram Frequencies:")
print(trigram_freq)