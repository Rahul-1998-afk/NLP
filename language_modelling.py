import nltk
from collections import defaultdict, Counter
import random
# Sample text data (corpus)
corpus = [
    "I love natural language processing",
    "I love machine learning",
    "I enjoy learning new things",
    "Natural language processing is fascinating"
]

tokenized_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]
# Create bigrams from the tokenized corpus
bigrams = []
for sentence in tokenized_corpus:
    bigrams.extend(list(nltk.bigrams(sentence)))

# Calculate bigram frequencies
bigram_freq = defaultdict(Counter)
for w1, w2 in bigrams:
    bigram_freq[w1][w2] += 1

# Calculate bigram probabilities
bigram_prob = defaultdict(dict)
for w1 in bigram_freq:
    total_count = float(sum(bigram_freq[w1].values()))
    for w2 in bigram_freq[w1]:
        bigram_prob[w1][w2] = bigram_freq[w1][w2] / total_count

# Function to generate text using the bigram model
def generate_sentence(start_word, num_words=10):
    current_word = start_word
    sentence = [current_word]
    for _ in range(num_words - 1):
        if current_word in bigram_prob:
            next_word = random.choices(
                list(bigram_prob[current_word].keys()),
                list(bigram_prob[current_word].values())
            )[0]
            sentence.append(next_word)
            current_word = next_word
        else:
            break
    return ' '.join(sentence)

# Generate a sentence starting with the word "i"
generated_sentence = generate_sentence("i", num_words=4)
print("Generated sentence:", generated_sentence)