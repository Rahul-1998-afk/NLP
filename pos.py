import nltk
nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag
from nltk import word_tokenize

text = "GeeksforGeeks is a Computer Science platform."
tokenized_text = word_tokenize(text)
tags = tokens_tag = pos_tag(tokenized_text)

print(tags)