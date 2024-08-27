from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data (documents)
documents = [
    "The cat sat on the mat.",
    "The cat sat on the bed.",
    "The dog barked."
]

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the model and transform the documents into TF-IDF representation
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (unique words in the corpus)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix into an array
tfidf_array = tfidf_matrix.toarray()

# Display the TF-IDF matrix
print("Feature Names (Words):", feature_names)
print("\nTF-IDF Matrix:")
print(tfidf_array)