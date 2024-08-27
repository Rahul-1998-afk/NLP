import spacy

# Load the pre-trained NLP model from spacy
nlp = spacy.load('en_core_web_sm')

# The sentence for which we want to perform NER
sentence = "Barack Obama was the 44th President of the United States."

# Process the sentence using the NLP model
doc = nlp(sentence)

# Print the named entities recognized in the sentence
print("Named Entities in the sentence:")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")