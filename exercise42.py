import nltk
from nltk import word_tokenize, pos_tag

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")

sentence = "Students are learning Natural Language Processing"
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)

print("Sentence:", sentence)
print("Tokens:", tokens)
print("POS Tags:", tagged)

print("\nWord -> POS (Sentence Structure):")
for word, tag in tagged:
    print(word, "->", tag)



