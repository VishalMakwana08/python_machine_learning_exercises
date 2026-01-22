import nltk
from nltk import pos_tag, word_tokenize, RegexpParser

# Downloading necessary resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

# Grammar definition for Noun Phrases (NP)
# Removed hidden non-breaking space characters
grammar = "NP: {<DT>?<JJ>*<NN.*>+}"

chunk_parser = RegexpParser(grammar)
tree = chunk_parser.parse(tagged)

print("Sentence:", text)
print("\nTokens:", tokens)
print("\nPOS Tagged Words:")
print(tagged)

print("\nChunk Tree (text form):")
print(tree)

# Optional: Visualize the tree (requires ghostscript installed on your OS)
# tree.draw()