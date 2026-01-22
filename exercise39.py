import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download tokenizer resources (run once)
nltk.download("punkt")
nltk.download("punkt_tab")  # some versions require this

# Step 1: Create Porter Stemmer object
stemmer = PorterStemmer()

# Step 2: Input text
text = "Stemming reduces words like running, runner, and runs to a root form."

# Step 3: Tokenize text into words
tokens = word_tokenize(text)

# Step 4: Apply stemming (only for alphabet words)
stemmed_words = [stemmer.stem(word) for word in tokens if word.isalpha()]

print("Original Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nStemmed Words:")
print(stemmed_words)
