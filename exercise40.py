import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (run once)
nltk.download("punkt")
nltk.download("punkt_tab")   # some versions require this
nltk.download("wordnet")
nltk.download("omw-1.4")

# Step 1: Create Lemmatizer object
lemmatizer = WordNetLemmatizer()

# Step 2: Input text
text = "The children are playing and the cats were running faster than the mice."

# Step 3: Tokenize text into words
tokens = word_tokenize(text)

# Step 4: Lemmatize only alphabetic words (ignore punctuation)
lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]

print("Original Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nLemmatized Words (Base Form):")
print(lemmatized_words)

