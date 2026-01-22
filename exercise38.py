import nltk

print("NLTK version:", nltk.__version__)

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")   # FIX for your error
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Test after download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

text = "NLTK is used for Natural Language Processing. It helps in text analysis."

tokens = word_tokenize(text)

sw = set(stopwords.words("english"))
filtered = [w for w in tokens if w.isalpha() and w.lower() not in sw]

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w.lower()) for w in filtered]

print("\nText:", text)
print("Tokens:", tokens)
print("Filtered:", filtered)
print("Lemmatized:", lemmatized)
