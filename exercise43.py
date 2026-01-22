import nltk
from nltk import CFG
from nltk.parse import ChartParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'a' | 'the'
N -> 'boy' | 'girl' | 'apple'
V -> 'eats' | 'likes'
""")

parser = ChartParser(grammar)

sentence = "the boy eats a apple".split()

print("Sentence:", " ".join(sentence))
print("\nParse Tree(s):")
found = False
for tree in parser.parse(sentence):
    found = True
    print(tree)

if not found:
    print("No parse tree found (sentence does not match grammar).")
