import nltk
from nltk import CFG
from nltk.parse import ChartParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the' | 'a'
N -> 'cat' | 'mouse'
V -> 'chased' | 'saw'
""")

parser = ChartParser(grammar)
sentence = "the cat chased a mouse".split()

print("Sentence:", " ".join(sentence))
print("\nGrammar Tree(s):")
for tree in parser.parse(sentence):
    print(tree)

