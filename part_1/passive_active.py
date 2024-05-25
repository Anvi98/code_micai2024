"""In this script i would like to detect if a sentence is active or in passive voice."""

import pandas as pd
import spacy


t1 = "John gives a banana to Paul"
t2 = "Paul was given a banana by John"
t3 = "It can also be called a CC"
t4 = "she was worried about the child she was to have."

nlp = spacy.load("en_core_web_sm")

doc = nlp(t1)
for t in doc:
    print(t.text, t.pos_, t.dep_)

print("------")
doc = nlp(t2)
for t in doc:
    print(t.text, t.pos_, t.dep_)

print("------")
doc = nlp(t3)
for t in doc:
    print(t.text, t.pos_, t.dep_)

print("------")
doc = nlp(t4)
for t in doc:
    print(t.text, t.pos_, t.dep_)
