import spacy
import pandas as pd

nlp = spacy.load("en_core_web_md")

raw = pd.read_csv("./fact_tokens.txt", header=None)
raw = pd.DataFrame(raw)
raw.columns = ["words"]
#raw = raw[0].apply(str.lower)
raw_2 = pd.read_csv("./fact_token_exp.txt", header=None)

#print(raw_2.head(10))

# Propcess raw 
pos = []
for tok in raw["words"]:
    doc = nlp(tok.lower())
    for t in doc:
        pos.append(t.pos_)


raw["POS"] = pos
print(raw.head())
