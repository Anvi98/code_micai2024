
import pandas as pd 

raw = pd.read_csv("./merged_data_with_text.csv")
raw2 = pd.read_csv("./transformed_data2.tsv", sep='\t')
print(raw["args"].value_counts(), raw2["labels"].value_counts())

text = "I am going to school"
text = text.split(" ")
s = "going to"

count = 0
for i in range(len(text) - 1):
    tmp_  = " ".join([text[i], text[i+1]])
    print(tmp_)
    if s == tmp_:
        count += 1

print(count)
        

