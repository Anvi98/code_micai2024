
import pandas as pd 

raw = pd.read_csv("./merged_data_with_text.csv")
raw2 = pd.read_csv("./transformed_data2.tsv", sep='\t')
print(raw["args"].value_counts(), raw2["labels"].value_counts())
