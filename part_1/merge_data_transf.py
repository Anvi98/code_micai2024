import pandas as pd 

raw1 = pd.read_csv("./transformed_data.tsv", sep="\t")
raw2 = pd.read_csv("./transformed_data2.tsv", sep="\t")

final = pd.concat([raw1, raw2], ignore_index=True)

final.to_csv("./transformed_final.csv", index=False)
