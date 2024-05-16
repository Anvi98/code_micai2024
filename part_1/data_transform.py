"""In this script, we transform two datasets from IBM Debater Argument Quality focusing on the pair samples 
for a binary classification. The dataset is restreint to 24.1k over 37k samples. """


import numpy as np 
import pandas as pd 
import os 

raw = pd.read_csv("../data/arg14/Online-shopping-brings-more-harm-than-good-(PRO).tsv", sep="\t")
cols = ["label", "a1", "a2"]
print(raw.info())

new_data = {'args': [], 'labels': []}

mem = 1
for i in range(len(raw)):
    tmp_label = raw.iloc[i]["label"]
    if mem == 1:
        new_data['args'].append(raw.iloc[i][f"{tmp_label}"])
        new_data['labels'].append(True)
        mem = -1
    elif mem == -1:
        if tmp_label == 'a1':
            new_data['args'].append(raw.iloc[i]['a2'])
        elif tmp_label == 'a2':
            new_data['args'].append(raw.iloc[i]['a1'])
        
        new_data['labels'].append(False)
        mem = 1

print(new_data)






