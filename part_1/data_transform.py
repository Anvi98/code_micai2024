"""In this script, we transform two datasets from IBM Debater Argument Quality focusing on the pair samples 
for a binary classification. The dataset is restreint to 24.1k over 37k samples. """


import numpy as np 
import pandas as pd 
import os 

# Get the list of tsv files from folders to read 

def get_files(folder_path):
    "Get the path of a folder containing files and return a list of the file names."
    return os.listdir(folder_path)

previous_loc = ".."
root_folder = "./data/arg14/"
files = get_files(os.path.join(previous_loc, root_folder))
print(files)

## get the .md files:

# Loop through each file and construct the dataset 
new_data = {'args': [], 'labels': []}

for file in files:
    # open file
    #with open(os.path.join(previous_loc, root_folder, file)) as f:
    #raw = pd.read_csv("../data/arg14/Online-shopping-brings-more-harm-than-good-(PRO).tsv", sep="\t")
    tmp_raw = pd.read_csv(f"{os.path.join(previous_loc, root_folder, file)}", sep="\t")
    #cols = ["label", "a1", "a2"]


    mem = 1
    for i in range(len(tmp_raw)):
        tmp_label = tmp_raw.iloc[i]["label"]
        if mem == 1:
            new_data['args'].append(tmp_raw.iloc[i][f"{tmp_label}"])
            new_data['labels'].append(True)
            mem = -1
        elif mem == -1:
            if tmp_label == 'a1':
                new_data['args'].append(tmp_raw.iloc[i]['a2'])
            elif tmp_label == 'a2':
                new_data['args'].append(tmp_raw.iloc[i]['a1'])
            
            new_data['labels'].append(False)
            mem = 1

new_data = pd.DataFrame(new_data)
new_data.to_csv("transformed_data.tsv", sep="\t", index=False)




