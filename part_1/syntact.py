""" This script contains all the algorithms to extract specific syntactic features to improve model performance
in Argument mining. The features we would like to extract should be related to: 
    - Passive and active voice features 
    - Facts and regulation structures and tokens 
    - Citation features. 
    - Explanation features."""

import pandas as pd 
import numpy as np 


# load dataasets
raw = pd.read_csv("../data/arguments.csv")
cols = ["is_argument", "premise"]
data_exp = raw[cols].copy()
data_nump = data_exp.to_numpy()

print(raw.columns)
