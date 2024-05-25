""" In this script, we just implement a simple EDA to have an understanding of the dataset. It focuses mostly
on the dimensions of the datasets and understand the limits of the datasets for the current research."""


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles, venn3_unweighted 
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer
import operator

# load dataset 
raw = pd.read_csv("../data/arguments.csv")
columns = ["is_argument", "relevance", "logical_quality", "rhetorical_quality", "dialectical_quality", "conclusion", "premise"]

## Data From Gramarly
previous_loc = ".."
files_folder = "./data/GAQcorpus/GAQcorpus/all/"
path_2_files = glob.glob(os.path.join(previous_loc, files_folder, "*.csv"))
df = pd.concat((pd.read_csv(f) for f in path_2_files), ignore_index=True)

## Change column_names to match_it
#print(df.columns)
df_cols = ["cogency_mean", "effectiveness_mean", "reasonableness_mean", "overall_mean", "argumentative_majority", "text", "title"]
final_eda_data = df[df_cols].copy()
final_eda_data.columns = ["cogency_(logic)", "effectiveness_(rhetoric)", "reasonableness_(dialectic)", "relevance", "is_argument", "text", "title"]

final_eda_data.replace("#", 0, inplace=True)
# Convert columns into float
columns_to_convert = ["cogency_(logic)", "effectiveness_(rhetoric)", "reasonableness_(dialectic)", "relevance", "is_argument"]
final_eda_data[columns_to_convert] = final_eda_data[columns_to_convert].astype(float)
print(final_eda_data.info())
#print(final_eda_data.columns, final_eda_data.info())
#final_eda_data.to_csv("merged_gaqcorpus.csv", index=True)

eda_data = raw[columns].copy()

# Number of arguments and not arguments
# Separate Args and non-args in two different pandas series
args = final_eda_data[final_eda_data["is_argument"] == True]

non_args = final_eda_data[final_eda_data["is_argument"] == False]
print(f"Number of arguments: {args.is_argument.count()} \nNumber of non-arguments: {non_args.is_argument.count()}")

# Number of relevant args (relevant >= 3)
relevant_args = args[args["relevance"] >= 3]
print(f"Number of relevant arguments(rel>=3): {relevant_args.is_argument.count()}")

# How many relevant args are logical 
num_rel_logic = relevant_args[relevant_args["cogency_(logic)"] >= 3]["is_argument"].count()
print(f"Number of relevant logical args: {num_rel_logic}")

# How many relevant args are dialectical
num_rel_dialect = relevant_args[relevant_args["reasonableness_(dialectic)"] >= 3]["is_argument"].count()
print(f"number of relevant dialect: {num_rel_dialect}")

# How many relevant args are Rhetorical 
num_rel_rheto= relevant_args[relevant_args["effectiveness_(rhetoric)"] >= 3]["is_argument"].count()
print(f"number of relevant rhetoric args: {num_rel_rheto}") 

# Number of relevant arguments which are alltogether Logic, Rhetoric and dialectic 
num_rel_all = relevant_args[(relevant_args["cogency_(logic)"] >=4) & 
                            (relevant_args["effectiveness_(rhetoric)"] >= 4) & 
                            (relevant_args["reasonableness_(dialectic)"] >= 4)]["is_argument"].count()

print(f"Number of Rel args that have the highest score for all dims: {num_rel_all}")

tmp_data = {
        "logic": list(relevant_args["cogency_(logic)"]),
        "dialectic":list(relevant_args["reasonableness_(dialectic)"]),
        "rhetoric": list(relevant_args["effectiveness_(rhetoric)"]),
        }

for i, (k,v) in enumerate(tmp_data.items()):
    tmp_c = 0
    for j in v:
        tmp_val = f"{j}r{tmp_c}"
        tmp_c += 1
        tmp_idx = v.index(j)
        v[tmp_idx] = tmp_val
#print(tmp_data["logic"])

tmp_df = pd.DataFrame(tmp_data)
set_logic = set(tmp_df["logic"])
set_dialec = set(tmp_df["dialectic"])
set_rheto= set(tmp_df["rhetoric"])
# Create sets for each column
venn3([set_logic, set_dialec, set_rheto], set_labels=("logic", "dialectic", "rhetoric"))
plt.savefig("proportion_dimensions_venn_diagrams_on_sample.jpeg")
#plt.show()

## Plotting

# number of Arguments vs non-arguments
fig = plt.figure(figsize=(10,5))
plt.bar(["Arguments", "Non-arguments"], [args.is_argument.count(), non_args.is_argument.count()])
plt.xlabel("Argument class")
plt.ylabel("No. of each class")
plt.title("Number of arguments and non-arguments")
plt.savefig("arg_vs_non_args.png")
plt.show()


# Number of Relevants vs non-relevant arguments 
num_non_rel_args = args["is_argument"].count() - relevant_args["is_argument"].count()
print(num_non_rel_args, args["is_argument"].count(), relevant_args.is_argument.count())

# Proportions of relevant args by dimensions
y = np.array([num_rel_logic, num_rel_dialect, num_rel_rheto])
plt.pie(y, labels = ["logical", "dialectical", "rhetorical"], explode=[0.1,0,0], autopct='%1.1f%%')
plt.savefig("proportions_dimensions_rel_args.png")
plt.show()

#Word count on relevant_args to identify repeated terms 
# On relevant_args only (1,2,3 grams)

vectorizer = CountVectorizer(analyzer="word", ngram_range=(1,1))
vectorizer2 = CountVectorizer(analyzer="word", ngram_range=(2,2))
vectorizer3 = CountVectorizer(analyzer="word", ngram_range=(3,3))

count_1 = vectorizer.fit_transform(relevant_args["text"])
unigram = vectorizer.get_feature_names_out()
df = pd.DataFrame(count_1.toarray(), columns=unigram)
#df.to_csv("test00.csv")
count_2 = vectorizer2.fit_transform(relevant_args["text"])
twograms = vectorizer2.get_feature_names_out()

count_3 = vectorizer3.fit_transform(relevant_args["text"])
trigrams = vectorizer3.get_feature_names_out()

words = relevant_args["text"]
words = list(words.to_dict().values())
words = " ".join(words).split(" ")

vocab = {}

for w in twograms:
    i = 0
    for i in range(len(words) - 1):
        tmp_w = " ".join([words[i], words[i+1]])
        if w.lower() == tmp_w.lower():
            if tmp_w.lower() in list(vocab.keys()):
                vocab[tmp_w.lower()] += 1 
            elif tmp_w.lower() not in list(vocab.keys()):
                vocab[tmp_w.lower()] = 1
    i +=1
    print(i)

vocab = sorted(vocab.items(), key=operator.itemgetter(1))

#print(vocab)

"""Notes on this short EDA: 
    - The dataset is comprised of 437 samples in an organised text format (csv)
    - out of the samples, 437 are considered by the annonators arguments while 57 as Non-arguments
    - While we have 437 considered arguments, only 280 are considered relevants taking a threshold of >=3. So 157 samples are considered non-relevant by annotators. 
    - We found in the relevants arguments that 185 arguments are logical, 193 are dialectical and 174 are rhetorical which makes the distribution of dimensions in relevant args balanced. 
    - Which means that for an argument to be logic, it has to be relevant at first glance with the conclusion or topic from the dataset."""
