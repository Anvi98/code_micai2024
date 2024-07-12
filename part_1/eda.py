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
plt.show()

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
def flatten_doc(doc):
    doc = list(doc.to_dict().values())
    doc = [" ".join(paragraph.split(" ")) for paragraph in doc] 
    flat = " ".join(doc)

    return flat #This is a string

def extract_rel_tokens(flat_doc, vectorizer):
    # Fit and transform the corpus
    count_matrix = vectorizer.fit_transform([flat_doc])
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    # Convert count matrix to DataFrame
    df = pd.DataFrame(count_matrix.toarray(), columns=feature_names)
    flat_dict = df.to_dict()

    for word,count in flat_dict.items():
        flat_dict[word] = count[0]

    flat_dict = dict(sorted(flat_dict.items(), key=lambda x: x[1], reverse=True))

    return flat_dict ## A dictionary

# On relevant_args only (1,2,3 grams)
#rel = relevant_args["text"]
rel_logic = relevant_args[relevant_args["cogency_(logic)"] >= 3]["text"]
rel_dialectic = relevant_args[relevant_args["reasonableness_(dialectic)"] >= 3]["text"]
rel_rhetoric = relevant_args[relevant_args["effectiveness_(rhetoric)"] >= 3]["text"]
corpora = [ rel_logic, rel_dialectic, rel_rhetoric]

# Initialize CountVectorizer with custom analyzer and n_grams from 1-3
vectorizer = CountVectorizer(ngram_range=(1, 1))
vectorizer2 = CountVectorizer(ngram_range=(2, 2))
vectorizer3 = CountVectorizer(ngram_range=(3, 3))
vectorizers = [vectorizer, vectorizer2, vectorizer3]

#words = relevant_args["text"]
#words = rel_logic
#count_rel = {"1-grams": [], "2-grams": [], "3-grams": []}
count_logic = {"1-grams": [], "2-grams": [], "3-grams": []}
count_dialectic = {"1-grams": [], "2-grams": [], "3-grams": []}
count_rethoric = {"1-grams": [], "2-grams": [], "3-grams": []}
counts = [count_logic, count_dialectic, count_rethoric]

idx = 0
for corpus in corpora:
    j = 0
    for vec in vectorizers:
        
        flat = flatten_doc(corpus)
        flat_dict = extract_rel_tokens(flat, vec)
        counts[idx][f"{j+1}-grams"].append(flat_dict) 
        j +=1
    idx +=1

#print(counts[0]["3-grams"])

## Filter by counts 
thresholds = [100, 100, 50]
#test = count_rel["1-grams"][0]


## Continue here to save the key terms for each qualities
filtered_logic = {"1-grams": [], '2-grams': [], "3-grams": []}
filtered_dialectic = {"1-grams": [], '2-grams': [], "3-grams": []}
filtered_rehtoric = {"1-grams": [], '2-grams': [], "3-grams": []}
filtered_words = [filtered_logic, filtered_dialectic, filtered_rehtoric]

for i in range(len(counts)):
    tmp_idx = 0
    for k,v in counts[i].items():
        for w, count in v[0].items():
            if count >= thresholds[tmp_idx]:
                filtered_words[i][f"{tmp_idx+1}-grams"].append(w)
            if count < thresholds[tmp_idx]:
                break
        tmp_idx +=1

#filtered = [k for k,v in test.items() if v > thresholds[0]]
#print(filtered)
#print(filtered_words[2]["3-grams"])
#print(counts[2]["3-grams"])
unigram_words = []
bi_grams = []
tri_grams = []
sets_grams = [unigram_words, bi_grams, tri_grams]

for i in range(len(filtered_words)):
    tmp_idx = 0
    for k,v in filtered_words[i].items():
        sets_grams[tmp_idx].append(v)
        tmp_idx +=1


# Convert to sets for plotting 
unigram_words = [set(l) for l in unigram_words]
bi_grams= [set(l) for l in bi_grams] 
tri_grams = [set(l) for l in tri_grams]

uni_logic, uni_dialectic, uni_rethoric = unigram_words
bi_logic, bi_dialectic, bi_rethoric = bi_grams
tri_logic, tri_dialectic, tri_rethoric = tri_grams

# Create sets for each column
venn3([tri_logic, tri_dialectic, tri_rethoric], set_labels=("logic", "dialectic", "rhetoric"))
#plt.savefig("proportion_dimensions_venn_diagrams_on_sample.jpeg")
plt.show()



"""Notes on this short EDA: 
    - The dataset is comprised of 6k samples in an organised text format (csv)
    - out of the samples, __ are considered by the annonators arguments while 57 as Non-arguments
    - While we have 6k considered arguments, only ____ are considered relevants taking a threshold of >=3. So ___ samples are considered non-relevant by annotators. 
    - We found in the relevants arguments that ___ arguments are logical, ___ are dialectical and ___ are rhetorical which makes the distribution of dimensions in relevant args balanced. 
    - Which means that for an argument to be logic, it has to be relevant at first glance with the conclusion or topic from the dataset."""

