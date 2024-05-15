from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

raw = pd.read_csv("../data/arguments.csv")
cols = ["is_argument", "premise", "rhetorical_quality", "dialectical_quality", "logical_quality"]

data = raw[cols].copy()

# Extract features 
## TFIDF
raw_features = data[["premise", "rhetorical_quality", "dialectical_quality", "logical_quality"]]
label = data["is_argument"]
label = label.astype(int)
label = label.to_numpy()


# Accessing coo_matrix
#for i in range(len(X.data)):
#    print("Element at ({}, {}): {}".format(X.row[i], X.col[i], X.data[i]))

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(raw_features, label, test_size=0.2, random_state=42)

# X and Y train manip for tfidf
X_prem_t, X_prem_te = X_train["premise"].to_numpy(), X_test["premise"].to_numpy()

X_others_t, X_others_te = X_train[["rhetorical_quality", "dialectical_quality", "logical_quality"]].to_numpy(), X_test[["rhetorical_quality", "dialectical_quality", "logical_quality"]].to_numpy()
X_others_t, X_others_te = X_others_t.reshape(-1,3), X_others_te.reshape(-1,3)

vectorizer = TfidfVectorizer(ngram_range=(1,3))
X_t_idf = vectorizer.fit_transform(X_prem_t)
X_te_idf = vectorizer.transform(X_prem_te)

X_tr_combined= hstack([X_t_idf, X_others_t]).toarray()
X_te_combined= hstack([X_te_idf, X_others_te]).toarray()


#data = load_breast_cancer()
#X = data.data
#y= data.target
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)


clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)


models,predictions = clf.fit(np.array(X_tr_combined), np.array(X_te_combined), y_train, y_test)

print(models,predictions)


