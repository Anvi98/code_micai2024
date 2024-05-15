"""This script is intended to assess the performance of Logistic Regression model
On several features of the datasets. """

from pandas.core.common import random_state
import sklearn
from sklearn.linear_model import LogisticRegression 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# load data
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

X_tr_combined= hstack([X_t_idf, X_others_t])
X_te_combined= hstack([X_te_idf, X_others_te])

# train 

lr_00 = LogisticRegression()
lr_00.fit(X_tr_combined, y_train)

# Predict 
y_preds = lr_00.predict(X_te_combined)

# Evaluate 
accuracy = accuracy_score(y_test, y_preds)
class_report = classification_report(y_test, y_preds)
print("Classification Report:")
print(class_report)
print(accuracy)
