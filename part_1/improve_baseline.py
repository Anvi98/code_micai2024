import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from rank_bm25 import BM25Okapi
import re
from nltk.tokenize import word_tokenize
from fast_bm25 import BM25
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import nltk
from scipy.stats import spearmanr


np.random.seed(9)

def evaluate_model(pipeline, param_grid, model_name):
    # Initialize grid search with pipeline and parameter grid
    cv = KFold(n_splits=5, shuffle=True, random_state=9)  # 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    # Fit the grid search object
    grid_search.fit(scores, y)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Train the best model
    best_model.fit(scores, y)

    # Predict on the training data
    y_pred = best_model.predict(scores)

    # Calculate F1 score
    f1 = f1_score(y, y_pred)

    # Print evaluation metrics
    print(f"{model_name} F1 Score:", f1)

    # Print classification report
    print(f"{model_name} Classification Report:")
    print(classification_report(y, y_pred))

# Download NLTK resources
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Expand contracted words (e.g., "can't" to "cannot")
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'s", " is", text)
    # Add more expansions as needed
    return text


# Define a function to extract the final classifier from the pipeline
def get_final_classifier(pipeline):
    # Get the named steps of the pipeline
    steps = pipeline.named_steps
    # Loop through the steps to find the final classifier
    for step_name, step_object in steps.items():
        # Check if the step is a classifier
        if hasattr(step_object, 'coef_'):
            return step_object
    # If no classifier is found, return None
    return None

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, bm25_scores, scores_tensor, y):
        self.bm25_scores = bm25_scores
        self.scores_tensor = scores_tensor
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.bm25_scores[idx], self.scores_tensor[idx], self.y[idx]


# Function to preprocess text and calculate BM25 scores
import multiprocessing
import functools 

# Define the process_text function outside of preprocess_and_calculate_bm25
def process_text(text, bm25):
    return bm25.get_scores(text)

def preprocess_and_calculate_bm25(data):
    # Define the number of CPU cores to use
    num_cores = min(multiprocessing.cpu_count(), 9)  # Using at most 6 CPU cores

    # Preprocess text
    X_preprocessed = [preprocess_text(text) for text in data['args']]
    corpus = [text.split() for text in X_preprocessed]
    bm25 = BM25Okapi(corpus)  # Initialize BM25 object

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_cores)

    # Define a partial function to pass bm25 to process_text
    partial_process_text = functools.partial(process_text, bm25=bm25)

    # Calculate BM25 scores in parallel
    scores = pool.map(partial_process_text, X_preprocessed)

    # Close the pool
    pool.close()
    pool.join()

    return scores

# Load dataset
data = pd.read_csv("./transformed_final.csv")
data = data.sample(frac=1).reset_index(drop=True)
data = data.iloc[:int(len(data)/2),:]
print(len(data))

# Define X and y
X = data['args']
y = data['labels']

# Preprocess text data and calculate BM25 scores
print("geting bm25 scores...")
scores = preprocess_and_calculate_bm25(data)
print("Done")

# Define parameter grids for grid search
param_grid_lr = {'clf__C': [1]}  # Parameters for Logistic Regression
param_grid_nb = {}  # Parameters for Naive Bayes
param_grid_svm = {'clf__C': [1], 'clf__kernel': ['linear']}  # Parameters for SVM
param_grid_rf = {'clf__n_estimators': [200]}  # Parameters for Random Forest

# Create pipelines with BM25 vectorizer and classifiers
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),  # Add feature scaling
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline_nb = Pipeline([
    ('clf', MultinomialNB())
])

pipeline_svm = Pipeline([
    ('clf', SVC())
])

pipeline_rf = Pipeline([
    ('clf', RandomForestClassifier())
])

cv = KFold(n_splits=3, shuffle=True, random_state=9)

# Define custom scorers for F1 score and Spearman correlation
def spearman_corr_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return spearmanr(y, y_pred).correlation

custom_scorers = {
    'f1_score': make_scorer(f1_score),
    'spearman_corr': make_scorer(spearman_corr_scorer)
}

# Define a custom function to evaluate the model

def evaluate_model(pipeline, param_grid, model_name):
    # Initialize grid search with pipeline and parameter grid
    cv = KFold(n_splits=5, shuffle=True, random_state=9)  # 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    # Fit the grid search object
    grid_search.fit(scores, y)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Train the best model
    best_model.fit(scores, y)

    # Predict on the training data
    y_pred = best_model.predict(scores)

    # Calculate F1 score
    f1 = f1_score(y, y_pred)

    # Print evaluation metrics
    print(f"{model_name} F1 Score:", f1)

    # Print classification report
    print(f"{model_name} Classification Report:")
    print(classification_report(y, y_pred))

    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(y, y_pred)
    print(f"{model_name} Spearman Correlation:", spearman_corr)# Perform grid search and evaluation for each model


print("Evaluation RF...")
evaluate_model(pipeline_rf, param_grid_rf, "Random Forest")
print("Evaluation LR...")
evaluate_model(pipeline_lr, param_grid_lr, "Logistic Regression")
print("Evaluation NB...")
evaluate_model(pipeline_nb, param_grid_nb, "Naive Bayes")
print("Evaluation SVM...")
evaluate_model(pipeline_svm, param_grid_svm, "SVM")


