import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import multiprocessing

# Random seed
np.random.seed(42)

# Load dataset
data = pd.read_csv("./transformed_final.csv")

# Randomize the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Define X and y
X = data['args']
y = data['labels']

# Define parameter grids for grid search
param_grid_lr = {'tfidf__ngram_range': [(1, 3)]}
param_grid_nb = {'tfidf__ngram_range': [(1, 3)]}
param_grid_svm = {'tfidf__ngram_range': [(1, 3)],
                  'clf__C': [1],
                  'clf__kernel': ['linear']}
param_grid_rf = {'tfidf__ngram_range': [(1, 3)],
                  'clf__n_estimators': [200]}

# Create a pipeline with TF-IDF vectorizer and classifier
pipeline_lr = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression(max_iter=1000))])
pipeline_nb = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
pipeline_svm = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC())])
pipeline_rf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier())])

# Perform grid search with 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Number of CPU cores
n_cores = multiprocessing.cpu_count()

def spearman_corr_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return spearmanr(y, y_pred).correlation

# Define custom scorers for F1 score and Spearman correlation
custom_scorers = {
    'f1_score': make_scorer(f1_score),
    'spearman_corr': make_scorer(spearman_corr_scorer)
}

def evaluate_model(grid_search, model_name):
    # Fit the grid search object
    grid_search.fit(X, y)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    # Final model with best parameters
    final_model = grid_search.best_estimator_

    # Evaluate model using cross-validation with parallel processing
    f1_scores = cross_val_score(final_model, X, y, cv=cv, scoring='f1', n_jobs=n_cores)
    spearman_scores = cross_val_score(final_model, X, y, cv=cv, scoring=spearman_corr_scorer, n_jobs=n_cores)

    # Print evaluation metrics
    print(f"{model_name} CV F1 Score:", f1_scores.mean())
    print(f"{model_name} CV Spearman Correlation:", spearman_scores.mean())

# Perform grid search and evaluation for each model
print("Evaluation LR...")
evaluate_model(GridSearchCV(pipeline_lr, param_grid_lr, cv=cv, scoring='accuracy'), "Logistic Regression")
print("Evaluation NB...")
evaluate_model(GridSearchCV(pipeline_nb, param_grid_nb, cv=cv, scoring='accuracy'), "Naive Bayes")
print("Evaluation SVM...")
evaluate_model(GridSearchCV(pipeline_svm, param_grid_svm, cv=cv, scoring='accuracy'), "SVM")
print("Evaluation RF...")
evaluate_model(GridSearchCV(pipeline_rf, param_grid_rf, cv=cv, scoring='accuracy'), "Random Forest")

