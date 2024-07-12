
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_numerical_features):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2 + num_numerical_features, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, textual_input, numerical_input):
        lstm_out, _ = self.lstm(textual_input)
        concat_input = torch.cat((lstm_out, numerical_input), dim=1)
        x = self.dropout(concat_input)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load Sentence-BERT model
sentence_bert_model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X_textual, X_numerical, y):
        self.X_textual = X_textual
        self.X_numerical = X_numerical
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_textual[idx], self.X_numerical[idx], self.y[idx]

# Load and preprocess the dataset
raw_data = pd.read_csv("./merged_gaqcorpus.csv")
cols = ["cogency_(logic)", "effectiveness_(rhetoric)", "reasonableness_(dialectic)", "is_argument"]

# Filtering out rows containing '#' in any column
raw_data = raw_data[~raw_data.apply(lambda row: row.astype(str).str.contains("#")).any(axis=1)]

# Convert the "is_argument" column to string
raw_data["is_argument"] = raw_data["is_argument"].astype(str).str.replace("#", "0").replace("nan", "0").astype(float)
raw_data["is_argument"] = raw_data["is_argument"].fillna(0)

# Extract features and target
X_textual = raw_data[["text", "title"]].copy()
X_numerical = raw_data[["cogency_(logic)", "effectiveness_(rhetoric)", "reasonableness_(dialectic)"]].copy()
X_numerical = X_numerical.apply(pd.to_numeric)
y = raw_data['is_argument'].copy()

# Preprocessing functions
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatization and POS tagging
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    pos_tags = []
    for token, tag in pos_tag(filtered_tokens):
        pos_tags.append(tag)
        if tag.startswith('N'):
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='n'))
        elif tag.startswith('V'):
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='v'))
        elif tag.startswith('J'):
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='a'))
        elif tag.startswith('R'):
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='r'))
        else:
            lemmatized_tokens.append(token)  # Assume as default
    return lemmatized_tokens, pos_tags

# Preprocess textual features
X_textual_preprocessed = []
X_pos_tags = []
for text in tqdm(X_textual['text']):
    preprocessed_text, pos_tags = preprocess_text(text)
    X_textual_preprocessed.append(preprocessed_text)
    X_pos_tags.append(pos_tags)

# Obtain Sentence-BERT embeddings for preprocessed text
X_textual_embeddings = sentence_bert_model.encode([' '.join(tokens) for tokens in X_textual_preprocessed])

# Convert arrays to PyTorch tensors
X_textual_tensor = torch.tensor(X_textual_embeddings, dtype=torch.float32)
X_numerical_tensor = torch.tensor(X_numerical.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define hyperparameters
num_folds = 5
num_epochs = 10
batch_size = 64
learning_rate = 0.001
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits

# Define evaluation metrics
def evaluate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return accuracy, precision, recall, f1

# Perform 5-fold cross-validation
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold = 0
results = []

for feature_to_exclude in ["cogency_(logic)", "effectiveness_(rhetoric)", "reasonableness_(dialectic)"]:
    fold_results = {"Feature Excluded": feature_to_exclude}
    for train_index, test_index in skf.split(X_textual_tensor, y):
        fold += 1
        print(f"Fold {fold}:")

        # Split data into train and test sets for this fold
        X_textual_train, X_textual_test = X_textual_tensor[train_index], X_textual_tensor[test_index]
        X_numerical_train, X_numerical_test = X_numerical_tensor[train_index], X_numerical_tensor[test_index]
        y_train, y_test = y_tensor[train_index], y_tensor[test_index]

        # Exclude the selected feature
        if feature_to_exclude == "cogency_(logic)":
            X_numerical_train = X_numerical_train[:, 1:]  # Exclude cogency_(logic)
            X_numerical_test = X_numerical_test[:, 1:]
        elif feature_to_exclude == "effectiveness_(rhetoric)":
            X_numerical_train = X_numerical_train[:, [0, 2]]  # Exclude effectiveness_(rhetoric)
            X_numerical_test = X_numerical_test[:, [0, 2]]
        elif feature_to_exclude == "reasonableness_(dialectic)":
            X_numerical_train = X_numerical_train[:, :-1]  # Exclude reasonableness_(dialectic)
            X_numerical_test = X_numerical_test[:, :-1]

        # Create DataLoader for train set
        train_dataset = CustomDataset(X_textual_train, X_numerical_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the model
        model = BiLSTM(embedding_dim=X_textual_train.shape[1], hidden_dim=64, num_layers=9, num_numerical_features=X_numerical_train.shape[1])

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            for textual, numerical, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(textual, numerical)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            outputs = model(X_textual_test, X_numerical_test)
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Threshold at 0.5 for binary classification

            # Calculate evaluation metrics
            true_labels = y_test.numpy()
            predicted_labels = predicted.squeeze().numpy()
            accuracy, precision, recall, f1 = evaluate_metrics(true_labels, predicted_labels)

            # Save results
            fold_results[f'Fold {fold}'] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}

    results.append(fold_results)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV file
results_df.to_csv('model_results.csv', index=False)
