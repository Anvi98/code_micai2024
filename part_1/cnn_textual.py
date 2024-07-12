import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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
y = raw_data['is_argument'].copy()

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X_textual, y):
        self.X_textual = X_textual
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_textual[idx], self.y[idx]

# Preprocessing functions
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

# Preprocess textual features
X_textual_preprocessed = []
for text in tqdm(X_textual['text']):
    preprocessed_text = preprocess_text(text)
    X_textual_preprocessed.append(preprocessed_text)

# Load Sentence-BERT model
sentence_bert_model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')

# Obtain Sentence-BERT embeddings for preprocessed text
X_textual_embeddings = sentence_bert_model.encode([' '.join(tokens) for tokens in X_textual_preprocessed])

# Convert arrays to PyTorch tensors
X_textual_tensor = torch.tensor(X_textual_embeddings, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define CNN model
class CNN(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, textual_input):
        x = self.fc1(textual_input)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define hyperparameters
num_epochs = 10
learning_rate = 0.001
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits

# Initialize the model
model = CNN(embedding_dim=X_textual_tensor.shape[1])

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    for i in range(len(X_textual_tensor)):
        optimizer.zero_grad()
        textual = X_textual_tensor[i].unsqueeze(0)
        label = y_tensor[i].reshape(-1, 1)
        output = model(textual)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = []
    for i in range(len(X_textual_tensor)):
        textual = X_textual_tensor[i].unsqueeze(0)
        output = model(textual)
        outputs.append(output.squeeze().item())

# Convert outputs to predictions
predictions = [1 if output > 0.5 else 0 for output in outputs]

# Calculate evaluation metrics
true_labels = y_tensor.numpy().astype(int)
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Print evaluation metrics
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, predictions))

# Save results to CSV file
results_df = pd.DataFrame({'True Labels': true_labels, 'Predictions': predictions})
results_df.to_csv('model_results_textual.csv', index=False)
