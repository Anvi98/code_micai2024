import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

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

# Define hyperparameters
num_folds = 5
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Initialize Sentence-BERT model
from sentence_transformers import SentenceTransformer
sentence_bert_model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')

# Preprocess textual features using Sentence-BERT
def preprocess_text(texts):
    preprocessed_texts = []
    for text in tqdm(texts):
        # Tokenization and embedding
        embeddings = sentence_bert_model.encode(text)
        preprocessed_texts.append(embeddings)
    return preprocessed_texts

# Define evaluation metrics
def evaluate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return accuracy, precision, recall, f1

# Function to train and evaluate model
def train_and_evaluate(X_textual_train, X_numerical_train, y_train, X_textual_test, X_numerical_test, y_test):
    # Convert arrays to PyTorch tensors
    X_textual_tensor_train = torch.tensor(X_textual_train, dtype=torch.float32)
    X_numerical_tensor_train = torch.tensor(X_numerical_train.values, dtype=torch.float32)
    y_tensor_train = torch.tensor(y_train.values, dtype=torch.float32)

    X_textual_tensor_test = torch.tensor(X_textual_test, dtype=torch.float32)
    X_numerical_tensor_test = torch.tensor(X_numerical_test.values, dtype=torch.float32)
    y_tensor_test = torch.tensor(y_test.values, dtype=torch.float32)

    # Initialize the model
    model = BiLSTM(embedding_dim=X_textual_tensor_train.shape[1], hidden_dim=64, num_layers=9, num_numerical_features=X_numerical_tensor_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for train set
    train_dataset = CustomDataset(X_textual_tensor_train, X_numerical_tensor_train, y_tensor_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
        outputs = model(X_textual_tensor_test, X_numerical_tensor_test)
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        # Calculate evaluation metrics
        true_labels = y_tensor_test.numpy()
        predicted_labels = predicted.squeeze().numpy()
        accuracy, precision, recall, f1 = evaluate_metrics(true_labels, predicted_labels)

        return accuracy, precision, recall, f1

# Perform experiments for each feature set
results = []
feature_sets = ["cogency_(logic)", "effectiveness_(rhetoric)", "reasonableness_(dialectic)"]
for feature_set in feature_sets:
    print(f"Experiment with feature set: {feature_set}")
    
    # Extract features for the current experiment
    X_numerical_experiment = X_numerical[[feature_set]]
    
    # Split the dataset into train and test sets
    X_textual_train, X_textual_test, X_numerical_train, X_numerical_test, y_train, y_test = train_test_split(
        preprocess_text(X_textual['text']), X_numerical_experiment, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    accuracy, precision, recall, f1 = train_and_evaluate(X_textual_train, X_numerical_train, y_train, X_textual_test, X_numerical_test, y_test)
    
    # Store results
    results.append({
        'Experiment': feature_set,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('ablation_results.csv', index=False)
