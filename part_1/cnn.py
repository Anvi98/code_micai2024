import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Load and preprocess the dataset
raw_data = pd.read_csv("./merged_gaqcorpus.csv")
cols = ["cogency_(logic)", "effectiveness_(rhetoric)", "reasonableness_(dialectic)", "is_argument"]

# Filtering out rows containing '#' in any column
raw_data = raw_data[~raw_data.apply(lambda row: row.astype(str).str.contains("#")).any(axis=1)]

# Convert the "is_argument" column to string
raw_data["is_argument"] = raw_data["is_argument"].astype(str).str.replace("#", "0").replace("nan", "0").astype(float)
raw_data["is_argument"] = raw_data["is_argument"].fillna(0)

# Extract features and target
X_numerical = raw_data[["cogency_(logic)", "effectiveness_(rhetoric)", "reasonableness_(dialectic)"]].copy()
X_numerical = X_numerical.apply(pd.to_numeric)
y = raw_data['is_argument'].copy()

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X_numerical, y):
        self.X_numerical = X_numerical
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_numerical[idx], self.y[idx]

# Convert arrays to PyTorch tensors
X_numerical_tensor = torch.tensor(X_numerical.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define CNN model
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, numerical_input):
        x = self.fc1(numerical_input)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define hyperparameters
num_epochs = 10
learning_rate = 0.001
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits

# Initialize the model
model = CNN(num_features=X_numerical_tensor.shape[1])

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    for i in range(len(X_numerical_tensor)):
        optimizer.zero_grad()
        numerical = X_numerical_tensor[i].unsqueeze(0)
        label = y_tensor[i].reshape(-1, 1)
        output = model(numerical)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = []
    for i in range(len(X_numerical_tensor)):
        numerical = X_numerical_tensor[i].unsqueeze(0)
        output = model(numerical)
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
results_df.to_csv('model_results_numerical.csv', index=False)

