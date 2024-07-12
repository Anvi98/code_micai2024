import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load data
raw_data = pd.read_csv("./merged_gaqcorpus.csv")

# Filtering out rows containing '#' in any column
raw_data = raw_data[~raw_data.apply(lambda row: row.astype(str).str.contains("#")).any(axis=1)]

# Convert the "is_argument" column to string and replace '#', 'nan' with '0', then convert to float
raw_data["is_argument"] = raw_data["is_argument"].astype(str).str.replace("#", "0").replace("nan", "0").astype(float)
raw_data["is_argument"].fillna(0, inplace=True)

# Extract features and target
X_textual = raw_data[["text", "title"]].copy()
y = raw_data['is_argument'].copy()

# Sentence-BERT embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
X_textual_embeddings = model.encode(X_textual['text'].tolist())

# Convert data to PyTorch tensors
X_textual_embeddings = torch.tensor(X_textual_embeddings, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

# Splitting data into train and test sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_textual_embeddings, y, test_size=0.2, random_state=42)

# Custom dataset for PyTorch
class CustomDataset(Dataset):
    def __init__(self, X_text, y):
        self.X_text = X_text
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_text[idx], self.y[idx]

train_dataset = CustomDataset(X_train_text, y_train)
test_dataset = CustomDataset(X_test_text, y_test)

# Model architecture
class CNN(nn.Module):
    def __init__(self, text_input_dim):
        super(CNN, self).__init__()
        self.text_conv1d = nn.Conv1d(in_channels=text_input_dim, out_channels=128, kernel_size=3)  # Changed kernel size
        self.text_maxpool = nn.MaxPool1d(kernel_size=2)  # Adjusted max-pooling layer
        self.text_fc = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x_text):
        # Apply Conv1d operation
        x_text = x_text.permute(0, 2, 1)  # Permute for Conv1d 
        x_text = self.text_conv1d(x_text)
        x_text = torch.relu(x_text)
        # Calculate output size after convolutional layer
        conv_output_size = x_text.size(2)
        x_text = self.text_maxpool(x_text)
        # Adjust pooling kernel size based on conv output size
        pool_kernel_size = min(2, conv_output_size)  # Pooling kernel cannot be larger than input size
        x_text = nn.functional.max_pool1d(x_text, kernel_size=pool_kernel_size)
        x_text = torch.squeeze(x_text, dim=2)
        x_text = self.text_fc(x_text)
        x_text = self.dropout(x_text)
        output = torch.sigmoid(self.fc_out(x_text))
        return output

# Instantiate model and define loss and optimizer
model = CNN(text_input_dim=X_textual_embeddings.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs_text, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_text)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train(model, train_loader, criterion, optimizer)

# Evaluate model
def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs_text, labels in test_loader:
            outputs = model(inputs_text)
            predicted = torch.round(outputs)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    return y_true, y_pred

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
y_true, y_pred = evaluate(model, test_loader)
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate metrics
accuracy = np.mean(y_true == y_pred)
f1_score = classification_report(y_true, y_pred, target_names=['non-argument', 'argument'])

print("Accuracy:", accuracy)
print("Classification Report:")
print(f1_score)

