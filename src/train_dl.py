import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from collections import Counter

import joblib

df = pd.read_csv("../dataset/processed_news.csv")

# Remove null values
df.dropna(subset=["clean_text"], inplace=True)

# Convert text column to string
df["clean_text"] = df["clean_text"].astype(str)

# Balance dataset
fake_samples = df[df["label"] == 0].sample(5000, random_state=42)
real_samples = df[df["label"] == 1].sample(5000, random_state=42)

df = pd.concat([fake_samples, real_samples])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Texts and labels
texts = df["clean_text"].tolist()
labels = df["label"].tolist()

# Tokenizer
def tokenize(text):
    return text.split()

# Build vocabulary
counter = Counter()

for text in texts:
    counter.update(tokenize(text))

# Larger vocabulary
vocab = {
    word: idx + 1
    for idx, (word, _) in enumerate(counter.most_common(15000))
}

# Maximum sequence length
max_len = 200

# Encode text
def encode(text):

    tokens = tokenize(text)

    encoded = [
        vocab.get(word, 0)
        for word in tokens
    ]

    # Truncate
    encoded = encoded[:max_len]

    # Padding
    encoded += [0] * (max_len - len(encoded))

    return encoded

# Encode all texts
X = [encode(text) for text in texts]
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Neural Network
class FakeNewsClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim):

        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(embed_dim, 256)

        self.relu1 = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)

        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Word embeddings
        embedded = self.embedding(x)

        # Average embeddings
        embedded = embedded.mean(dim=1)

        # Neural network
        x = self.fc1(embedded)

        x = self.relu1(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.relu2(x)

        x = self.fc3(x)

        return self.sigmoid(x)

# Model setup
vocab_size = len(vocab) + 1
embed_dim = 128

model = FakeNewsClassifier(vocab_size, embed_dim)

# Loss function
criterion = nn.BCELoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training loop
epochs = 15

for epoch in range(epochs):

    model.train()

    # Forward pass
    outputs = model(X_train).squeeze()

    loss = criterion(outputs, y_train)

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # Accuracy during training
    predictions = (outputs >= 0.5).float()

    train_accuracy = accuracy_score(
        y_train.detach().numpy(),
        predictions.detach().numpy()
    )

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {loss.item():.4f} | "
        f"Training Accuracy: {train_accuracy * 100:.2f}%"
    )

# Evaluation
model.eval()

with torch.no_grad():

    test_outputs = model(X_test).squeeze()

    test_predictions = (test_outputs >= 0.5).float()

test_accuracy = accuracy_score(
    y_test.numpy(),
    test_predictions.numpy()
)

print(f"\n✅ Deep Learning Model Accuracy: {test_accuracy * 100:.2f}%")

# Save model
torch.save(
    model.state_dict(),
    "../models/fake_news_dl_model.pt"
)

# Save vocabulary
joblib.dump(
    vocab,
    "../models/vocab.pkl"
)

print("✅ Deep Learning model saved successfully!")