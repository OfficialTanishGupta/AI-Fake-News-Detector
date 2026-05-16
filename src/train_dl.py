import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import re
import joblib

# Load dataset
df = pd.read_csv("../dataset/processed_news.csv")
df = df.dropna(subset=["clean_text"])


# Use smaller dataset for faster training initially
df = df.sample(10000, random_state=42)

texts = df["clean_text"].tolist()
labels = df["label"].tolist()

# Tokenizer
def tokenize(text):
    return text.split()

# Build vocabulary
counter = Counter()

for text in texts:
    counter.update(tokenize(text))

vocab = {
    word: idx + 1
    for idx, (word, _) in enumerate(counter.most_common(5000))
}

# Encode text
max_len = 100

def encode(text):
    tokens = tokenize(text)

    encoded = [
        vocab.get(word, 0)
        for word in tokens
    ]

    encoded = encoded[:max_len]

    encoded += [0] * (max_len - len(encoded))

    return encoded

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

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.fc1 = nn.Linear(embed_dim, 128)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)

        embedded = embedded.mean(dim=1)

        x = self.fc1(embedded)

        x = self.relu(x)

        x = self.fc2(x)

        return self.sigmoid(x)

# Model setup
vocab_size = len(vocab) + 1
embed_dim = 64

model = FakeNewsClassifier(vocab_size, embed_dim)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5

for epoch in range(epochs):

    model.train()

    outputs = model(X_train).squeeze()

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()

with torch.no_grad():

    predictions = model(X_test).squeeze()

    predictions = (predictions >= 0.5).float()

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\n✅ Deep Learning Model Accuracy: {accuracy * 100:.2f}%")

# Save model
torch.save(model.state_dict(), "../models/fake_news_dl_model.pt")

# Save vocab
joblib.dump(vocab, "../models/vocab.pkl")

print("✅ Deep Learning model saved successfully!")