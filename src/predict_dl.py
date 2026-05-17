import torch
import torch.nn as nn
import joblib

# Load vocab
vocab = joblib.load("../models/vocab.pkl")

# Tokenizer
def tokenize(text):
    return text.lower().split()

# Max sequence length
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

        embedded = self.embedding(x)

        embedded = embedded.mean(dim=1)

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

# Load trained weights
model.load_state_dict(
    torch.load("../models/fake_news_dl_model.pt")
)

model.eval()

# User input
news = input("Enter news text: ")

# Encode input
encoded = encode(news)

input_tensor = torch.tensor(
    [encoded],
    dtype=torch.long
)

# Prediction
with torch.no_grad():

    output = model(input_tensor)

    confidence = output.item()

# Output result
if confidence >= 0.5:

    print("\n✅ REAL NEWS")

    print(f"Confidence Score: {confidence * 100:.2f}%")

else:

    print("\n🚨 FAKE NEWS DETECTED")

    print(f"Confidence Score: {(1 - confidence) * 100:.2f}%")