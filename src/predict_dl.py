import torch
import torch.nn as nn
import joblib

# Load vocab
vocab = joblib.load("../models/vocab.pkl")

# Tokenizer
def tokenize(text):
    return text.lower().split()

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

# Load model
vocab_size = len(vocab) + 1
embed_dim = 64

model = FakeNewsClassifier(vocab_size, embed_dim)

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

# Result
if confidence >= 0.5:
    print(f"\n✅ REAL NEWS")
    print(f"Confidence Score: {confidence * 100:.2f}%")
else:
    print(f"\n🚨 FAKE NEWS DETECTED")
    print(f"Confidence Score: {(1 - confidence) * 100:.2f}%")