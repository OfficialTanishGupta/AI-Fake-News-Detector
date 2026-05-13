import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load datasets
fake_df = pd.read_csv("../dataset/Fake.csv")
true_df = pd.read_csv("../dataset/True.csv")

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0)

# Keep useful columns
df = df[["text", "label"]]

# Remove null values
df.dropna(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

# Apply preprocessing
df["clean_text"] = df["text"].apply(clean_text)

# Save processed dataset
df.to_csv("../dataset/processed_news.csv", index=False)

print("✅ Preprocessing completed successfully!")
print(df.head())