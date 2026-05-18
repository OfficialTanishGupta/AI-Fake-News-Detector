import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("../dataset/processed_news.csv")

# Remove null values
df.dropna(subset=["clean_text"], inplace=True)

# Convert to string
# Balance dataset
fake_df = df[df["label"] == 0].sample(15000, random_state=42)

real_df = df[df["label"] == 1].sample(15000, random_state=42)

df = pd.concat([fake_df, real_df])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X = df["clean_text"]
y = df["label"]

# Better TF-IDF
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Strong classifier
base_model = LinearSVC(
    class_weight="balanced"
)

# Add probability support
model = CalibratedClassifierCV(base_model)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")

# Report
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(
    model,
    "../models/fake_news_model.pkl"
)

# Save vectorizer
joblib.dump(
    vectorizer,
    "../models/tfidf_vectorizer.pkl"
)

print("✅ Model and vectorizer saved successfully!")