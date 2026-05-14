import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("../dataset/processed_news.csv")
df["clean_text"] = df["clean_text"].fillna("")

X = df["clean_text"]
y = df["label"]

vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")

print(classification_report(y_test, y_pred))

joblib.dump(model, "../models/fake_news_model.pkl")

joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")

