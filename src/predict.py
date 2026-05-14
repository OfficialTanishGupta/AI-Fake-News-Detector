import joblib

# Load model and vectorizer
model = joblib.load("../models/fake_news_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# Test input
news = input("Enter news text: ")

# Vectorize
news_vectorized = vectorizer.transform([news])

# Prediction
prediction = model.predict(news_vectorized)[0]

# Confidence score
confidence = model.predict_proba(news_vectorized).max()

# Output
if prediction == 0:
    print(f"\n🚨 FAKE NEWS DETECTED")
else:
    print(f"\n✅ REAL NEWS")

print(f"Confidence Score: {confidence * 100:.2f}%")