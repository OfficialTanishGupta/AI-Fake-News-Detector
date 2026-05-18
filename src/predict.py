import joblib

# Load model
model = joblib.load(
    "../models/fake_news_model.pkl"
)

# Load vectorizer
vectorizer = joblib.load(
    "../models/tfidf_vectorizer.pkl"
)

# User input
news = input("Enter news text: ")

# Transform text
news_vectorized = vectorizer.transform([news])

# Prediction probabilities
probabilities = model.predict_proba(news_vectorized)[0]

fake_prob = probabilities[0]
real_prob = probabilities[1]

# Smarter threshold
threshold = 0.60

# Output
if real_prob >= threshold:

    print("\n✅ REAL NEWS")

    print(f"Confidence Score: {real_prob * 100:.2f}%")

elif fake_prob >= threshold:

    print("\n🚨 FAKE NEWS DETECTED")

    print(f"Confidence Score: {fake_prob * 100:.2f}%")

else:

    print("\n⚠️ UNCERTAIN RESULT")

    print(
        f"Fake Probability: {fake_prob * 100:.2f}%"
    )

    print(
        f"Real Probability: {real_prob * 100:.2f}%"
    )