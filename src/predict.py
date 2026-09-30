import os
import joblib
import sys

# Allow Python to find preprocess.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text


# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Model locations
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "vectorizer.pkl"
)


# Load trained model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(review):
    """Predict whether a hotel review is positive or negative."""

    cleaned_review = clean_text(review)

    review_vector = vectorizer.transform(
        [cleaned_review]
    )

    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":

    review = input("Enter a hotel review: ")

    result = predict_sentiment(review)

    print(f"\nPredicted Sentiment: {result}")