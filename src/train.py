import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Allow importing preprocess.py from the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Hotel_Reviews.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "vectorizer.pkl"
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")


# --------------------------------------------------
# Prepare positive and negative reviews
# --------------------------------------------------

df["Positive_Review"] = df["Positive_Review"].replace(
    "No Positive", ""
)

df["Negative_Review"] = df["Negative_Review"].replace(
    "No Negative", ""
)


positive = df[["Positive_Review"]].copy()
positive.columns = ["Review"]
positive["Sentiment"] = 1

negative = df[["Negative_Review"]].copy()
negative.columns = ["Review"]
negative["Sentiment"] = 0


# Remove empty reviews
positive = positive[
    positive["Review"].str.strip() != ""
]

negative = negative[
    negative["Review"].str.strip() != ""
]


# Combine datasets
sentiment_df = pd.concat(
    [positive, negative],
    ignore_index=True
)

# Shuffle
sentiment_df = sentiment_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print(f"Prepared dataset shape: {sentiment_df.shape}")
print("\nClass distribution:")
print(sentiment_df["Sentiment"].value_counts())


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

print("\nCleaning reviews...")

sentiment_df["Cleaned"] = sentiment_df["Review"].apply(
    clean_text
)

# Remove empty reviews after cleaning
sentiment_df = sentiment_df[
    sentiment_df["Cleaned"].str.strip() != ""
]


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    sentiment_df["Cleaned"],
    sentiment_df["Sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=sentiment_df["Sentiment"]
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Training matrix: {X_train_vec.shape}")
print(f"Testing matrix: {X_test_vec.shape}")


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=300
)

model.fit(X_train_vec, y_train)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

preds = model.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))


# --------------------------------------------------
# Save model and vectorizer
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nVectorizer saved to:")
print(VECTORIZER_PATH)

print("\nTraining complete!")