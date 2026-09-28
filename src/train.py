import sys
sys.path.append("../src")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocess import clean_text

df = pd.read_csv("data/hotel_reviews.csv")

df = df[['Review', 'Rating']].dropna()
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x > 3 else 0)
df['Cleaned'] = df['Review'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['Cleaned'], df['Sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model training complete. Files saved.")
