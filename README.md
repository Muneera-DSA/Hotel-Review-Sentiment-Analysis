# 🌟 Hotel Review Sentiment Analysis
An end-to-end NLP project for hotel review sentiment classification using TF-IDF and Logistic Regression.

## 📌 Overview
This project builds a sentiment analysis model for hotel reviews using Natural Language Processing (NLP) and machine learning.
It classifies customer reviews as Positive or Negative, helping hospitality businesses understand guest satisfaction at scale.

### The project includes:
- Text preprocessing
- Feature engineering
- Model training & evaluation
- Exported model for reuse
- Clean, modular code structure

---

## 📂 Project Structure
hotel-sentiment-analysis/
│── data/
│   └── hotel_reviews.csv
│── src/
│   ├── preprocess.py
│   └── train.py
│── notebooks/
│   └── exploration.ipynb
│── README.md
│── requirements.txt
│── .gitignore

---

## 📊 Dataset
Kaggle Dataset: **515K Hotel Reviews Data in Europe**  
🔗 https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe

After downloading, place the CSV file here:
data/hotel_reviews.csv

---

## 🧠 Features Implemented
- Text cleaning (lowercasing, punctuation removal, stopwords, lemmatization)
- TF‑IDF vectorization
- Logistic Regression classifier
- Train/test split
- Evaluation using Accuracy, Precision, Recall, F1‑Score
- Confusion Matrix
- Exported model + vectorizer (.pkl)

---

## 🚀 How to Run

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Train the model
python src/train.py

### 3️⃣ Output
- Classification report  
- Confusion matrix  
- Saved model files:
  - `sentiment_model.pkl`
  - `vectorizer.pkl`

---

## 📈 Results
The model achieves **~85–90% accuracy**, depending on preprocessing and TF‑IDF parameters.

---

## 🛠 Tech Stack
- Python  
- Scikit‑Learn  
- NLTK  
- Pandas  
- NumPy  
- Joblib  

---

## 👤 Author
**Muneera Mohamed**  
MSc Data Science | NLP & Analytics Enthusiast  
GitHub: *add your link here*
