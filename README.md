# 🌟 Hotel Review Sentiment Analysis
An end-to-end NLP project for hotel review sentiment classification using TF-IDF and Logistic Regression.

## 📌 Overview
This project builds a sentiment analysis model for hotel reviews using Natural Language Processing (NLP) and machine learning.
It classifies hotel review text as **Positive** or **Negative**, helping analyze customer feedback and understand guest sentiment at scale.

### The project includes:
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Model training and evaluation
- Sentiment prediction on new reviews
- Modular and reusable Python c

---

## 📂 Project Structure

Hotel-Review-Sentiment-Analysis/
│
├── data/
│   └── README.md
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
---

## 📊 Dataset
Kaggle Dataset: **515K Hotel Reviews Data in Europe**  
🔗 https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe

After downloading, place the CSV file here:
data/Hotel_Reviews.csv


🧠 Methodology
1. Text Preprocessing

The review text is cleaned using:

Lowercasing
Removal of non-alphabetic characters
Stopword removal
Lemmatization
2. TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts review text into numerical features.

The model uses:

Maximum features: 10,000
N-gram range: (1, 2)

This means the model considers both individual words and two-word phrases.

3. Train/Test Split

The dataset is divided into:

80% training data
20% testing data

A stratified split is used to maintain the class distribution between training and testing sets.

4. Classification Model

A Logistic Regression classifier is trained on the TF-IDF features.

Logistic Regression was selected because it is effective for binary classification problems involving high-dimensional sparse text features.

📈 Model Performance

The final model achieved approximately:

<img width="566" height="297" alt="image" src="https://github.com/user-attachments/assets/5f0988cb-25dc-4334-a5cc-35a290cc346c" />


The relatively balanced precision, recall, and F1-scores across both classes indicate that the model performs consistently on positive and negative reviews.

🚀 How to Run

1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Download the dataset

Download the hotel reviews dataset and place:

Hotel_Reviews.csv

in the project root directory.

3️⃣ Train the model

From the project root:

python src/train.py

The training script will:

Load the dataset
Prepare positive and negative reviews
Clean the text
Create TF-IDF features
Train the Logistic Regression model
Display the classification report
Display the confusion matrix
Save the trained model and vectorizer
4️⃣ Make a prediction

Run:

python src/predict.py

Enter a hotel review when prompted.

Example:

Enter a hotel review: The room was clean and the staff were very helpful.

Predicted Sentiment: Positive
🛠 Tech Stack
Python
Pandas
Scikit-Learn
NLTK
TF-IDF
Logistic Regression
Joblib
🔍 Key Learning Outcomes

This project demonstrates practical experience with:

Natural Language Processing
Text preprocessing
Feature engineering
TF-IDF vectorization
Binary classification
Handling class imbalance
Model evaluation
Confusion matrix analysis
Model serialization
Building a reusable prediction pipeline



## 👤 Author
**Muneera Mohamed**  
MSc Data Science | NLP & Analytics Enthusiast  
[GitHub: *add your link here*](https://github.com/Muneera-DSA)
