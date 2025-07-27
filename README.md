
# 📩 SMS/Email Spam Classifier using Machine Learning

## 🧠 Overview
This project focuses on building an **intelligent spam classifier** for SMS or email messages using machine learning. The goal is to detect whether a given message is **Spam (Unwanted)** or **Not Spam (Ham)** based on its content.  
We use **Natural Language Processing (NLP)** techniques to preprocess text and then classify it using a **Multinomial Naive Bayes** model, with an interactive frontend built using **Streamlit**.

---

## 🚀 Features
- Cleaned and preprocessed textual data
- Text tokenization, stopword removal, and stemming
- Feature extraction using TF-IDF vectorizer
- Trained model using Naive Bayes algorithm
- Built and deployed using Streamlit for real-time spam detection
- Lightweight and fast

---

## ❓ Problem Statement
With the increasing use of SMS and email communication, **spam messages** have become a widespread problem. They not only clutter users' inboxes but also pose risks such as phishing and fraud.  
This project aims to **automate spam detection** by training a classifier that can predict whether a given message is spam or not, thus reducing manual filtering and improving user experience.

---

## 📂 Dataset
**File Used:** `spam.csv`  
**Source:** Public UCI/Kaggle SMS Spam Dataset  
**Shape:** ~5,500 messages with 2 main columns:
- **`v1`**: Class label – `ham` (not spam) or `spam`
- **`v2`**: Message content (text)

We renamed:
- `v1` → `target`
- `v2` → `text`

---

## 🧪 Methodology
1. **Data Cleaning**: Remove unnecessary columns, handle duplicates, and rename labels
2. **Label Encoding**: Convert `ham/spam` to `0/1`
3. **Text Preprocessing**:
   - Lowercasing
   - Tokenization
   - Removing stopwords and punctuation
   - Stemming
4. **Vectorization**: Convert text to numerical format using TF-IDF
5. **Model Training**: Train a `Multinomial Naive Bayes` classifier
6. **Model Evaluation**: Use accuracy, precision, and confusion matrix
7. **Model Deployment**: Use `pickle` to save model and vectorizer and build a `Streamlit` web app

---

## 🗂 Project Structure
```
📦 spam-classifier
├── app.py                  # Streamlit frontend
├── spam.csv                # Dataset
├── model.pkl               # Trained classifier
├── vectorizer.pkl          # Trained TF-IDF vectorizer
├── README.md               # Project documentation
└── notebook.ipynb          # Training and experimentation notebook
```

---

## 🪜 Steps Performed
1. Import libraries
2. Load and clean data
3. Preprocess and transform text
4. Exploratory Data Analysis (EDA)
5. Feature extraction using TF-IDF
6. Train/test split
7. Train Multinomial Naive Bayes model
8. Evaluate model
9. Save model and vectorizer
10. Build Streamlit interface for prediction

---

## 📊 Results
- The trained model achieved high accuracy and precision in detecting spam.
- Real-time prediction available via the Streamlit app.
- Lightweight and fast-performing interface.

---

## 📈 Model Evaluation
| Metric     | Value (Example) |
|------------|-----------------|
| Accuracy   | ~97%            |
| Precision  | ~95%            |
| Confusion Matrix | Used to analyze FP/FN |

*Note: You can modify these values based on actual evaluation results.*

---

## 📘 Terminology

| Term | Description |
|------|-------------|
| **Spam** | Unwanted or unsolicited messages, usually advertisements or malicious content |
| **Ham** | Legitimate or useful messages (not spam) |
| **Tokenization** | Splitting text into smaller units like words or sentences |
| **Stopwords** | Common words like "is", "the", "and" that are removed because they carry little useful info |
| **Stemming** | Reducing words to their root form (e.g., "loved" → "love") |
| **TF-IDF (Term Frequency–Inverse Document Frequency)** | A statistical measure to evaluate how important a word is in a document |
| **Vectorizer** | A tool that converts text into numeric format (vectors) for ML models |
| **Naive Bayes Classifier** | A probabilistic ML model based on Bayes’ theorem used for text classification |
| **Confusion Matrix** | Table used to describe performance of a classifier (True Positives, False Positives, etc.) |
| **Precision** | Percentage of predicted spams that are actually spam |
| **Accuracy** | Overall correctness of the model’s predictions |
| **Streamlit** | A Python framework to build interactive data science web apps |
| **Pickle** | A Python module for serializing and saving objects (like models) |

---

## 🙏 Acknowledgements
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Streamlit](https://streamlit.io/)
- Dataset Providers (UCI/Kaggle)
- OpenAI ChatGPT for guidance and formatting
