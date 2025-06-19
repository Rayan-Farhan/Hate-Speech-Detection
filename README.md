# Hate-Speech-Detection

This project focuses on detecting hate speech, offensive language, and neutral content in tweets using both traditional machine learning models and transformer-based NLP models.

## Project Overview

We use a labeled dataset of tweets (`labeled_data.csv`) containing three classes:
- `0` → Hate Speech  
- `1` → Offensive Language  
- `2` → Neither (Neutral)

The objective was to build a robust multi-class text classification system capable of identifying toxic content in social media text.

## Exploratory Data Analysis (EDA)

The dataset is explored using:
- **Class distribution visualizations** with Seaborn
- **Word cloud generation** to understand common terms per class
- **Basic text statistics** like tweet length and frequency

These insights guided preprocessing and model selection.

## Text Preprocessing

The preprocessing pipeline includes:
- Lowercasing
- Removing URLs, mentions, hashtags, punctuation, and numbers
- Tokenization and cleanup using regular expressions

The cleaned text is then transformed using **TF-IDF vectorization** with both unigrams and bigrams.

## Machine Learning Models

I trained and evaluated the following traditional ML models:
- **Logistic Regression** with `class_weight='balanced'`
- **Random Forest Classifier**
- **Support Vector Machine (LinearSVC)**
- **Multinomial Naive Bayes**

Each model was evaluated using:
- Accuracy
- Precision, Recall, F1-score (via `classification_report`)

### Results
- **Random Forest** achieved the highest accuracy (~90%) and performed well across all classes after class balancing.
- **SVM** and **Logistic Regression** achieved around ~85% accuracy.

## Transformer-based Comparison

To benchmark against modern NLP models, I used:
- `cardiffnlp/twitter-roberta-base-sentiment` from Hugging Face 🤗 Transformers

I built a sentiment analysis pipeline and tested it on the same tweets to observe prediction behavior and compare label confidence.

## Tools & Libraries

- **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**  
- **Scikit-learn** – for TF-IDF vectorization, machine learning models, and evaluation metrics  
- **Hugging Face Transformers** – used for sentiment analysis with `pipeline` and `AutoModelForSequenceClassification`  
- **WordCloud** – for visualizing most frequent words per class  
- **Regex** – for text preprocessing and cleaning  

---

## Dataset

The dataset used in this project is provided as `labeled_data.csv`. Here's the link: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
