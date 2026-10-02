import os
import urllib.request
import zipfile
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Download NLTK data resources
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------
# 1. DATA ACQUISITION & LOADING
# ---------------------------------------------------------
def load_data():
    """
    Downloads and extracts the UCI SMS Spam Collection dataset.
    """
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam.zip"
    data_path = "SMSSpamCollection"

    if not os.path.exists(data_path):
        print("Downloading dataset from UCI Machine Learning Repository...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Dataset extracted successfully.")

    # Load dataset into pandas dataframe
    df = pd.read_csv(data_path, sep='\t', names=['label', 'text'])
    print(f"Dataset Loaded: {df.shape[0]} samples.")
    return df

# ---------------------------------------------------------
# 2. TEXT PREPROCESSING
# ---------------------------------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans text by converting to lowercase, removing punctuation/numbers,
    tokenizing, removing stop words, and applying stemming.
    """
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords + apply stemming
    words = text.split()
    cleaned_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# ---------------------------------------------------------
# 3. MAIN WORKFLOW
# ---------------------------------------------------------
def main():
    # Step 1: Load Data
    df = load_data()
    
    # Step 2: Preprocess Data
    print("\n[INFO] Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Encode target labels (ham = 0, spam = 1)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['cleaned_text']
    y = df['label_num']

    # Step 3: Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Step 4: TF-IDF Feature Extraction
    print("[INFO] Extracting TF-IDF Features...")
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Step 5: Model Selection and Training
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Machine': SVC(kernel='linear')
    }

    results = {}
    print("\n================ MODEL EVALUATION ================")
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")

    # Plot Confusion Matrix for Best Model (Naive Bayes)
    best_model = models['Multinomial Naive Bayes']
    y_pred_best = best_model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix - Multinomial Naive Bayes')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n[INFO] Confusion matrix saved as 'confusion_matrix.png'.")

    # ---------------------------------------------------------
    # 6. CUSTOM INFERENCE / SAMPLE PREDICTION
    # ---------------------------------------------------------
    print("\n================ CUSTOM PREDICTIONS ================")
    sample_messages = [
        "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch near the campus library today?",
        "URGENT: Your mobile bank account security is compromised. Reply instantly with PIN to verify."
    ]

    for msg in sample_messages:
        cleaned_msg = preprocess_text(msg)
        vec_msg = tfidf.transform([cleaned_msg])
        prediction = best_model.predict(vec_msg)[0]
        label = "SPAM" if prediction == 1 else "HAM (Legitimate)"
        print(f"Message: \"{msg}\"")
        print(f"Prediction: {label}\n")

if __name__ == "__main__":
    main()