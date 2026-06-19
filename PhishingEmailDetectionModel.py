import pandas as pd
import re
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "emails.csv")

df = pd.read_csv(csv_path)

# Features and Labels
X = df["text"]
y = df["label"]

# Convert email text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build Model Pipeline
model = Pipeline([
    ('tfidf', vectorizer),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test Emails
sample_emails = [
    "Verify your account immediately by clicking this link http://fakebank.com",
    "Meeting scheduled tomorrow at 10 AM",
    "Your password has expired. Update now.",
    "Project report submission deadline extended."
]

print("\nSample Predictions:")
for email in sample_emails:
    prediction = model.predict([email])[0]

    if prediction == 1:
        print(f"\nEmail: {email}")
        print("Prediction: Phishing")
    else:
        print(f"\nEmail: {email}")
        print("Prediction: Safe")