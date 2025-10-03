import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_text
import models as models

df = pd.read_csv("News.csv")
df.columns = df.columns.str.strip().str.lower()
df["text"] = df["text"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Logistic Regression Report:")
print(models.train_logistic_regression(X_train, X_test, y_train, y_test))

print("\nDecision Tree Report:")
print(models.train_decision_tree(X_train, X_test, y_train, y_test))

print("\nRandom Forest Report:")
print(models.train_random_forest(X_train, X_test, y_train, y_test))