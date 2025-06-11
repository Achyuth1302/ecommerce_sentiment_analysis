import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model(df):
    X = df['clean_text']
    y = df['sentiment']

    # Make sure at least two classes exist
    if len(y.unique()) < 2:
        raise ValueError("Need at least two sentiment classes to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)

    return model, vectorizer, X_test_vec, y_test
