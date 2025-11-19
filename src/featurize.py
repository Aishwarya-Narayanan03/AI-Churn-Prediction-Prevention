# src/featurize.py

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

def clean_text(text):
    text = text.lower().strip()
    return text

def featurize(df: pd.DataFrame, return_splits=True):
    """
    Featurize reviews data with TF-IDF
    
    Args:
        df: DataFrame with 'text' and 'rating' columns
        return_splits: If True, returns train/test splits (for notebooks)
                      If False, returns X and vectorizer only (for CLI)
    
    Returns:
        If return_splits=True: X_train, X_test, y_train, y_test, vectorizer
        If return_splits=False: X, vectorizer
    """
    # Create binary label: 1 if rating >= 4 (positive), 0 otherwise (negative)
    if 'rating' in df.columns:
        df["label"] = (df["rating"] >= 4).astype(int)
    
    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["clean_text"])
    
    if return_splits and 'label' in df.columns:
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, vectorizer
    else:
        return X, vectorizer


def main():
    input_parquet = "Data/processed/reviews_raw.parquet"
    output_features = "Data/processed/features.pkl"
    output_vectorizer = "Data/processed/tfidf.pkl"

    df = pd.read_parquet(input_parquet)

    X, vectorizer = featurize(df, return_splits=False)

    os.makedirs("Data/processed", exist_ok=True)

    with open(output_features, "wb") as f:
        pickle.dump(X, f)

    with open(output_vectorizer, "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Feature engineering completed!")
    print(f"Saved: {output_features}")
    print(f"Saved: {output_vectorizer}")


if __name__ == "__main__":
    main()

