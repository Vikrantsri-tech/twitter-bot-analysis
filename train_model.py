"""Train/evaluate a text-classification baseline on labelled CSV data.

This uses an SVM baseline, not a Relevance Vector Machine (RVM).
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocess import clean_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/tweets.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Read data/README.md for the expected format."
        )

    df = pd.read_csv(path)
    for col in (args.text_column, args.label_column):
        if col not in df.columns:
            raise ValueError(f"Required column {col!r} is missing from {path}")

    df = df[[args.text_column, args.label_column]].dropna()
    df[args.text_column] = df[args.text_column].map(clean_text)
    df = df[df[args.text_column].str.len() > 0]

    if df[args.label_column].nunique() < 2:
        raise ValueError("At least two distinct label classes are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_column],
        df[args.label_column],
        test_size=0.2,
        random_state=42,
        stratify=df[args.label_column],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("classifier", SVC(kernel="rbf", class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
