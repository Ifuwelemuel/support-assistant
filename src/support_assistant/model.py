"""Intent classifier: one pipeline from raw text to a predicted intent."""

import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from support_assistant.data import TicketDataError, load_intents, normalise_text

TRAIN_PATH = Path("data/raw/banking77_train.csv")


def build_pipeline():
    """An unfitted pipeline: TF-IDF features, then logistic regression."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def split_intents(df, validation_size=0.2, seed=42):
    """Drop repeated messages, then split into train and validation frames, stratified by intent."""
    unique = df[~df["text"].map(normalise_text).duplicated()]
    return train_test_split(
        unique, test_size=validation_size, stratify=unique["category"], random_state=seed
    )


def train(path=TRAIN_PATH, seed=42):
    """Fit the pipeline on the training part; return it with its validation accuracy."""
    train_df, validation_df = split_intents(load_intents(path), seed=seed)
    pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df["category"])
    accuracy = pipeline.score(validation_df["text"], validation_df["category"])
    return pipeline, accuracy


def main(argv=None):
    """Command line: train on a labelled CSV and print the validation accuracy."""
    parser = argparse.ArgumentParser(description="Train the intent classifier.")
    parser.add_argument("--data", type=Path, default=TRAIN_PATH)
    args = parser.parse_args(argv)
    try:
        _, accuracy = train(args.data)
    except TicketDataError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"validation accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
