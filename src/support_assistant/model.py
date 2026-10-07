"""Intent classifier: one pipeline from raw text to a predicted intent, and how it is judged."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from support_assistant.data import TicketDataError, load_intents, normalise_text

TRAIN_PATH = Path("data/raw/banking77_train.csv")

CLASSIFIERS = {
    "baseline": (DummyClassifier, {"strategy": "most_frequent"}),
    "naive_bayes": (MultinomialNB, {}),
    "logistic": (LogisticRegression, {"max_iter": 1000}),
    "linear_svc": (LinearSVC, {}),
}


def build_pipeline(classifier="logistic"):
    """An unfitted pipeline: TF-IDF features, then the named  classifier."""
    if classifier not in CLASSIFIERS:
        raise ValueError(f"unknown classifier '{classifier}'; expected one of {list(CLASSIFIERS)}")
    estimator_class, settings = CLASSIFIERS[classifier]
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", estimator_class(**settings)),
        ]
    )


def score_predictions(y_true, y_pred):
    """The two numbers every result is reported with: accuracy and macro-F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def unique_messages(df):
    """The frame without repeated messages (compared after normalising)."""
    return df[~df["text"].map(normalise_text).duplicated()]


def split_intents(df, validation_size=0.2, seed=42):
    """Drop repeated messages, then split into train and validation frames, stratified by intent."""
    unique = unique_messages(df)
    return train_test_split(
        unique, test_size=validation_size, stratify=unique["category"], random_state=seed
    )


def train(path=TRAIN_PATH, classifier="logistic", seed=42):
    """Fit the pipeline on the training part; return it with its validation scores."""
    train_df, validation_df = split_intents(load_intents(path), seed=seed)
    pipeline = build_pipeline(classifier)
    pipeline.fit(train_df["text"], train_df["category"])
    predicted = pipeline.predict(validation_df["text"])
    return pipeline, score_predictions(validation_df["category"], predicted)


def compare_models(df, folds=5, seed=42):
    """Cross-validate every classifier on the same folds; one row per model, best macro-F1 first."""
    unique = unique_messages(df)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name in CLASSIFIERS:
        result = cross_validate(
            build_pipeline(name),
            unique["text"],
            unique["category"],
            cv=splitter,
            scoring=["accuracy", "f1_macro"],
        )
        rows.append(
            {
                "model": name,
                "macro_f1": result["test_f1_macro"].mean(),
                "macro_f1_std": result["test_f1_macro"].std(),
                "accuracy": result["test_accuracy"].mean(),
                "accuracy_std": result["test_accuracy"].std(),
                "fit_seconds": result["fit_time"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def main(argv=None):
    """Command line: train one model, or with --compare cross-validate all of them."""
    parser = argparse.ArgumentParser(description="Train or compare intent classifiers.")
    parser.add_argument("--data", type=Path, default=TRAIN_PATH)
    parser.add_argument("--compare", action="store_true", help="cross-validate every model")
    args = parser.parse_args(argv)
    try:
        if args.compare:
            print(compare_models(load_intents(args.data)).round(4).to_string(index=False))
        else:
            _, scores = train(args.data)
            accuracy, macro_f1 = scores["accuracy"], scores["macro_f1"]
            print(f"validation accuracy: {accuracy:.4f} | macro-F1: {macro_f1:.4f}")
    except TicketDataError as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
