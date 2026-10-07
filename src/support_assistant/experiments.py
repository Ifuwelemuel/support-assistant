"""Experiment tracking: every configuration tried is recorded as one MLflow run."""

import argparse
from pathlib import Path

import mlflow
from sklearn.model_selection import StratifiedKFold, cross_validate

from support_assistant.data import TicketDataError, load_intents
from support_assistant.model import TRAIN_PATH, build_pipeline, unique_messages

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "banking77-intents"


def parameter_grid():
    """Every configuration to try: 2 classifiers x 2 n-gram ranges x 3 values of C = 12."""
    grid = []
    for classifier in ["logistic", "linear_svc"]:
        for ngram_range in [(1, 1), (1, 2)]:
            for c_value in [0.1, 1.0, 10.0]:
                grid.append(
                    {
                        "classifier": classifier,
                        "tfidf__ngram_range": ngram_range,
                        "classifier__C": c_value,
                    }
                )
    return grid


def run_name(params):
    """A short readable label for a configuration, e.g. logistic-ngram2-C10.0."""
    longest = params["tfidf__ngram_range"][1]
    return f"{params['classifier']}-ngram{longest}-C{params['classifier__C']}"


def run_experiment(df, params, folds=5, seed=42, log_model=True):
    """Cross-validate one configuration and record it as one MLflow run; return its mean scores."""
    settings = dict(params)
    classifier = settings.pop("classifier")
    pipeline = build_pipeline(classifier).set_params(**settings)
    unique = unique_messages(df)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    with mlflow.start_run(run_name=run_name(params)):
        mlflow.log_params(params)
        mlflow.log_params({"folds": folds, "seed": seed, "rows": len(unique)})
        result = cross_validate(
            pipeline,
            unique["text"],
            unique["category"],
            cv=splitter,
            scoring=["accuracy", "f1_macro"],
        )
        scores = {
            "macro_f1": result["test_f1_macro"].mean(),
            "macro_f1_std": result["test_f1_macro"].std(),
            "accuracy": result["test_accuracy"].mean(),
            "accuracy_std": result["test_accuracy"].std(),
            "fit_seconds": result["fit_time"].mean(),
        }
        mlflow.log_metrics(scores)
        mlflow.log_dict({"macro_f1_per_fold": result["test_f1_macro"].tolist()}, "folds.json")
        if log_model:
            pipeline.fit(unique["text"], unique["category"])
            logged = mlflow.sklearn.log_model(pipeline, name="model")
            mlflow.set_tag("model_uri", logged.model_uri)
    return scores


def main(argv=None):
    """Command line: run the whole grid, one MLflow run per configuration."""
    parser = argparse.ArgumentParser(description="Run the experiment grid and track it in MLflow.")
    parser.add_argument("--data", type=Path, default=TRAIN_PATH)
    args = parser.parse_args(argv)
    try:
        df = load_intents(args.data)
    except TicketDataError as exc:
        print(f"FAIL: {exc}")
        return 1
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    grid = parameter_grid()
    for number, params in enumerate(grid, start=1):
        scores = run_experiment(df, params)
        label = f"{number:2}/{len(grid)}  {run_name(params):28}"
        print(f"{label}  macro-F1 {scores['macro_f1']:.4f} ± {scores['macro_f1_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
