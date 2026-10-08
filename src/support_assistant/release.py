"""Release: fit the chosen configuration, score it once on the test set, register and export it."""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import mlflow
import sklearn
from mlflow import MlflowClient

from support_assistant.data import TicketDataError, load_intents
from support_assistant.experiments import EXPERIMENT, TRACKING_URI
from support_assistant.model import (
    TRAIN_PATH,
    compare_with_baseline,
    count_overlap,
    fit_final,
    save_model,
    unique_messages,
)

TEST_PATH = Path("data/raw/banking77_test.csv")
MODEL_NAME = "banking77-intent-classifier"
MODELS_DIR = Path("models")

FINAL_PARAMS = {"classifier": "linear_svc", "tfidf__ngram_range": (1, 2), "classifier__C": 1.0}


class ReleaseError(Exception):
    """Raised when a release must not go ahead."""


def git_commit():
    """The short hash of the commit the code is at, so a model can be traced to its source."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def release(version=1, train_path=TRAIN_PATH, test_path=TEST_PATH, params=None):
    """Fit, score once on the test set, record in MLflow, register, and export. Returns the card."""
    params = params or FINAL_PARAMS
    card_path = MODELS_DIR / f"intent-classifier-v{version}.json"
    if card_path.exists():
        raise ReleaseError(
            f"version {version} is already released ({card_path}); the test set is scored once"
        )
    train_df, test_df = load_intents(train_path), load_intents(test_path)
    pipeline = fit_final(train_df, params)
    card = {
        "name": MODEL_NAME,
        "version": version,
        "released": date.today().isoformat(),
        "git_commit": git_commit(),
        "scikit_learn": sklearn.__version__,
        "params": params,
        "train_rows": len(unique_messages(train_df)),
        "test_rows": len(test_df),
        "test_overlap": count_overlap(train_df, test_df),
        "test_scores": compare_with_baseline(pipeline, train_df, test_df),
    }
    scores = card["test_scores"]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"release-v{version}"):
        mlflow.log_params(params)
        mlflow.log_params({"train_rows": card["train_rows"], "test_rows": card["test_rows"]})
        for who in ["model", "baseline", "margin"]:
            mlflow.log_metrics({f"test_{who}_{name}": value for name, value in scores[who].items()})
        mlflow.log_metric("test_overlap", card["test_overlap"])
        mlflow.log_dict(card, "model-card.json")
        logged = mlflow.sklearn.log_model(pipeline, name="model")
    registered = mlflow.register_model(logged.model_uri, MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", registered.version)

    card["registry_version"] = int(registered.version)
    save_model(pipeline, MODELS_DIR / f"intent-classifier-v{version}.joblib")
    card_path.write_text(json.dumps(card, indent=2) + "\n")
    return card


def main(argv=None):
    """Command line: release one version and print the comparison with the baseline."""
    parser = argparse.ArgumentParser(description="Release the intent classifier.")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        card = release(args.version)
    except (TicketDataError, ReleaseError) as exc:
        print(f"FAIL: {exc}")
        return 1
    scores = card["test_scores"]
    print(f"released {card['name']} v{card['version']} at commit {card['git_commit']}")
    print(f"{'':10}{'accuracy':>10}{'macro-F1':>10}")
    for who in ["model", "baseline", "margin"]:
        print(f"{who:10}{scores[who]['accuracy']:>10.4f}{scores[who]['macro_f1']:>10.4f}")
    print(f"test messages also in train: {card['test_overlap']} of {card['test_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
