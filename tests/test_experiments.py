import mlflow
import pandas as pd

from support_assistant.experiments import parameter_grid, run_experiment, run_name


def numbered_frame():
    rows = [(f"card question number {i}", "card_arrival") for i in range(10)]
    rows += [(f"rate question number {i}", "exchange_rate") for i in range(10)]
    return pd.DataFrame(rows, columns=["text", "category"])


def test_grid_has_twelve_distinct_configurations():
    grid = parameter_grid()
    assert len(grid) == 12
    assert len({run_name(params) for params in grid}) == 12


def test_run_name_describes_the_configuration():
    params = {"classifier": "logistic", "tfidf__ngram_range": (1, 2), "classifier__C": 10.0}
    assert run_name(params) == "logistic-ngram2-C10.0"


def test_run_experiment_records_one_run_with_params_and_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'test.db'}")
    mlflow.set_experiment("test")
    params = {"classifier": "linear_svc", "tfidf__ngram_range": (1, 1), "classifier__C": 1.0}

    scores = run_experiment(numbered_frame(), params, folds=2, log_model=False)

    runs = mlflow.search_runs(experiment_names=["test"])
    assert len(runs) == 1
    assert runs["params.classifier"][0] == "linear_svc"
    assert runs["metrics.macro_f1"][0] == scores["macro_f1"]
    assert params["classifier"] == "linear_svc"
