import pandas as pd
import pytest

from support_assistant import release as release_module
from support_assistant.model import (
    compare_with_baseline,
    count_overlap,
    fit_final,
    load_model,
    save_model,
)
from support_assistant.release import ReleaseError, release

PARAMS = {"classifier": "linear_svc", "tfidf__ngram_range": (1, 1), "classifier__C": 1.0}


def frame(rows):
    return pd.DataFrame(rows, columns=["text", "category"])


TRAIN = frame(
    [(f"card question number {i}", "card_arrival") for i in range(10)]
    + [(f"rate question number {i}", "exchange_rate") for i in range(10)]
)
TEST = frame(
    [
        ("a card question", "card_arrival"),
        ("card question again", "card_arrival"),
        ("a rate question", "exchange_rate"),
        ("RATE question  number 3", "exchange_rate"),
    ]
)


def test_fit_final_fits_and_leaves_the_params_untouched():
    pipeline = fit_final(TRAIN, PARAMS)
    assert list(pipeline.predict(["card question"])) == ["card_arrival"]
    assert PARAMS["classifier"] == "linear_svc"


def test_margin_is_model_minus_baseline():
    results = compare_with_baseline(fit_final(TRAIN, PARAMS), TRAIN, TEST)
    assert results["model"]["accuracy"] == 1.0
    assert results["baseline"]["accuracy"] == 0.5
    assert results["margin"]["accuracy"] == 0.5


def test_count_overlap_ignores_case_and_spacing():
    assert count_overlap(TRAIN, TEST) == 1


def test_saved_model_loads_and_predicts_the_same(tmp_path):
    pipeline = fit_final(TRAIN, PARAMS)
    path = save_model(pipeline, tmp_path / "models" / "model.joblib")
    loaded = load_model(path)
    assert list(loaded.predict(TEST["text"])) == list(pipeline.predict(TEST["text"]))


def test_release_refuses_to_score_the_test_set_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(release_module, "MODELS_DIR", tmp_path)
    (tmp_path / "intent-classifier-v1.json").write_text("{}")
    with pytest.raises(ReleaseError, match="already released"):
        release(version=1)
