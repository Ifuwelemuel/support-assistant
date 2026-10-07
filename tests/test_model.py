import pandas as pd
import pytest

from support_assistant.data import normalise_text
from support_assistant.model import (
    CLASSIFIERS,
    build_pipeline,
    compare_models,
    score_predictions,
    split_intents,
)

MESSAGES = [
    ("where is my card", "card_arrival"),
    ("my card has not arrived", "card_arrival"),
    ("still waiting for my new card to arrive", "card_arrival"),
    ("what is the exchange rate", "exchange_rate"),
    ("which exchange rate do you use", "exchange_rate"),
    ("how is the exchange rate calculated", "exchange_rate"),
]


def test_every_registered_classifier_fits_and_predicts():
    texts = [text for text, _ in MESSAGES]
    labels = [label for _, label in MESSAGES]
    for name in CLASSIFIERS:
        pipeline = build_pipeline(name)
        pipeline.fit(texts, labels)
        assert len(pipeline.predict(["where is my card"])) == 1


def test_unknown_classifier_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="unknown classifier 'magic'"):
        build_pipeline("magic")


def test_score_predictions_matches_the_hand_calculation():
    y_true = ["card_arrival"] * 6 + ["exchange_rate"] * 3 + ["lost_or_stolen_card"]
    y_pred = (
        ["card_arrival"] * 6 + ["exchange_rate", "exchange_rate", "card_arrival"] + ["card_arrival"]
    )
    scores = score_predictions(y_true, y_pred)
    assert scores["accuracy"] == pytest.approx(0.8)
    assert scores["macro_f1"] == pytest.approx(0.5524, abs=0.0001)


def test_compare_models_scores_every_model_on_the_same_folds():
    table = compare_models(numbered_frame(), folds=2)
    assert set(table["model"]) == set(CLASSIFIERS)
    assert list(table["macro_f1"]) == sorted(table["macro_f1"], reverse=True)
    assert table.loc[table["model"] == "baseline", "accuracy"].item() == 0.5


def numbered_frame():
    rows = [(f"card question number {i}", "card_arrival") for i in range(10)]
    rows += [(f"rate question number {i}", "exchange_rate") for i in range(10)]
    rows += [
        ("CARD question  number 3", "card_arrival"),
        ("Rate Question number 7", "exchange_rate"),
    ]
    return pd.DataFrame(rows, columns=["text", "category"])


def test_pipeline_fits_in_one_call_and_predicts():
    texts = [text for text, _ in MESSAGES]
    labels = [label for _, label in MESSAGES]
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)
    predicted = pipeline.predict(["has my card arrived yet", "tell me the exchange rate"])
    assert list(predicted) == ["card_arrival", "exchange_rate"]


def test_pipeline_steps_are_named():
    assert list(build_pipeline().named_steps) == ["tfidf", "classifier"]


def test_split_drops_repeats_and_keeps_intents_in_proportion():
    train_df, validation_df = split_intents(numbered_frame())
    assert len(train_df) == 16
    temp_df = validation_df["category"].value_counts().to_dict()
    assert temp_df == {"card_arrival": 2, "exchange_rate": 2}
    train_keys = set(train_df["text"].map(normalise_text))
    assert train_keys.isdisjoint(validation_df["text"].map(normalise_text))


def test_split_is_the_same_every_time():
    first = split_intents(numbered_frame())[1]
    second = split_intents(numbered_frame())[1]
    assert list(first["text"]) == list(second["text"])
