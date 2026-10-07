import pandas as pd

from support_assistant.data import normalise_text
from support_assistant.model import build_pipeline, split_intents

MESSAGES = [
    ("where is my card", "card_arrival"),
    ("my card has not arrived", "card_arrival"),
    ("still waiting for my new card to arrive", "card_arrival"),
    ("what is the exchange rate", "exchange_rate"),
    ("which exchange rate do you use", "exchange_rate"),
    ("how is the exchange rate calculated", "exchange_rate"),
]


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
