import pytest

from support_assistant.data import TicketDataError, load_intents, normalise_text


def test_normalise_text_lowercases_and_collapses_whitespace():
    assert normalise_text("\nWhere  is my\tCARD? \n") == "where is my card?"


def test_load_intents_reads_text_and_category(tmp_path):
    path = tmp_path / "intents.csv"
    path.write_text(
        "text,category\n"
        '"Where is my card, please?",card_arrival\n'
        "My card is broken,card_not_working\n"
    )
    df = load_intents(path)
    assert df.shape == (2, 2)
    assert df["text"][0] == "Where is my card, please?"


def test_load_intents_rejects_wrong_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("message,label\nhello,greeting\n")
    with pytest.raises(TicketDataError, match="expected columns"):
        load_intents(path)


def test_load_intents_missing_file_says_how_to_get_it(tmp_path):
    with pytest.raises(TicketDataError, match="make data"):
        load_intents(tmp_path / "nope.csv")
