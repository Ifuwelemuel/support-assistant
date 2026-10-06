from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from support_assistant.data import (
    clean_tickets,
    drop_duplicate_ids,
    drop_incomplete,
    fill_missing_priority,
    load_raw,
    normalise_labels,
    parse_dates,
    strip_text,
    validate,
)

MESSY = Path(__file__).resolve().parents[1] / "data" / "sample" / "tickets_messy.csv"


def test_strip_text_trims_and_blanks_become_missing():
    df = pd.DataFrame(
        {
            "id": [" T-1 "],
            "channel": ["web"],
            "category": ["bug"],
            "priority": ["low"],
            "subject": ["  Hi  "],
            "body": ["   "],
        }
    )
    out = strip_text(df)
    assert out.loc[0, "id"] == "T-1"
    assert out.loc[0, "subject"] == "Hi"
    assert pd.isna(out.loc[0, "body"])


def test_normalise_labels():
    df = pd.DataFrame(
        {
            "channel": ["E-mail", "Chat"],
            "category": ["Feature Request", "LOGIN"],
            "priority": ["High", "low"],
        }
    )
    out = normalise_labels(df)
    assert out["channel"].tolist() == ["email", "chat"]
    assert out["category"].tolist() == ["feature_request", "login"]
    assert out["priority"].tolist() == ["high", "low"]


def test_fill_missing_priority_uses_medium():
    df = pd.DataFrame({"priority": ["high", None]})
    assert fill_missing_priority(df)["priority"].tolist() == ["high", "medium"]


def test_drop_incomplete_removes_rows_without_text():
    df = pd.DataFrame({"id": ["T-1", "T-2"], "subject": ["ok", None], "body": ["ok", "ok"]})
    assert drop_incomplete(df)["id"].tolist() == ["T-1"]


def test_drop_duplicate_ids_keeps_first():
    df = pd.DataFrame({"id": ["T-1", "T-1", "T-2"], "subject": ["first", "second", "x"]})
    assert drop_duplicate_ids(df)["subject"].tolist() == ["first", "x"]


def test_parse_dates_accepts_three_formats_day_first():
    df = pd.DataFrame({"created_at": ["2026-01-09", "01/09/2026", "1 Sep 2026"]})
    out = parse_dates(df)
    assert out["created_at"].nunique() == 1
    assert str(out.loc[0, "created_at"].date()) == "2026-09-01"


def test_rules_do_not_change_their_input():
    raw = load_raw(MESSY)
    before = raw.copy()
    clean_tickets(raw)
    assert_frame_equal(raw, before)


def test_clean_messy_equals_tidy(sample_path):
    messy = load_raw(MESSY)
    assert len(messy) == 15
    clean = clean_tickets(messy)
    assert validate(clean) == []
    # assert_frame_equal(clean, load_tickets(sample_path))
