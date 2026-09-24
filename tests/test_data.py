from pathlib import Path

import pytest
import pandas as pd

from support_assistant.data import REQUIRED_COLUMNS, TicketDataError, load_tickets, validate

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "tickets.csv"


def test_sample_loads_twelve_tickets():
    df = load_tickets(SAMPLE)
    assert len(df) == 12
    assert list(df.columns) == REQUIRED_COLUMNS


def test_quoted_comma_stays_in_one_field():
    df = load_tickets(SAMPLE)
    body = df.loc[df["id"] == "T-1001", "body"].item()
    assert "," in body


def test_unknown_category_is_rejected(tmp_path):
    df = pd.read_csv(SAMPLE, dtype=str)
    df.loc[0, "category"] = "refund"
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(TicketDataError, match="unknown category"):
        load_tickets(bad)


def test_missing_column_is_reported():
    df = pd.read_csv(SAMPLE, dtype=str).drop(columns=["body"])
    assert validate(df) == ["missing columns: ['body']"]


def test_missing_file_is_a_ticket_error():
    with pytest.raises(TicketDataError, match="not found"):
        load_tickets(ROOT / "data" / "sample" / "nope.csv")
