from pathlib import Path

import pytest

from support_assistant.analysis import (
    Ticket,
    TicketError,
    count_by,
    filter_by,
    high_priority,
    load_tickets,
    top_words,
    write_report,
)

SAMPLE = Path("data/sample/ticket.csv")

def test_count_by_priority(tickets):
    assert count_by(tickets, "priority") == {"high": 3, "medium": 5, "low": 4}


def test_load_tickets_reads_all_tickets(tickets):
    assert len(tickets) == 12
    assert tickets[0].id == "T-1001"
    assert tickets[-1].id == "T-1012"
    assert isinstance(tickets[0], Ticket)


def test_count_by_category(tickets):
    assert count_by(tickets, "category") == {
        "billing": 3,
        "login": 3,
        "shipping": 2,
        "bug": 2,
        "feature_request": 2,
    }


def test_filter_by_billing_ids(tickets):
    ids = []
    for ticket in filter_by(tickets, "category", "billing"):
        ids.append(ticket.id)
    assert ids == ["T-1001", "T-1006", "T-1010"]


def test_filter_by_no_match_is_empty(tickets):
    assert filter_by(tickets, "channel", "fax") == []


def test_high_priority_ids(tickets):
    assert len(high_priority(tickets)) == 3
    assert high_priority(tickets)[0].id == "T-1001"


def test_top_words_first_two(tickets):
    assert top_words(tickets)[:2] == [("export", 2), ("change", 2)]


def test_top_words_respects_n(tickets):
    assert len(top_words(tickets, 5)) == 5
    assert len(top_words(tickets, 1)) == 1


def test_missing_file_raises_ticket_error():
    with pytest.raises(TicketError, match="not found"):
        load_tickets("data/sample/nope.csv")


def test_unknown_field_raises_ticket_error(tickets):
    with pytest.raises(TicketError, match="unknown field 'colour'"):
        count_by(tickets, "colour")
    with pytest.raises(TicketError, match="unknown field"):
        filter_by(tickets, "colour", "red")


def test_write_report_creates_file(tickets, tmp_path):
    report = write_report(tickets, tmp_path / "out" / "summary.txt")
    assert report.exists()
    text = report.read_text()
    assert text.startswith("12 tickets")
    assert "export: 2" in text


def test_ticket_is_high_and_describe():
    ticket = Ticket("T-1", "2026-01-01", "email", "bug", "high", "Crash", "It crashed")
    assert ticket.is_high()
    assert ticket.describe() == ' T-1 (high) Crash'
    assert not Ticket("T-2", "2026-01-01", "chat", "bug", "low", "Slow", "Slow").is_high()


def test_wrong_columns_raise_ticket_error(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,subject\nT-9,Only two columns\n")
    with pytest.raises(TicketError, match="does not match the Ticket fields"):
        load_tickets(bad)


def test_tickets_with_same_data_are_equal(tickets):
    assert tickets[0] == load_tickets(SAMPLE)[0]
    assert tickets[0] != tickets[1]
