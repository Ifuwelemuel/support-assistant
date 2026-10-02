from pathlib import Path
from support_assistant.analysis import count_by, filter_by, high_priority, load_rows, top_words

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "tickets.csv"

def test_count_by_priority():
    rows = load_rows(SAMPLE)
    assert count_by(rows,"priority") == {"high":3,"medium":5,"low":4}


def test_load_rows_reads_all_tickets():
    rows = load_rows(SAMPLE)
    assert len(rows) == 12
    assert rows[0]["id"] == "T-1001"
    assert rows[-1]["id"] == "T-1012"

def test_count_category():
    rows = load_rows(SAMPLE)
    assert count_by(rows, "category") == {
        "billing": 3,
        "login": 3,
        "shipping": 2,
        "bug": 2,
        "feature_request": 2,
    }

def test_filter_by_billing_ids():
    rows = load_rows(SAMPLE)
    ids = []
    for row in filter_by(rows, "category", "billing"):
        ids.append(row["id"])
    assert ids == ["T-1001","T-1006","T-1010"]

def test_filter_by_no_match_is_empty():
    rows = load_rows(SAMPLE)
    assert filter_by(rows,"channel","fax") == []


def test_high_priority_ids():
    rows = load_rows(SAMPLE)
    assert len(high_priority(rows)) ==3
    assert high_priority(rows)[0]["id"] == "T-1001"



def test_top_word_first_two():
    rows = load_rows(SAMPLE)
    assert top_words(rows)[:2] == [("export" , 2),("change",2)]


def test_top_words_respects_n():
    rows = load_rows(SAMPLE)
    assert len(top_words(rows, 5)) == 5
    assert len(top_words(rows,1)) == 1
