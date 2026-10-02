"""plain - python analysis of support tickets"""

import csv
from pathlib import Path

Sample = Path("/Users/mac/Desktop/2026/Bootcamp/code/support-assistant/data/sample/ticket.csv")


class TicketError(Exception):
    """Raised when the ticket data cannot be used: missing file, unknown field, o rows."""




def load_rows(path=Sample):
    """Read a ticket csv a list of dicts one dict per ticket"""
    path = Path(path)
    if not path.is_file():
        raise TicketError(f"ticket file not found: {path}")

    with open(path, newline="") as f:
        return list(csv.DictReader(f))
def check_field(rows, field):
    """Raise TicketError if field is not a colum  of the data"""
    if rows and field not in rows[0]:
        raise TicketError(f"unknown field'{field}';expected one of {list(rows[0].keys())}")

def count_by(rows, field):
    """count how many rows have each value of field"""
    check_field(rows,field)
    counts = {}
    for row in rows:
        value = row[field]
        if value in counts:
            counts[value] = counts[value] + 1

        else:
            counts[value] = 1
    return counts

def filter_by(rows,field,value):
    """Return the rows whose filed equals value, as a new list"""
    check_field(rows,field)
    result = []
    for row in rows:
        if row[field] == value:
            result.append(row)
    return result

def high_priority(rows):
    """return only the high-priority tickets."""
    return filter_by(rows, "priority", "high")

def count_of(pair):
    return pair[1]

def top_words(rows,n=3):
    """The n most common words across ticket subject, as (word, count) pairs"""
    counts = {}
    for row in rows:
        for word in row["subject"].lower().split():
            counts[word] = counts.get(word,0) + 1
    ranked = sorted(counts.items(), key=count_of, reverse=True)
    return ranked[:n]

def write_report(rows, path="reports/summary.txt"):
    """Write a plain-text summary of the tickets to `path` and return the path."""
    path = Path(path)
    lines = [f"{len(rows)} tickets", ""]
    for field in ["category", "priority", "channel"]:
        lines.append(f"By {field}:")
        for value, n in count_by(rows, field).items():
            lines.append(f"  {value}: {n}")
        lines.append("")
    lines.append("High priority:")
    for row in high_priority(rows):
        lines.append(f"  {row['id']}  {row['subject']}")
    lines.append("")
    lines.append("Top words in subjects:")
    for word, n in top_words(rows, 5):
        lines.append(f"  {word}: {n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    try:
        rows = load_rows()

    except TicketError as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    report = write_report(rows)
    print(f"wrote {report}")
    print(report.read_text())

