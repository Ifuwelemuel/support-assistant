"""plain - python analysis of support tickets"""

import csv
from pathlib import Path
from dataclasses import dataclass, fields
import pandas as pd

Sample = Path("data/sample/tickets.csv")


class TicketError(Exception):
    """Raised when the ticket data cannot be used: missing file, unknown field, o rows."""


@dataclass
class Ticket:
    """One support ticket. every field is text as it appears in the csv"""

    id: str
    created_at: str
    channel: str
    category: str
    priority: str
    subject: str
    body: str

    def is_high(self):
        """True for high-priority teicket"""

        return self.priority == "high"

    def describe(self):
        """A one line human summary: id, priority sunject"""
        return f" {self.id} ({self.priority}) {self.subject}"


FIELDS = [f.name for f in fields(Ticket)]


def load_tickets(path=Sample):
    """Read a ticket csv and return a list of tICKET OBJECT"""
    tickets = []

    for row in load_rows(path):
        try:
            tickets.append(Ticket(**row))

        except TypeError:
            raise TicketError(f"row {row.get('id', '?')} does not match the Ticket fields {FIELDS}")
    return tickets


def load_rows(path=Sample):
    """Read a ticket csv a list of dicts one dict per ticket"""
    path = Path(path)
    if not path.is_file():
        raise TicketError(f"ticket file not found: {path}")

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_frame(tickets):
    """The tickets as a pandas table: one row per ticket, one column per field"""
    return pd.DataFrame(tickets, columns=FIELDS)


def from_frame(df):
    """Rows of a table turned back into ticlets objects"""

    return [Ticket(**row) for row in df.to_dict("records")]


def check_field(tickets, field):
    """Raise TicketError if field is not a colum  of the data"""
    if field not in FIELDS:
        raise TicketError(f"unknown field '{field}';expected one of {FIELDS}")


def count_by(tickets, field):
    """Count how many tickets have each value of field; ties in first-seen order."""
    check_field(tickets, field)
    # counts = to_frame(tickets)[field].value_counts(sort=False)
    return to_frame(tickets)[field].value_counts(sort=False).to_dict()


def filter_by(tickets, field, value):
    """Return the tickets whose `field` equals `value`, as a new list."""
    check_field(tickets, field)
    df = to_frame(tickets)
    return from_frame(df[df[field] == value])


def high_priority(tickets):
    """return only the high-priority tickets."""
    return filter_by(tickets, "priority", "high")


def count_of(pair):
    return pair[1]


def top_words(tickets, n=3):
    """The n most common subject words as (word, count) pairs; ties in first-seen order."""
    words = to_frame(tickets)["subject"].str.lower().str.split().explode()
    counts = words.value_counts(sort=False).sort_values(ascending=False, kind="stable")
    return list(counts.head(n).items())


def write_report(tickets, path="reports/summary.txt"):
    """Write a plain-text summary of the tickets to `path` and return the path."""
    path = Path(path)
    lines = [f"{len(tickets)} tickets", ""]
    for field in ["category", "priority", "channel"]:
        lines.append(f"By {field}:")
        for value, n in count_by(tickets, field).items():
            lines.append(f"  {value}: {n}")
        lines.append("")
    lines.append("High priority:")
    for ticket in high_priority(tickets):
        lines.append(f"  {ticket.id}  {ticket.subject}")
    lines.append("")
    lines.append("Top words in subjects:")
    for word, n in top_words(tickets, 5):
        lines.append(f"  {word}: {n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    try:
        tickets = load_tickets()

    except TicketError as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    report = write_report(tickets)
    print(f"wrote {report}")
    print(report.read_text())
