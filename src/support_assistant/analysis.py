"""plain - python analysis of support tickets"""

import csv
from pathlib import Path
from dataclasses import dataclass,fields

Sample = Path("data/sample/ticket.csv")


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
            raise TicketError(f"row {row.get('id','?')} does not match the Ticket fields {FIELDS}")
    return tickets






def load_rows(path=Sample):
    """Read a ticket csv a list of dicts one dict per ticket"""
    path = Path(path)
    if not path.is_file():
        raise TicketError(f"ticket file not found: {path}")

    with open(path, newline="") as f:
        return list(csv.DictReader(f))
def check_field(tickets, field):
    """Raise TicketError if field is not a colum  of the data"""
    if field  not in FIELDS:
        raise TicketError(f"unknown field '{field}';expected one of {FIELDS}")

def count_by(tickets, field):
    """count how many rows have each value of field"""
    check_field(tickets,field)
    counts = {}
    for ticket in tickets:
        value = getattr(ticket,field)
        if value in counts:
            counts[value] = counts[value] + 1

        else:
            counts[value] = 1
    return counts

def filter_by(tickets,field,value):
    """Return the rows whose filed equals value, as a new list"""
    check_field(tickets,field)
    result = []
    for ticket in tickets:
        if getattr(ticket, field) == value:
            result.append(ticket)
    return result

def high_priority(tickets):
    """return only the high-priority tickets."""
    return filter_by(tickets, "priority", "high")

def count_of(pair):
    return pair[1]

def top_words(tickets,n=3):
    """The n most common words across ticket subject, as (word, count) pairs"""
    counts = {}
    for ticket in tickets:
        for word in ticket.subject.lower().split():
            counts[word] = counts.get(word,0) + 1
    ranked = sorted(counts.items(), key=count_of, reverse=True)
    return ranked[:n]

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

