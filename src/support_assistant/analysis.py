"""plain - python analysis of support tickets"""

import csv
from pathlib import Path

Sample = Path("data/sample/ticket.csv")


def load_rows(path=Sample):
    with open(Sample) as f:
        return list(csv.DictReader(f))


def count_by(rows, field):
    """count how many rows have each value of field"""
    counts = {}
    for row in rows:
        value = row[field]
        if value in counts:
            counts[value] = counts[value] + 1

        else:
            counts[value] = 1
    return counts


if __name__ == "__main__":
    rows = load_rows()
    print(len(rows), "tickets")
    for field in ["category","priority","channel"]:
        print(field,count_by(rows,field))

