"""plain - python analysis of support tickets"""

import csv
from pathlib import Path

Sample = Path("/Users/mac/Desktop/2026/Bootcamp/code/support-assistant/data/sample/ticket.csv")


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

def filter_by(rows,field,value):
    """Return the rows whose filed equals value, as a new list"""
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

def top_words(row,n=3):
    """The n most common words across ticket subject, as (word, count) pairs"""
    counts = {}
    for row in rows:
        for word in row["subject"].lower().split():
            counts[word] = counts.get(word,0) + 1
    ranked = sorted(counts.items(), key=count_of, reverse=True)
    return ranked[:n]


if __name__ == "__main__":
    rows = load_rows()
    print(len(rows), "tickets")
    for field in ["category","priority","channel"]:
        print(field,count_by(rows,field))
    print("high priority:")
    for row in high_priority(rows):
        print(" ", row["id"], row["subject"])
    print("top words:", top_words(rows))
    print("top five:", top_words(rows,5))

