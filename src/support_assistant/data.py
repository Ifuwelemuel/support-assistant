"""Load and validate support tickets."""

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["id", "created_at", "channel", "category", "priority", "subject", "body"]
ALLOWED_CATEGORIES = {"billing", "login", "shipping", "bug", "feature_request"}
ALLOWED_PRIORITIES = {"low", "medium", "high"}


class TicketDataError(ValueError):
    """Raised when a ticket file fails validation."""


def validate(df: pd.DataFrame) -> list[str]:
    """Return every problem found in ``df``; an empty list means it is valid."""
    issues: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        return issues
    if df.empty:
        issues.append("no data rows")
        return issues
    bad_category = df.loc[~df["category"].isin(ALLOWED_CATEGORIES), "id"].tolist()
    if bad_category:
        issues.append(f"unknown category in: {bad_category}")
    bad_priority = df.loc[~df["priority"].isin(ALLOWED_PRIORITIES), "id"].tolist()
    if bad_priority:
        issues.append(f"unknown priority in: {bad_priority}")
    empty_text = df.loc[df["subject"].isna() | df["body"].isna(), "id"].tolist()
    if empty_text:
        issues.append(f"empty subject or body in: {empty_text}")
    duplicates = df.loc[df["id"].duplicated(), "id"].tolist()
    if duplicates:
        issues.append(f"duplicate ids: {duplicates}")
    return issues


def load_tickets(path: str | Path) -> pd.DataFrame:
    """Read a ticket CSV and return a validated DataFrame, or raise TicketDataError."""
    path = Path(path)
    if not path.is_file():
        raise TicketDataError(f"data file not found: {path}")
    df = pd.read_csv(path, dtype=str)
    issues = validate(df)
    if issues:
        raise TicketDataError("; ".join(issues))
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: validate a ticket CSV, exit 0 if clean."""
    parser = argparse.ArgumentParser(description="Validate a support-ticket CSV.")
    parser.add_argument("path", nargs="?", default="data/sample/tickets.csv")
    args = parser.parse_args(argv)
    try:
        df = load_tickets(args.path)
    except TicketDataError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(df)} tickets in {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
