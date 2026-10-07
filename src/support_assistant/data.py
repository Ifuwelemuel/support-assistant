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


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a ticket csv with no validation for that still needs cleaning"""
    return pd.read_csv(path, dtype=str)


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


TEXT_COLUMNS = ["id", "channel", "category", "priority", "subject", "body"]
LABEL_COLUMNS = ["channel", "category", "priority"]


def strip_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace in every text column; cells left blank become missing."""
    df = df.copy()
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip().replace("", pd.NA)
    return df


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the label columns, drop hyphens, join words with underscore"""
    df = df.copy()
    for col in LABEL_COLUMNS:
        df[col] = (
            df[col].str.lower().str.replace("-", "", regex=False).str.replace(" ", "_", regex=False)
        )

    return df


def fill_missing_priority(df: pd.DataFrame, default: str = "medium") -> pd.DataFrame:
    df = df.copy()
    df["priority"] = df["priority"].fillna(default)

    return df


def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop tickets with no id, subject or body — there is nothing to classify."""
    return df.dropna(subset=["id", "subject", "body"])


def drop_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first row for each id."""
    return df.drop_duplicates(subset="id", keep="first")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse created_at from any of the formats seen in the wild, day first."""
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", dayfirst=True)
    return df


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning rule in order and return a validated, tidy frame."""
    df = strip_text(df)
    df = normalise_labels(df)
    df = fill_missing_priority(df)
    df = drop_incomplete(df)
    df = drop_duplicate_ids(df)
    df = parse_dates(df)
    df = df.sort_values("id").reset_index(drop=True)
    issues = validate(df)
    if issues:
        raise TicketDataError("; ".join(issues))
    return df


def category_shares(df: pd.DataFrame) -> pd.Series:
    """share of tickets in each category, larger first"""

    return df["category"].value_counts(normalize=True)


INTENT_COLUMNS = ["text", "category"]


def normalise_text(text: str) -> str:
    """Lower-case and collapse all whitespace: the form used a compare two messages"""
    return " ".join(text.lower().split())


def load_intents(path: str | Path) -> pd.DataFrame:
    """Read a lablelled-message CSV -columns text, category- or raise Ticket error"""
    path = Path(path)
    if not path.is_file():
        raise TicketDataError(f"data file not found: {path}(run 'make data')")
    df = pd.read_csv(path)
    if list(df.columns) != INTENT_COLUMNS:
        raise TicketDataError(f"expected columns: {INTENT_COLUMNS}, found {list(df.columns)}")
    if df.isna().any().any():
        n_missing = int(df.isna().any(axis=1).sum())
        raise TicketDataError(f"{n_missing} rows have a missing text or category")
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
