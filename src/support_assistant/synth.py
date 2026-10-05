"""Generate realistic synthetic support tickets for development and testing."""

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

CATEGORIES = {"billing": 0.30, "login": 0.25, "bug": 0.20, "shipping": 0.15, "feature_request": 0.10}
CHANNELS = {"email": 0.5, "chat": 0.3, "web": 0.2}
PRIORITY_BY_CATEGORY = {
    "billing": {"low": 0.3, "medium": 0.5, "high": 0.2},
    "login": {"low": 0.2, "medium": 0.4, "high": 0.4},
    "bug": {"low": 0.2, "medium": 0.4, "high": 0.4},
    "shipping": {"low": 0.4, "medium": 0.5, "high": 0.1},
    "feature_request": {"low": 0.7, "medium": 0.3, "high": 0.0},
}
SUBJECTS = {
    "billing": ["Charged twice this month", "Invoice missing VAT number", "Update card details", "Refund not received"],
    "login": ["Cannot reset password", "Locked out after 2FA change", "Email change not saving", "Session expires too fast"],
    "bug": ["App crashes on export", "Search returns no results", "Dashboard shows wrong totals", "Upload fails for large files"],
    "shipping": ["Where is my order", "Wrong item delivered", "Package arrived damaged", "Delivery date keeps moving"],
    "feature_request": ["Dark mode please", "Export to Excel", "Bulk edit for tickets", "Keyboard shortcuts"],
}
BODIES = [
    "Happened again today after the update.",
    "This has been going on for a week now.",
    "Can someone look into this as soon as possible?",
    "Let me know if you need screenshots.",
    "Second time I am writing about this.",
]


def _pick(rng, weights):
    """One key from a {value: weight} dict, chosen in proportion to the weights."""
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def make_tickets(n, seed=42, start=date(2026, 7, 1), days=92):
    """Return n synthetic tickets as a DataFrame; the same seed always gives the same tickets."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        category = _pick(rng, CATEGORIES)
        created = start + timedelta(days=rng.randrange(days))
        if created.weekday() >= 5 and rng.random() < 0.6:
            created -= timedelta(days=created.weekday() - 4)
        rows.append(
            {
                "id": f"T-{10001 + i}",
                "created_at": created.isoformat(),
                "channel": _pick(rng, CHANNELS),
                "category": category,
                "priority": _pick(rng, PRIORITY_BY_CATEGORY[category]),
                "subject": rng.choice(SUBJECTS[category]),
                "body": rng.choice(BODIES),
            }
        )
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def main(argv=None):
    """Command line: write synthetic tickets to a CSV under data/raw/."""
    parser = argparse.ArgumentParser(description="Write synthetic support tickets to a CSV.")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw/tickets_5000.csv")
    args = parser.parse_args(argv)
    df = make_tickets(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} tickets to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
