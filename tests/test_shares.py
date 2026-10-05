import pandas as pd

from support_assistant.data import category_shares


def test_category_shares_are_fractions_largest_first():
    df = pd.DataFrame({"category": ["bug", "bug", "login", "billing"]})
    shares = category_shares(df)
    assert shares.index[0] == "bug"
    assert shares["bug"] == 0.5
    assert shares["login"] == 0.25
    assert shares.sum() == 1.0
