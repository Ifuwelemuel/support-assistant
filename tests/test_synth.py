from support_assistant.data import load_tickets, validate
from support_assistant.synth import make_tickets


def test_output_is_valid():
    assert validate(make_tickets(500, seed=1)) == []


def test_same_seed_same_tickets():
    assert make_tickets(100, seed=1).equals(make_tickets(100, seed=1))
    assert not make_tickets(100, seed=1).equals(make_tickets(100, seed=2))


def test_round_trip_through_csv_and_loader(tmp_path):
    out = tmp_path / "tickets.csv"
    make_tickets(300, seed=3).to_csv(out, index=False)
    loaded = load_tickets(out)
    assert len(loaded) == 300
    assert loaded["id"].iloc[0] == "T-10001"
    assert loaded["id"].is_unique


def test_category_shares_match_weights_at_scale():
    shares = make_tickets(5000).category.value_counts(normalize=True)
    assert 0.27 < shares["billing"] < 0.33
    assert 0.07 < shares["feature_request"] < 0.13


def test_feature_requests_are_never_high():
    df = make_tickets(2000, seed=5)
    assert ((df["category"] == "feature_request") & (df["priority"] == "high")).sum() == 0
