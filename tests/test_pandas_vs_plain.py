from support_assistant.analysis import count_by, high_priority, top_words
from support_assistant.data import load_tickets as load_df


def test_pandas_counts_match_plain(tickets,sample_path):
    df = load_df(sample_path)
    for field in ["category","priority","channel"]:
        assert df[field].value_counts().to_dict() == count_by(tickets,field)


def test_pandas_high_priority_matches_plain(tickets, sample_path):
    df = load_df(sample_path)
    pandas_ids = df[df["priority"] == "high"]["id"].tolist()
    plain_ids = [t.id for t in high_priority(tickets)]
    assert pandas_ids == plain_ids


def test_pandas_top_words_match_plain(tickets, sample_path):
    df = load_df(sample_path)
    words = df["subject"].str.lower().str.split().explode()
    assert words.value_counts().head(2).to_dict() == dict(top_words(tickets, 2))
