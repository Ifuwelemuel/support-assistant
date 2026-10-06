from support_assistant.analysis import count_by


def test_count_by_keeps_first_seen_order(tickets):
    assert list(count_by(tickets, "priority")) == ["high", "medium", "low"]
    assert list(count_by(tickets, "channel")) == ["email", "chat", "web"]
