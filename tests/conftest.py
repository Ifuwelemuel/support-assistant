from pathlib import Path

import pytest

from support_assistant.analysis import load_tickets

SAMPLE = Path("data/sample/ticket.csv")

@pytest.fixture
def tickets():
    """The twelve sample tickets, loaded fresh for each test that asks for them."""
    return load_tickets(SAMPLE)


