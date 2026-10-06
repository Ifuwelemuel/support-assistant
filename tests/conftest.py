from pathlib import Path

import pytest

from support_assistant.analysis import load_tickets

SAMPLE = Path("data/sample/tickets.csv")


@pytest.fixture
def tickets():
    """The twelve sample tickets, loaded fresh for each test that asks for them."""
    return load_tickets(SAMPLE)


@pytest.fixture
def sample_path():
    """Path to sample CSV, for tests that need to load it their own way."""

    return SAMPLE
