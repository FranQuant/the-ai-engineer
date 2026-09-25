"""Shared fixtures; puts the capstone modules on sys.path."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data  # noqa: E402


@pytest.fixture(scope="session")
def documents():
    """All documents of the frozen snapshot, normalized, with splits."""
    return data.load_documents()


@pytest.fixture(scope="session")
def t_documents(documents):
    return [d for d in documents if d.split == "T"]
