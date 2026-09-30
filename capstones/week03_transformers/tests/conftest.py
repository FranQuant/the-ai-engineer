"""Shared fixtures; puts src/ and corpus/builders/ on sys.path."""

import sys
from pathlib import Path

import pytest

W3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W3 / "src"))
builders_dir = W3 / "corpus" / "builders"  # make_split.py, for test_split
sys.path.insert(0, str(builders_dir))

import data  # noqa: E402


@pytest.fixture(scope="session")
def documents():
    """All documents of the frozen snapshot, normalized, with splits."""
    return data.load_documents()


@pytest.fixture(scope="session")
def t_documents(documents):
    return [d for d in documents if d.split == "T"]
