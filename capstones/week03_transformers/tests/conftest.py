"""Shared fixtures; puts src/ and corpus/build/ on sys.path."""

import sys
from pathlib import Path

import pytest

W3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W3 / "src"))
build_dir = W3 / "corpus" / "build"  # make_split.py, for test_split
sys.path.insert(0, str(build_dir))

import data  # noqa: E402


@pytest.fixture(scope="session")
def documents():
    """All documents of the frozen snapshot, normalized, with splits."""
    return data.load_documents()


@pytest.fixture(scope="session")
def t_documents(documents):
    return [d for d in documents if d.split == "T"]
