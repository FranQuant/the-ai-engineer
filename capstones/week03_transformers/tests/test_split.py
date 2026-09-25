"""Regression tests for make_split.py (Phase 1 Codex review item #4)."""

import collections
import datetime as dt
import json

import pytest

import data
import make_split
from make_split import (NEAR_END, TRAIN_END, InfeasibleError,
                        assign_split, availability_date)

SPLIT_SHA256 = (
    "f15633e9a53d311dd866e196c017a0f1bc87fd5ea020813fe1fce829e8d82bd7")


@pytest.fixture(scope="module")
def real_split():
    return make_split.build(data.CORPUS, data.MANIFEST)


def test_output_matches_committed_manifest(tmp_path, capsys):
    out = tmp_path / "split_manifest.json"
    assert make_split.main(["--output", str(out)]) == 0
    assert data.sha256_bytes(out.read_bytes()) == SPLIT_SHA256
    assert out.read_bytes() == data.SPLIT_MANIFEST.read_bytes()
    assert SPLIT_SHA256 in capsys.readouterr().out
    assert not list(tmp_path.glob(".*.tmp"))


def test_frozen_constant_matches():
    assert data.SPLIT_MANIFEST_SHA256 == SPLIT_SHA256


@pytest.mark.parametrize("boundary", [TRAIN_END, NEAR_END])
@pytest.mark.parametrize("days", [-30, 0, 30])
def test_within_30_days_excluded(boundary, days):
    avail = boundary + dt.timedelta(days=days)
    assert assign_split(avail) == ("excluded", boundary.isoformat())


@pytest.mark.parametrize("avail, split", [
    (TRAIN_END - dt.timedelta(days=31), "T"),
    (TRAIN_END + dt.timedelta(days=31), "N"),
    (NEAR_END - dt.timedelta(days=31), "N"),
    (NEAR_END + dt.timedelta(days=31), "F"),
])
def test_31_days_assigned(avail, split):
    assert assign_split(avail) == (split, None)


def test_availability_dates():
    meeting = dt.date(2019, 10, 30)
    assert availability_date(meeting, {"statement"}, None) == meeting
    assert (availability_date(meeting, {"statement", "minutes"}, None)
            == dt.date(2019, 11, 20))
    assert (availability_date(meeting, {"minutes"}, "2019-11-25")
            == dt.date(2019, 11, 25))


def test_real_meetings_without_minutes(real_split):
    no_minutes = [m for m in real_split["meetings"]
                  if [d["genre"] for d in m["documents"]] == ["statement"]]
    assert no_minutes
    for m in no_minutes:
        assert m["availability_date"] == m["meeting_id"]


def test_real_statement_and_minutes_never_split(real_split):
    _, manifest, _ = data.load_corpus(data.CORPUS, data.MANIFEST)
    split_of = {d["document_id"]: m["split"]
                for m in real_split["meetings"] for d in m["documents"]}
    by_meeting = collections.defaultdict(set)
    for e in manifest["documents"]:
        meeting = e.get("statement_date") or e["meeting_end_date"]
        by_meeting[meeting].add(split_of[e["document_id"]])
    assert all(len(s) == 1 for s in by_meeting.values())
    assert sum(len(m["documents"]) == 2
               for m in real_split["meetings"]) == 91


# Synthetic snapshot -------------------------------------------------------

def _write_snapshot(tmp_path, docs):
    """docs: (genre, meeting_date, body, release_date or None)."""
    parts, entries = [], []
    for genre, date, body, release in docs:
        tag = "statement" if genre == "statement" else "minutes"
        doc_id = f"fomc-{tag}-{date}"
        parts.append(f"<|fomc_{tag}|>\ndate: {date}\ndocument_id: {doc_id}"
                     f"\n\n{body}\n<|end_fomc_{tag}|>\n")
        entry = {"document_id": doc_id,
                 "source_corpus": "statements" if tag == "statement"
                 else "minutes",
                 "normalized_characters": len(body)}
        if tag == "statement":
            entry["statement_date"] = date
        else:
            entry["meeting_end_date"] = date
            if release:
                entry["release_date"] = release
        entries.append(entry)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(parts), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "corpus_sha256": data.sha256_bytes(corpus.read_bytes()),
        "document_count": len(entries), "documents": entries}))
    return corpus, manifest


@pytest.fixture
def no_thresholds(monkeypatch):
    for name in ("MIN_F_DOCS", "MIN_F_PAIRS", "MIN_N_DOCS"):
        monkeypatch.setattr(make_split, name, 0)


def test_synthetic_meetings(tmp_path, no_thresholds):
    corpus, manifest = _write_snapshot(tmp_path, [
        # Statement alone would be T (46 days before the boundary), but the
        # minutes (+21 days) fall within 30 days: the whole meeting goes.
        ("statement", "2019-11-15", "Rates held.", None),
        ("minutes", "2019-11-15", "Members discussed rates.", None),
        ("statement", "2019-10-01", "Rates cut.", None),
        ("minutes", "2019-10-01",
         "Members agreed. Rates held; discussed sharply.", None),
        # Unscheduled meeting, no minutes: available on the meeting date.
        ("statement", "2020-03-15", "Rates cut sharply.", None),
        # Manifest release date overrides the 21-day rule.
        ("statement", "2021-10-20", "Rates held.", None),
        ("minutes", "2021-10-20", "Members discussed.", "2022-02-15"),
    ])
    out = make_split.build(corpus, manifest)
    meetings = {m["meeting_id"]: m for m in out["meetings"]}
    assert meetings["2019-11-15"]["split"] == "excluded"
    assert meetings["2019-11-15"]["availability_date"] == "2019-12-06"
    assert meetings["2019-10-01"]["split"] == "T"
    assert meetings["2020-03-15"]["split"] == "N"
    assert meetings["2020-03-15"]["availability_date"] == "2020-03-15"
    assert meetings["2021-10-20"]["split"] == "F"
    for m in ("2019-11-15", "2019-10-01", "2021-10-20"):
        assert [d["genre"] for d in meetings[m]["documents"]] == [
            "minutes", "statement"]
    assert out["counts"]["excluded"] == {
        "meetings": 1, "documents": 2, "statement": 1, "minutes": 1,
        "matched_pairs": 1}


def test_synthetic_normalized_before_char_check(tmp_path, no_thresholds):
    corpus, manifest = _write_snapshot(tmp_path, [
        ("statement", "2018-01-31", "Jorgensen spoke.", None),
        ("statement", "2023-01-31", "Jørgensen spo­ke.", None),
    ])
    out = make_split.build(corpus, manifest)
    f_doc = next(m for m in out["meetings"] if m["split"] == "F")
    assert f_doc["documents"][0]["body_code_points"] == len(
        "Jorgensen spoke.")


def test_synthetic_character_check_fails_closed(tmp_path, no_thresholds):
    corpus, manifest = _write_snapshot(tmp_path, [
        ("statement", "2018-01-31", "Rates held.", None),
        ("statement", "2023-01-31", "Rates held ☃.", None),
    ])
    with pytest.raises(InfeasibleError, match="U\\+2603"):
        make_split.build(corpus, manifest)


def test_synthetic_thresholds_fail_closed(tmp_path):
    corpus, manifest = _write_snapshot(tmp_path, [
        ("statement", "2018-01-31", "Rates held.", None),
    ])
    with pytest.raises(InfeasibleError, match="F documents 0 < 40"):
        make_split.build(corpus, manifest)
