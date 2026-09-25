#!/usr/bin/env python3
"""Regenerate the Week 3 v2 meeting-level time split (DESIGN.md v0.8, §2-3).

Reads the static corpus snapshot and its manifest, normalizes every document
(§3), parses the bodies, assigns each meeting to Train / Near / Far by its
availability date with the 30-day boundary exclusion (§2), runs the §2
feasibility thresholds and the §3 character-coverage check, and writes
split_manifest.json atomically. Fails closed (exit 1, existing output left
unchanged) if the corpus hash does not match its manifest, if the corpus
format is unexpected, or if any §2 / §3 check fails (§10). Corpus handling
(hash check, normalization, body parser) lives in data.py.

Output is deterministic: no timestamps, sorted keys, fixed float-free content,
so two runs on the same snapshot are byte-identical.

Usage:
    python make_split.py                      # writes ./split_manifest.json
    python make_split.py --output other.json
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from data import (CORPUS, GENRES, HERE, MANIFEST, load_corpus, normalize,
                  parse_documents, sha256_bytes)

OUTPUT = HERE / "split_manifest.json"
SCHEMA_VERSION = "1.0"
DESIGN_VERSION = "v0.4"

# §2
MINUTES_RELEASE_LAG_DAYS = 21
BOUNDARY_EXCLUSION_DAYS = 30
TRAIN_END = dt.date(2019, 12, 31)
NEAR_END = dt.date(2021, 12, 31)
MIN_F_DOCS = 40
MIN_F_PAIRS = 15
MIN_N_DOCS = 16


def availability_date(meeting_date: dt.date, genres: set[str],
                      minutes_release_date: str | None) -> dt.date:
    """§2: date the meeting's last document became public.

    v0.4: a meeting with no minutes is available on its meeting date.
    """
    if "minutes" in genres:
        if minutes_release_date:
            return dt.date.fromisoformat(minutes_release_date)
        return meeting_date + dt.timedelta(days=MINUTES_RELEASE_LAG_DAYS)
    return meeting_date


def assign_split(avail: dt.date) -> tuple[str, str | None]:
    for boundary in (TRAIN_END, NEAR_END):
        if abs((avail - boundary).days) <= BOUNDARY_EXCLUSION_DAYS:
            return "excluded", boundary.isoformat()
    if avail <= TRAIN_END:
        return "T", None
    if avail <= NEAR_END:
        return "N", None
    return "F", None


class InfeasibleError(ValueError):
    """A §2 threshold or the §3 character check failed (§10)."""


def build(corpus_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Return the split manifest.

    Raises ValueError on any failure (InfeasibleError for §10).
    """
    text, manifest, manifest_bytes = load_corpus(corpus_path, manifest_path)
    raw_docs = {d["document_id"]: d for d in parse_documents(text)}
    docs = {d["document_id"]: d
            for d in parse_documents(normalize(text))}
    manifest_ids = {d["document_id"] for d in manifest["documents"]}
    if (set(docs) != manifest_ids
            or len(docs) != manifest["document_count"]):
        raise ValueError("corpus documents do not match manifest")
    for entry in manifest["documents"]:
        raw_body = raw_docs[entry["document_id"]]["body"]
        if len(raw_body) != entry["normalized_characters"]:
            raise ValueError(
                f"body length mismatch for {entry['document_id']}")

    meetings: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"documents": {}, "minutes_release_date": None})
    for entry in manifest["documents"]:
        doc = docs[entry["document_id"]]
        genre = GENRES[entry["source_corpus"]]
        meeting_id = entry.get("statement_date") or entry["meeting_end_date"]
        if doc["genre"] != genre or doc["date"] != meeting_id:
            raise ValueError(
                f"manifest/corpus disagree for {entry['document_id']}")
        if genre in meetings[meeting_id]["documents"]:
            raise ValueError(
                f"two {genre} documents for meeting {meeting_id}")
        meetings[meeting_id]["documents"][genre] = entry["document_id"]
        if genre == "minutes":
            meetings[meeting_id]["minutes_release_date"] = entry.get(
                "release_date")

    records = []
    for meeting_id in sorted(meetings):
        m = meetings[meeting_id]
        avail = availability_date(dt.date.fromisoformat(meeting_id),
                                  set(m["documents"]),
                                  m["minutes_release_date"])
        split, boundary = assign_split(avail)
        records.append({
            "meeting_id": meeting_id,
            "split": split,
            "availability_date": avail.isoformat(),
            "excluded_boundary": boundary,
            "documents": [
                {"document_id": m["documents"][g], "genre": g,
                 "body_code_points": len(docs[m["documents"][g]]["body"])}
                for g in sorted(m["documents"])
            ],
        })

    counts: dict[str, dict[str, int]] = {}
    for split in ("T", "N", "F", "excluded"):
        rs = [r for r in records if r["split"] == split]
        genres = collections.Counter(
            d["genre"] for r in rs for d in r["documents"])
        counts[split] = {
            "meetings": len(rs),
            "documents": sum(genres.values()),
            "statement": genres["statement"],
            "minutes": genres["minutes"],
            "matched_pairs": sum(1 for r in rs if len(r["documents"]) == 2),
        }

    chars = {s: collections.Counter() for s in ("T", "N", "F", "excluded")}
    for r in records:
        for d in r["documents"]:
            chars[r["split"]].update(docs[d["document_id"]]["body"])
    t_vocab = set(chars["T"])
    offending = {
        s: {f"U+{ord(c):04X}": n
            for c, n in sorted(chars[s].items()) if c not in t_vocab}
        for s in ("N", "F")
    }

    failures = []
    f_docs, f_pairs = counts["F"]["documents"], counts["F"]["matched_pairs"]
    n_docs = counts["N"]["documents"]
    if f_docs < MIN_F_DOCS:
        failures.append(f"F documents {f_docs} < {MIN_F_DOCS}")
    if f_pairs < MIN_F_PAIRS:
        failures.append(f"F matched pairs {f_pairs} < {MIN_F_PAIRS}")
    if n_docs < MIN_N_DOCS:
        failures.append(f"N documents {n_docs} < {MIN_N_DOCS}")
    for s in ("N", "F"):
        if offending[s]:
            failures.append(
                f"{s} characters absent from T: {offending[s]}")

    out = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "source": {
            "corpus_file": corpus_path.name,
            "corpus_sha256": manifest["corpus_sha256"],
            "manifest_file": manifest_path.name,
            "manifest_sha256": sha256_bytes(manifest_bytes),
        },
        "rules": {
            "minutes_release_lag_days": MINUTES_RELEASE_LAG_DAYS,
            "boundary_exclusion_days": BOUNDARY_EXCLUSION_DAYS,
            "train_end": TRAIN_END.isoformat(),
            "near_end": NEAR_END.isoformat(),
            "normalization": (
                "NFD; drop Mn; drop Cf; map U+00F8->o, U+00D8->O, "
                "delete U+00AE U+2122 U+00A9; NFC"),
        },
        "counts": counts,
        "t_character_vocab": [f"U+{ord(c):04X}" for c in sorted(t_vocab)],
        "meetings": records,
    }
    if failures:
        raise InfeasibleError("; ".join(failures))
    return out


def report(split: dict[str, Any]) -> None:
    for s, c in split["counts"].items():
        print(f"  {s:9s} " + " ".join(f"{k}={v}" for k, v in c.items()))
    by_year = collections.Counter()
    for r in split["meetings"]:
        for d in r["documents"]:
            by_year[(r["split"], r["meeting_id"][:4], d["genre"])] += 1
    for s in ("T", "N", "F", "excluded"):
        years = sorted({y for (ss, y, _) in by_year if ss == s})
        print(f"  {s} by meeting year: " + ", ".join(
            f"{y} S{by_year[(s, y, 'statement')]}"
            f"/M{by_year[(s, y, 'minutes')]}" for y in years))
    for r in split["meetings"]:
        if r["split"] == "excluded":
            print(f"  excluded {r['meeting_id']} "
                  f"avail={r['availability_date']} "
                  f"boundary={r['excluded_boundary']}")
    lengths = [d["body_code_points"]
               for r in split["meetings"] for d in r["documents"]]
    print(f"  body code points: min={min(lengths)} "
          f"median={statistics.median(lengths)} max={max(lengths)}")
    print(f"  T character vocab size: {len(split['t_character_vocab'])}")


def write_atomic(path: Path, text: str) -> None:
    """Write to a temp file in the same directory, then os.replace.

    No temp file survives a failure.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        split = build(args.corpus, args.manifest)
    except InfeasibleError as e:
        print(f"INFEASIBLE (DESIGN.md §10): {e}")
        return 1
    except ValueError as e:
        print(f"FAILED: {e}")
        return 1
    report(split)
    write_atomic(args.output,
                 json.dumps(split, indent=2, sort_keys=True) + "\n")
    digest = sha256_bytes(args.output.read_bytes())
    print(f"FEASIBLE; wrote {args.output.name} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
