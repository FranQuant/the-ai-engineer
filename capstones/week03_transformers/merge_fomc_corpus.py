#!/usr/bin/env python3
"""Merge the FOMC statements and minutes corpora into one combined,
manifest-tracked training corpus for the Week 3 capstone.

Deliberately simple by design: each source corpus already implements its
own discovery, extraction, normalization, and train/validation split logic
(reviewed and self-tested independently). This script does not re-derive
any of that -- it concatenates the two already-serialized corpus files
(each internally document-delimited with its own START/END markers) and
combines their manifests into one provenance record. Per-document splits
are inherited as-is from each source manifest.

Usage:
    python merge_fomc_corpus.py \\
        --statements-corpus fomc_statements_2015_2025.txt \\
        --statements-manifest fomc_statements_2015_2025_manifest.json \\
        --minutes-corpus /tmp/fomc_minutes_candidate/fomc_minutes_1993_prewarsh.txt \\
        --minutes-manifest /tmp/fomc_minutes_candidate/fomc_minutes_1993_prewarsh_manifest.json \\
        --output-dir .

    python merge_fomc_corpus.py --self-test
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "1.0"
CORPUS_ID = "fomc-combined-training-corpus-v1"
OUTPUT_CORPUS = "fomc_training_corpus.txt"
OUTPUT_MANIFEST = "fomc_training_corpus_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge(
    statements_corpus: Path,
    statements_manifest: Path,
    minutes_corpus: Path,
    minutes_manifest: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    stmt_text = statements_corpus.read_text(encoding="utf-8")
    min_text = minutes_corpus.read_text(encoding="utf-8")
    stmt_manifest = load_manifest(statements_manifest)
    min_manifest = load_manifest(minutes_manifest)

    # Integrity check: the source files must match the hashes their own
    # manifests recorded at generation time, so a merge never silently
    # combines a stale or hand-edited corpus file with its manifest.
    stmt_actual_sha = sha256_bytes(stmt_text.encode("utf-8"))
    if stmt_actual_sha != stmt_manifest["corpus_sha256"]:
        raise SystemExit(
            f"Statements corpus file does not match its manifest's recorded sha256 "
            f"(file={stmt_actual_sha}, manifest={stmt_manifest['corpus_sha256']}) -- "
            f"re-run build_fomc_corpus.py or point at the matching manifest."
        )
    min_actual_sha = sha256_bytes(min_text.encode("utf-8"))
    if min_actual_sha != min_manifest["corpus_sha256"]:
        raise SystemExit(
            f"Minutes corpus file does not match its manifest's recorded sha256 "
            f"(file={min_actual_sha}, manifest={min_manifest['corpus_sha256']}) -- "
            f"re-run build_fomc_minutes_corpus.py or point at the matching manifest."
        )

    # Duplicate-ID check across the two sources (should never happen given
    # the disjoint document_id prefixes "fomc-YYYY-MM-DD" vs
    # "fomc-minutes-YYYY-MM-DD", but checked explicitly rather than assumed).
    stmt_ids = {d["document_id"] for d in stmt_manifest["documents"]}
    min_ids = {d["document_id"] for d in min_manifest["documents"]}
    overlap = stmt_ids & min_ids
    if overlap:
        raise SystemExit(f"Duplicate document_id(s) across sources: {sorted(overlap)}")

    combined_text = stmt_text.rstrip("\n") + "\n\n" + min_text.lstrip("\n")
    combined_bytes = combined_text.encode("utf-8")

    combined_documents = [
        {**d, "source_corpus": "statements"} for d in stmt_manifest["documents"]
    ] + [
        {**d, "source_corpus": "minutes"} for d in min_manifest["documents"]
    ]
    train_count = sum(1 for d in combined_documents if d["split"] == "train")
    validation_count = sum(1 for d in combined_documents if d["split"] == "validation")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "statements": {
                "corpus_id": stmt_manifest["corpus_id"],
                "corpus_sha256": stmt_manifest["corpus_sha256"],
                "freeze_start": stmt_manifest["freeze_start"],
                "freeze_end": stmt_manifest["freeze_end"],
                "document_count": stmt_manifest["document_count"],
            },
            "minutes": {
                "corpus_id": min_manifest["corpus_id"],
                "corpus_sha256": min_manifest["corpus_sha256"],
                "freeze_start": min_manifest["freeze_start"],
                "freeze_end": min_manifest["freeze_end"],
                "document_count": min_manifest["document_count"],
                "known_coverage_gap": min_manifest.get("known_coverage_gap"),
            },
        },
        "document_count": len(combined_documents),
        "train_document_count": train_count,
        "validation_document_count": validation_count,
        "corpus_characters": len(combined_text),
        "corpus_utf8_bytes": len(combined_bytes),
        "corpus_sha256": sha256_bytes(combined_bytes),
        "documents": combined_documents,
    }

    corpus_path = output_dir / OUTPUT_CORPUS
    manifest_path = output_dir / OUTPUT_MANIFEST
    corpus_path.write_bytes(combined_bytes)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return corpus_path, manifest_path, manifest


def run_self_tests(tmp_dir: Path) -> None:
    """Offline test using synthetic corpus/manifest pairs (no real files needed)."""
    stmt_text = "<|fomc_statement|>\ndate: 2020-01-01\n\nHello statement.\n<|end_fomc_statement|>\n"
    min_text = "<|fomc_minutes|>\ndate: 2020-01-15\n\nHello minutes.\n<|end_fomc_minutes|>\n"
    stmt_manifest = {
        "corpus_id": "test-statements", "corpus_sha256": sha256_bytes(stmt_text.encode()),
        "freeze_start": "2020-01-01", "freeze_end": "2020-01-01",
        "document_count": 1,
        "documents": [{"document_id": "fomc-2020-01-01", "split": "train"}],
    }
    min_manifest = {
        "corpus_id": "test-minutes", "corpus_sha256": sha256_bytes(min_text.encode()),
        "freeze_start": "2020-01-01", "freeze_end": "2020-01-15",
        "document_count": 1, "known_coverage_gap": "test gap note",
        "documents": [{"document_id": "fomc-minutes-2020-01-15", "split": "validation"}],
    }
    (tmp_dir / "s.txt").write_text(stmt_text)
    (tmp_dir / "s.json").write_text(json.dumps(stmt_manifest))
    (tmp_dir / "m.txt").write_text(min_text)
    (tmp_dir / "m.json").write_text(json.dumps(min_manifest))

    corpus_path, manifest_path, manifest = merge(
        tmp_dir / "s.txt", tmp_dir / "s.json", tmp_dir / "m.txt", tmp_dir / "m.json", tmp_dir / "out"
    )
    assert manifest["document_count"] == 2
    assert manifest["train_document_count"] == 1
    assert manifest["validation_document_count"] == 1
    assert "Hello statement." in corpus_path.read_text()
    assert "Hello minutes." in corpus_path.read_text()
    assert manifest["sources"]["minutes"]["known_coverage_gap"] == "test gap note"

    # Integrity check: a hand-edited corpus (mismatched sha256) must be rejected.
    (tmp_dir / "s.txt").write_text(stmt_text + "TAMPERED")
    try:
        merge(tmp_dir / "s.txt", tmp_dir / "s.json", tmp_dir / "m.txt", tmp_dir / "m.json", tmp_dir / "out2")
        raise AssertionError("tampered corpus should have been rejected")
    except SystemExit as exc:
        assert "does not match its manifest" in str(exc)

    print("[OK] all offline self-tests passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statements-corpus", type=Path)
    p.add_argument("--statements-manifest", type=Path)
    p.add_argument("--minutes-corpus", type=Path)
    p.add_argument("--minutes-manifest", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            run_self_tests(Path(td))
        return 0
    required = [args.statements_corpus, args.statements_manifest, args.minutes_corpus, args.minutes_manifest, args.output_dir]
    if any(r is None for r in required):
        print("error: all of --statements-corpus/--statements-manifest/--minutes-corpus/"
              "--minutes-manifest/--output-dir are required unless --self-test", file=sys.stderr)
        return 2
    corpus_path, manifest_path, manifest = merge(
        args.statements_corpus, args.statements_manifest,
        args.minutes_corpus, args.minutes_manifest, args.output_dir,
    )
    print(f"documents: {manifest['document_count']} "
          f"(train {manifest['train_document_count']}, validation {manifest['validation_document_count']})")
    print(f"corpus: {corpus_path} ({manifest['corpus_characters']:,} chars, {manifest['corpus_utf8_bytes']:,} bytes)")
    print(f"manifest: {manifest_path}")
    print(f"corpus_sha256: {manifest['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
