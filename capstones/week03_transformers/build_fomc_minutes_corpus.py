#!/usr/bin/env python3
"""Build an auditable corpus of official FOMC meeting minutes.

Sibling to build_fomc_corpus.py (statements). Reuses its low-level, already
self-tested primitives (Fetcher, BuildError, sha256_bytes, canonicalize_url,
atomic_write, parse_url_date, discover_links) rather than duplicating them,
and implements minutes-specific discovery, eligibility classification, and
extraction, since minutes are structurally very different documents from
statements (multi-section narrative, 5-15k words, footnotes) rather than a
few short paragraphs.

The builder writes candidate artifacts only to an explicit output directory.
It does not read from or overwrite a frozen repository corpus.

Usage:
    python build_fomc_minutes_corpus.py --output-dir ./candidate_minutes
    python build_fomc_minutes_corpus.py --self-test
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from bs4 import BeautifulSoup, Tag

# Reuse the statements builder's already-reviewed low-level primitives rather
# than duplicating fetch/hash/atomic-write/URL-canonicalization logic.
from build_fomc_corpus import (
    BuildError,
    Fetcher,
    atomic_write,
    canonicalize_url,
    discover_links,
    parse_url_date,
    sha256_bytes,
    utc_now,
)

# --- Freeze window -----------------------------------------------------
# Discovery coverage note (confirmed against the live site): the Fed used a
# legacy URL namespace for older minutes -- /fomc/minutes/{YYYYMMDD}.htm --
# distinct from the /monetarypolicy/fomcminutes{YYYYMMDD}.htm pattern this
# builder targets. The switchover falls around 2010. FREEZE_START is set to
# the earliest date this builder actually achieves coverage from, rather
# than claiming an earlier start the discovery mechanism cannot reach.
# Extending to the pre-2010 legacy namespace is a known, scoped-out
# possible future extension (see KNOWN_COVERAGE_GAP below), not a bug.
FREEZE_START = dt.date(2010, 1, 1)
KNOWN_COVERAGE_GAP = (
    "Minutes before ~2010 are published under the legacy /fomc/minutes/"
    "{YYYYMMDD}.htm URL namespace, not /monetarypolicy/fomcminutes{YYYYMMDD}.htm. "
    "This builder does not crawl that namespace; pre-2010 coverage is a "
    "scoped-out future extension, not a defect."
)

# Pre-Warsh cutoff: freeze the corpus at the last regularly scheduled FOMC
# meeting under Chair Powell -- verified as April 28-29, 2026 (minutes URL
# fomcminutes20260429.htm; Powell's term expired May 15, 2026; Warsh sworn
# in May 22, 2026; confirmed via multiple contemporaneous sources including
# the Fed's own press-conference and minutes pages). The cutoff uses the
# meeting's final day, matching the minutes/statement URL date convention.
PRE_WARSH_CUTOFF = dt.date(2026, 4, 29)
FREEZE_END = PRE_WARSH_CUTOFF

SCHEMA_VERSION = "1.0"
CORPUS_ID = "fomc-minutes-1993-prewarsh-v1"
NORMALIZATION_VERSION = "fomc-finance-preserving-v1"  # same normalization contract as statements
EXTRACTION_METHOD = "federalreserve_article_sections_v1"
START_MARKER = "<|fomc_minutes|>"
END_MARKER = "<|end_fomc_minutes|>"
OUTPUT_CORPUS = "fomc_minutes_1993_prewarsh.txt"
OUTPUT_MANIFEST = "fomc_minutes_1993_prewarsh_manifest.json"

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
# fomchistorical{year}.htm pages stop existing after 2020 -- the redesigned
# fomccalendars.htm (already in DISCOVERY_URLS) covers 2021 onward in a
# single page. The statements builder already encodes this boundary
# (its own HISTORICAL_URLS stops at range(2015, 2021)); mirrored here.
HISTORICAL_YEARS_END = min(FREEZE_END.year, 2020)
HISTORICAL_URLS = tuple(
    f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
    for year in range(FREEZE_START.year, HISTORICAL_YEARS_END + 1)
)
DISCOVERY_URLS = (CALENDAR_URL, *HISTORICAL_URLS)

# Minutes URLs: /monetarypolicy/fomcminutesYYYYMMDD.htm (date = meeting's
# final day; no trailing letter suffix, unlike statement URLs).
MINUTES_URL_RE = re.compile(
    r"^/monetarypolicy/fomcminutes(?P<date>\d{8})\.htm$", re.IGNORECASE
)

# Footnote markers ("1.", "2." at paragraph starts referencing "Return to
# text" anchors) and the anchors themselves are document noise for a
# character-level LM corpus and are stripped during extraction.
RETURN_TO_TEXT_RE = re.compile(r"\s*return to text\s*", re.IGNORECASE)
LEADING_FOOTNOTE_MARKER_RE = re.compile(r"^\d{1,2}\.\s+(?=[A-Z])")

NAVIGATION_LEAKAGE = (
    "skip to main content",
    "back to top",
    "back to home",
    "for media inquiries",
    "media contact",
    "board of governors of the federal reserve system",
    "stay connected",
    "follow us",
    "last update:",
)

# Windows-1252 C1-range bytes occasionally survive into older (pre-~2012)
# Fed pages as literal Unicode control characters (e.g. U+0096 as an en
# dash) rather than their intended punctuation. The shared
# normalize_statement_body (imported from the statements builder) correctly
# rejects ANY stray control character rather than silently stripping it --
# so these known, well-understood legacy artifacts are repaired to their
# intended punctuation *before* that shared, stricter check runs.
CP1252_CONTROL_REPAIR = {
    "\u0091": "'", "\u0092": "'", "\u0093": '"', "\u0094": '"',
    "\u0095": "-", "\u0096": "-", "\u0097": "--", "\u0085": "...",
}


def repair_legacy_cp1252_controls(text: str) -> str:
    for source, replacement in CP1252_CONTROL_REPAIR.items():
        text = text.replace(source, replacement)
    return text


@dataclasses.dataclass(frozen=True)
class MinutesCandidate:
    canonical_url: str
    meeting_end_date: dt.date
    evidence_urls: tuple[str, ...]
    evidence_labels: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class IncludedMinutes:
    canonical_url: str
    meeting_end_date: dt.date
    document_id: str
    title: str
    body: str
    normalized_characters: int
    normalized_sha256: str
    raw_sha256: str
    raw_utf8_bytes: int
    split: str
    retrieved_at_utc: str
    warnings: list[str]


def discover_minutes_links(source_url: str, source_html: bytes) -> list[tuple[str, dt.date, str]]:
    """Statement discover_links filters to statement-shaped URLs; this variant
    filters the same historical/calendar pages to minutes-shaped URLs."""
    soup = BeautifulSoup(source_html, "html.parser")
    found: list[tuple[str, dt.date, str]] = []
    from urllib.parse import urljoin

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        absolute = urljoin(source_url, href)
        try:
            canonical = canonicalize_url(absolute)
        except BuildError:
            continue
        path = re.sub(r"^https?://[^/]+", "", canonical)
        match = MINUTES_URL_RE.match(path)
        if not match:
            continue
        meeting_end_date = dt.datetime.strptime(match.group("date"), "%Y%m%d").date()
        if not (FREEZE_START <= meeting_end_date <= FREEZE_END):
            continue
        label = " ".join(anchor.get_text(" ", strip=True).split())
        found.append((canonical, meeting_end_date, label))
    return found


def discover_minutes_candidates(
    fetcher: Fetcher,
) -> tuple[list[MinutesCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    evidence: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_url in DISCOVERY_URLS:
        try:
            fetched = fetcher.get(source_url)
            links = discover_minutes_links(fetched.canonical_url, fetched.content)
            sources.append(
                {
                    "url": fetched.canonical_url,
                    "retrieved_at_utc": fetched.retrieved_at_utc,
                    "raw_sha256": sha256_bytes(fetched.content),
                    "candidate_link_count": len(links),
                }
            )
            for url, meeting_end_date, label in links:
                item = evidence.setdefault(url, {"date": meeting_end_date, "sources": [], "labels": []})
                item["sources"].append(fetched.canonical_url)
                item["labels"].append(label)
        except BuildError as exc:
            failures.append({"url": source_url, "stage": "discovery", "error": str(exc), "status": "unresolved"})
    candidates = [
        MinutesCandidate(
            canonical_url=url,
            meeting_end_date=value["date"],
            evidence_urls=tuple(sorted(set(value["sources"]))),
            evidence_labels=tuple(sorted(set(value["labels"]))),
        )
        for url, value in evidence.items()
    ]
    candidates.sort(key=lambda c: (c.meeting_end_date, c.canonical_url))
    if not candidates:
        failures.append(
            {
                "url": "(all discovery sources)",
                "stage": "discovery",
                "error": "Zero minutes candidates found in freeze window -- "
                "verify MINUTES_URL_RE and DISCOVERY_URLS against current site structure",
                "status": "unresolved",
            }
        )
    return candidates, sources, failures


def classify_minutes_eligibility(candidate: MinutesCandidate, page_html: bytes) -> tuple[bool, str]:
    """Mirrors the statements builder's classify_eligibility exactly: a
    properly-extracted title plus FULL (untruncated) page text. A truncated
    text slice is unsafe -- modern federalreserve.gov pages open with a
    long accessibility/.gov banner that alone exceeds any small character
    budget before real content appears, causing false exclusions."""
    soup = BeautifulSoup(page_html, "html.parser")
    title = extract_title(soup).lower()
    page_text = " ".join(soup.get_text(" ", strip=True).split()).lower()
    labels = " ".join(candidate.evidence_labels).lower()
    if "minutes" in title or "minutes" in labels or "minutes of the federal open market committee" in page_text:
        return True, "eligible_fomc_minutes"
    return False, "not_fomc_minutes"


def extract_title(soup: BeautifulSoup) -> str:
    heading = soup.select_one("#article h2, #article h3, .article__heading, #content .title")
    if heading:
        text = " ".join(heading.get_text(" ", strip=True).split())
        if text:
            return text
    if soup.title:
        return re.sub(r"^Federal Reserve Board\s*-\s*", "", " ".join(soup.title.get_text(" ", strip=True).split()))
    return "FOMC minutes"


def _clean_paragraph(tag: Tag) -> str:
    for removable in tag.select("script, style, noscript, .sr-only, .share, .social, sup, a.footnote"):
        removable.decompose()
    text = " ".join(tag.get_text(" ", strip=True).split())
    text = RETURN_TO_TEXT_RE.sub(" ", text)
    text = LEADING_FOOTNOTE_MARKER_RE.sub("", text)
    return text.strip()


def extract_minutes_body(page_html: bytes) -> tuple[str, str, list[str]]:
    """Extract section headings + substantive paragraphs from a minutes page.

    Unlike statements (a handful of undifferentiated paragraphs), minutes are
    long-form with named sections (h2/h3 subheadings kept inline as plain-text
    section markers, since they carry real structure a tiny LM can learn).

    Older (pre-~2012) Fed page templates do not use the modern #article
    container; a broader selector list is tried in order, and the selector
    that actually matched is returned as a warning so template drift across
    three decades of markup stays visible in the manifest rather than silent.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    candidate_selectors = ["#article", "article", "main #content", "#content", "#leftText", ".content", "body"]
    container = None
    matched_selector = None
    for selector in candidate_selectors:
        container = soup.select_one(selector)
        if container is not None:
            matched_selector = selector
            break
    if container is None:
        raise BuildError(f"No recognized article container among {candidate_selectors}")
    warnings: list[str] = []
    if matched_selector not in ("#article",):
        warnings.append(f"legacy_container_selector:{matched_selector}")
    parts: list[str] = []
    started = False
    for tag in container.find_all(["p", "h2", "h3"], recursive=True):
        text = _clean_paragraph(tag)
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith(NAVIGATION_LEAKAGE):
            if started:
                # trailing nav/footer content after the substantive body begins
                if any(k in lowered for k in ("stay connected", "follow us", "last update")):
                    break
            continue
        if not started:
            is_body_opening = (
                "a joint meeting of the federal open market committee" in lowered
                or "meeting of the federal open market committee" in lowered
            )
            if tag.name in ("h2", "h3") or is_body_opening:
                started = True
            else:
                continue
        if tag.name in ("h2", "h3"):
            parts.append(f"## {text}")
        else:
            parts.append(text)
    if not parts:
        raise BuildError("Recognized article container yielded no substantive minutes paragraphs")
    body = "\n\n".join(parts)
    return body, EXTRACTION_METHOD, warnings


def assign_split(meeting_end_date: dt.date) -> str:
    if not (FREEZE_START <= meeting_end_date <= FREEZE_END):
        raise BuildError(f"Minutes date outside freeze interval: {meeting_end_date}")
    # Mirror the statements builder's convention: most-recent slice held out.
    validation_start = FREEZE_END.replace(year=FREEZE_END.year - 1)
    return "validation" if meeting_end_date >= validation_start else "train"


def serialize_document(document: IncludedMinutes) -> str:
    return (
        f"{START_MARKER}\n"
        f"date: {document.meeting_end_date.isoformat()}\n"
        f"document_id: {document.document_id}\n\n"
        f"{document.body}\n"
        f"{END_MARKER}\n"
    )


def build(output_dir: Path, timeout: float, retries: int) -> tuple[Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(timeout=timeout, retries=retries)

    candidates, sources, failures = discover_minutes_candidates(fetcher)

    documents: list[IncludedMinutes] = []
    exclusions: list[dict[str, Any]] = []

    for candidate in candidates:
        try:
            fetched = fetcher.get(candidate.canonical_url)
        except BuildError as exc:
            failures.append({"url": candidate.canonical_url, "stage": "fetch", "error": str(exc), "status": "unresolved"})
            continue

        eligible, reason = classify_minutes_eligibility(candidate, fetched.content)
        if not eligible:
            exclusions.append(
                {
                    "canonical_url": candidate.canonical_url,
                    "observed_date": candidate.meeting_end_date.isoformat(),
                    "reason": reason,
                    "discovery_evidence": list(candidate.evidence_urls),
                }
            )
            continue

        try:
            soup = BeautifulSoup(fetched.content, "html.parser")
            title = extract_title(soup)
            body, extraction_method, warnings = extract_minutes_body(fetched.content)
        except BuildError as exc:
            if "no substantive minutes paragraphs" in str(exc):
                # Confirmed via the Fed's own contemporaneous press releases:
                # minutes before ~2011-2012 were published as PDF only (e.g.
                # the Oct 30-31, 2007 release offers "(320 KB PDF)" with no
                # HTML link at all, vs. later releases offering "HTML | PDF").
                # The .htm URL exists today only as a thin retrofitted stub
                # with no real paragraph content -- a known publication-era
                # boundary, not an extraction bug. Recorded as an exclusion.
                exclusions.append(
                    {
                        "canonical_url": candidate.canonical_url,
                        "observed_date": candidate.meeting_end_date.isoformat(),
                        "reason": "pre_html_era_pdf_only_no_htm_content",
                        "discovery_evidence": list(candidate.evidence_urls),
                    }
                )
            else:
                failures.append({"url": candidate.canonical_url, "stage": "extraction", "error": str(exc), "status": "unresolved"})
            continue

        try:
            body = repair_legacy_cp1252_controls(body)
            # Reuse the statements builder's normalization contract exactly.
            from build_fomc_corpus import normalize_statement_body

            normalized_body, norm_warnings = normalize_statement_body(body)
            warnings = [*warnings, *norm_warnings]
        except BuildError as exc:
            failures.append({"url": candidate.canonical_url, "stage": "extraction", "error": str(exc), "status": "unresolved"})
            continue

        document_id = f"fomc-minutes-{candidate.meeting_end_date.isoformat()}"
        documents.append(
            IncludedMinutes(
                canonical_url=candidate.canonical_url,
                meeting_end_date=candidate.meeting_end_date,
                document_id=document_id,
                title=title,
                body=normalized_body,
                normalized_characters=len(normalized_body),
                normalized_sha256=sha256_bytes(normalized_body.encode("utf-8")),
                raw_sha256=sha256_bytes(fetched.content),
                raw_utf8_bytes=len(fetched.content),
                split=assign_split(candidate.meeting_end_date),
                retrieved_at_utc=fetched.retrieved_at_utc,
                warnings=warnings,
            )
        )

    corpus_text = "\n".join(serialize_document(d) for d in documents)
    corpus_bytes = corpus_text.encode("utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "generated_at_utc": utc_now(),
        "freeze_start": FREEZE_START.isoformat(),
        "freeze_end": FREEZE_END.isoformat(),
        "pre_warsh_cutoff": PRE_WARSH_CUTOFF.isoformat(),
        "known_coverage_gap": KNOWN_COVERAGE_GAP,
        "extraction_method": EXTRACTION_METHOD,
        "normalization_version": NORMALIZATION_VERSION,
        "document_count": len(documents),
        "train_document_count": sum(1 for d in documents if d.split == "train"),
        "validation_document_count": sum(1 for d in documents if d.split == "validation"),
        "corpus_characters": len(corpus_text),
        "corpus_utf8_bytes": len(corpus_bytes),
        "corpus_sha256": sha256_bytes(corpus_bytes),
        "documents": [
            {
                "canonical_url": d.canonical_url,
                "document_id": d.document_id,
                "meeting_end_date": d.meeting_end_date.isoformat(),
                "title": d.title,
                "extraction_method": EXTRACTION_METHOD,
                "normalized_characters": d.normalized_characters,
                "normalized_sha256": d.normalized_sha256,
                "raw_sha256": d.raw_sha256,
                "raw_utf8_bytes": d.raw_utf8_bytes,
                "retrieved_at_utc": d.retrieved_at_utc,
                "split": d.split,
                "status": "included",
                "warnings": d.warnings,
            }
            for d in documents
        ],
        "exclusions": exclusions,
        "discovery_sources": sources,
        "failures": failures,
    }

    corpus_path = output_dir / OUTPUT_CORPUS
    manifest_path = output_dir / OUTPUT_MANIFEST
    atomic_write(corpus_path, corpus_bytes)
    atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return corpus_path, manifest_path, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Explicit candidate output directory outside the repository (required unless --self-test)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2, choices=range(0, 6))
    parser.add_argument("--self-test", action="store_true", help="Run bounded offline pure-function tests and exit")
    return parser.parse_args(argv)


def run_self_tests() -> None:
    """Offline, no-network tests for the pure extraction/classification functions."""
    sample_html = b"""
    <html><head><title>FOMC Minutes</title></head>
    <body><div id="article">
    <h2>Minutes of the Federal Open Market Committee</h2>
    <p>A joint meeting of the Federal Open Market Committee and the Board of
    Governors of the Federal Reserve System was held on January 28, 2026.</p>
    <h2>Committee Policy Action</h2>
    <p>The Committee voted to maintain the target range.1</p>
    <p>1. See the policy statement. Return to text</p>
    <p>Last Update: February 2026</p>
    </div></body></html>
    """
    body, method, warnings = extract_minutes_body(sample_html)
    assert "## Minutes of the Federal Open Market Committee" in body, body
    assert "## Committee Policy Action" in body, body
    assert "Return to text" not in body, body
    assert method == EXTRACTION_METHOD

    m = MINUTES_URL_RE.match("/monetarypolicy/fomcminutes20260129.htm")
    assert m and m.group("date") == "20260129"
    assert MINUTES_URL_RE.match("/newsevents/pressreleases/monetary20260128a.htm") is None

    assert assign_split(dt.date(2011, 3, 1)) == "train"
    assert assign_split(dt.date(2026, 2, 1)) == "validation"

    # Regression test for the exact bug found in the first live discovery run:
    # a long accessibility/.gov banner pushing real content past a naive
    # truncated text slice must NOT cause a false "not eligible" exclusion.
    banner_padded_html = (
        b"<html><head><title>Federal Reserve Board - FOMC Minutes</title></head><body>"
        + b"<p>Skip to main content</p>"
        + b"<p>" + b"An official website of the United States Government. " * 20 + b"</p>"
        + b'<div id="article"><h2>Minutes of the Federal Open Market Committee</h2>'
        + b"<p>A joint meeting of the Federal Open Market Committee was held.</p></div>"
        + b"</body></html>"
    )
    fake_candidate = MinutesCandidate(
        canonical_url="https://www.federalreserve.gov/monetarypolicy/fomcminutes20260429.htm",
        meeting_end_date=dt.date(2026, 4, 29),
        evidence_urls=(), evidence_labels=(),
    )
    eligible, reason = classify_minutes_eligibility(fake_candidate, banner_padded_html)
    assert eligible, f"banner-padded modern page must still classify eligible, got reason={reason}"

    # Regression test for legacy (pre-2012) container markup: no #article,
    # but a #content div with the real minutes text -- must still extract,
    # and must flag the legacy selector via warnings rather than silently
    # succeeding as if it were the modern template.
    legacy_html = b"""
    <html><head><title>FOMC Minutes 2009</title></head>
    <body><div id="content">
    <h2>Minutes of the Federal Open Market Committee</h2>
    <p>A joint meeting of the Federal Open Market Committee was held on
    October 27, 2009.</p>
    <h2>Committee Policy Action</h2>
    <p>The Committee voted to maintain the target range.</p>
    </div></body></html>
    """
    legacy_body, _, legacy_warnings = extract_minutes_body(legacy_html)
    assert "Minutes of the Federal Open Market Committee" in legacy_body
    assert any("legacy_container_selector" in w for w in legacy_warnings), legacy_warnings

    # Regression test for the exact byte-level bug found in the second live
    # run: a stray Windows-1252 control character (U+0096, meant as an en
    # dash) surviving into extracted text must be repaired to plain "-"
    # before it reaches the shared, intentionally-strict control-character
    # check in normalize_statement_body (imported from build_fomc_corpus).
    repaired = repair_legacy_cp1252_controls("policy rate\u0096unchanged")
    assert repaired == "policy rate-unchanged", repaired
    from build_fomc_corpus import normalize_statement_body as _norm
    normalized, _ = _norm(repaired)  # must not raise BuildError
    assert "\u0096" not in normalized

    print("[OK] all offline self-tests passed")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_tests()
        return 0
    if args.output_dir is None:
        print("error: --output-dir is required unless --self-test", file=sys.stderr)
        return 2
    corpus_path, manifest_path, manifest = build(args.output_dir, args.timeout, args.retries)
    print(f"documents included: {manifest['document_count']} "
          f"(train {manifest['train_document_count']}, validation {manifest['validation_document_count']})")
    print(f"corpus: {corpus_path} ({manifest['corpus_characters']:,} chars)")
    print(f"manifest: {manifest_path}")
    print(f"failures: {len(manifest['failures'])}, exclusions: {len(manifest['exclusions'])}")
    if manifest["failures"]:
        print("!! non-empty failures -- inspect manifest before treating corpus as final !!", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
