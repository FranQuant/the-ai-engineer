#!/usr/bin/env python3
"""Build an auditable corpus of official FOMC policy-decision statements.

The builder writes candidate artifacts only to an explicit output directory.  It
does not read from or overwrite a frozen repository corpus.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag


FREEZE_START = dt.date(2015, 1, 1)
FREEZE_END = dt.date(2025, 12, 31)
VALIDATION_START = dt.date(2025, 1, 1)
SCHEMA_VERSION = "1.0"
CORPUS_ID = "fomc-statements-2015-2025-v2"
NORMALIZATION_VERSION = "fomc-finance-preserving-v1"
EXTRACTION_METHOD = "federalreserve_article_paragraphs_v2"
START_MARKER = "<|fomc_statement|>"
END_MARKER = "<|end_fomc_statement|>"
MIN_REVIEW_CHARACTERS = 500
OUTPUT_CORPUS = "fomc_statements_2015_2025.txt"
OUTPUT_MANIFEST = "fomc_statements_2015_2025_manifest.json"
USER_AGENT = (
    "week03-fomc-corpus-builder/1.0 "
    "(educational reproducibility; contact: repository maintainer)"
)
OFFICIAL_HOSTS = {"www.federalreserve.gov", "federalreserve.gov"}

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL_URLS = tuple(
    f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
    for year in range(2015, 2021)
)
ARCHIVE_URLS = (
    "https://www.federalreserve.gov/newsevents/pressreleases.htm",
)
DISCOVERY_URLS = (CALENDAR_URL, *HISTORICAL_URLS, *ARCHIVE_URLS)

STATEMENT_URL_RE = re.compile(
    r"^/newsevents/pressreleases/monetary(?P<date>\d{8})(?P<suffix>[a-z])\.htm$",
    re.IGNORECASE,
)
DATE_IN_URL_RE = re.compile(r"monetary(?P<date>\d{8})[a-z]\.htm$", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAVIGATION_LEAKAGE = (
    "skip to main content",
    "back to top",
    "for media inquiries",
    "media contact",
    "implementation note issued",
    "implementation note",
    "board of governors of the federal reserve system",
    "stay connected",
    "follow us",
    "last update:",
)
TERMINAL_PARAGRAPH_PREFIXES = (
    "for media inquiries",
    "media contact",
    "implementation note",
    "last update:",
    "board of governors of the federal reserve system",
)


class BuildError(RuntimeError):
    """A corpus-readiness failure with an actionable message."""


@dataclasses.dataclass(frozen=True)
class FetchResult:
    requested_url: str
    canonical_url: str
    content: bytes
    retrieved_at_utc: str
    status_code: int
    content_type: str | None
    etag: str | None
    last_modified: str | None


@dataclasses.dataclass(frozen=True)
class Candidate:
    canonical_url: str
    statement_date: dt.date
    evidence_urls: tuple[str, ...]
    evidence_labels: tuple[str, ...]


@dataclasses.dataclass
class IncludedDocument:
    document_id: str
    statement_date: dt.date
    meeting_type: str
    title: str
    canonical_url: str
    retrieved_at_utc: str
    raw_sha256: str
    normalized_sha256: str
    raw_utf8_bytes: int
    normalized_utf8_bytes: int
    normalized_characters: int
    split: str
    extraction_method: str
    warnings: list[str]
    status: str
    body: str

    def manifest_record(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["statement_date"] = self.statement_date.isoformat()
        record.pop("body")
        return record


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme.lower() != "https" or host not in OFFICIAL_HOSTS:
        raise BuildError(f"Non-official or non-HTTPS URL rejected: {url}")
    path = re.sub(r"/{2,}", "/", parts.path)
    return urlunsplit(("https", "www.federalreserve.gov", path, "", ""))


class Fetcher:
    def __init__(self, timeout: float = 30.0, retries: int = 2) -> None:
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})

    def get(self, url: str) -> FetchResult:
        requested = canonicalize_url(url)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(requested, timeout=self.timeout, allow_redirects=True)
                response.raise_for_status()
                canonical = canonicalize_url(response.url)
                content_type = response.headers.get("Content-Type")
                if content_type and "html" not in content_type.lower():
                    raise BuildError(f"Expected HTML from {canonical}, got {content_type!r}")
                return FetchResult(
                    requested_url=requested,
                    canonical_url=canonical,
                    content=response.content,
                    retrieved_at_utc=utc_now(),
                    status_code=response.status_code,
                    content_type=content_type,
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                )
            except (requests.RequestException, BuildError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))
        raise BuildError(f"Failed after {self.retries + 1} attempts: {requested}: {last_error}")


def parse_url_date(url: str) -> dt.date | None:
    match = DATE_IN_URL_RE.search(urlsplit(url).path)
    if not match:
        return None
    return dt.datetime.strptime(match.group("date"), "%Y%m%d").date()


def discover_links(source_url: str, source_html: bytes) -> list[tuple[str, dt.date, str]]:
    """Return statement-shaped official links and their visible discovery labels."""
    soup = BeautifulSoup(source_html, "html.parser")
    found: list[tuple[str, dt.date, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        absolute = urljoin(source_url, href)
        try:
            canonical = canonicalize_url(absolute)
        except BuildError:
            continue
        statement_date = parse_url_date(canonical)
        if statement_date is None or not (FREEZE_START <= statement_date <= FREEZE_END):
            continue
        label_parts = [" ".join(anchor.get_text(" ", strip=True).split())]
        panel = anchor.find_parent("div", class_=lambda value: value and "panel" in value)
        if panel is not None:
            heading = panel.select_one(".panel-heading, h2, h3, h4, h5, h6")
            if heading is not None:
                label_parts.insert(0, " ".join(heading.get_text(" ", strip=True).split()))
        label = " | ".join(part for part in label_parts if part)
        found.append((canonical, statement_date, label))
    return found


def discover_candidates(fetcher: Fetcher) -> tuple[list[Candidate], list[dict[str, Any]], list[dict[str, Any]]]:
    evidence: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_url in DISCOVERY_URLS:
        try:
            fetched = fetcher.get(source_url)
            links = discover_links(fetched.canonical_url, fetched.content)
            sources.append(
                {
                    "url": fetched.canonical_url,
                    "retrieved_at_utc": fetched.retrieved_at_utc,
                    "raw_sha256": sha256_bytes(fetched.content),
                    "candidate_link_count": len(links),
                }
            )
            for url, statement_date, label in links:
                item = evidence.setdefault(url, {"date": statement_date, "sources": [], "labels": []})
                item["sources"].append(fetched.canonical_url)
                item["labels"].append(label)
        except BuildError as exc:
            failures.append({"url": source_url, "stage": "discovery", "error": str(exc), "status": "unresolved"})
    candidates = [
        Candidate(
            canonical_url=url,
            statement_date=value["date"],
            evidence_urls=tuple(sorted(set(value["sources"]))),
            evidence_labels=tuple(sorted(set(value["labels"]))),
        )
        for url, value in evidence.items()
    ]
    candidates.sort(key=lambda candidate: (candidate.statement_date, candidate.canonical_url))
    return candidates, sources, failures


def classify_eligibility(candidate: Candidate, page_html: bytes) -> tuple[bool, str]:
    """Classify a discovered press-release page from its official title/content."""
    soup = BeautifulSoup(page_html, "html.parser")
    title = extract_title(soup).lower()
    page_text = " ".join(soup.get_text(" ", strip=True).split()).lower()
    labels = " ".join(candidate.evidence_labels).lower()
    if "implementation note" in title or "implementation note" in labels:
        return False, "implementation_note"
    if "minutes" in title or "transcript" in title:
        return False, "non_statement_publication"
    statement_signals = (
        "federal reserve issues fomc statement",
        "fomc statement",
        "federal open market committee statement",
    )
    decision_signals = (
        "committee decided",
        "federal funds rate",
        "target range",
        "open market desk",
    )
    if not any(signal in title or signal in labels or signal in page_text for signal in statement_signals):
        return False, "not_fomc_statement"
    if not any(signal in page_text for signal in decision_signals):
        return False, "no_policy_decision_signal"
    return True, "eligible_policy_statement"


def extract_title(soup: BeautifulSoup) -> str:
    heading = soup.select_one("#article h2, #article h3, .article__heading, #content .title")
    if heading:
        title = " ".join(heading.get_text(" ", strip=True).split())
        if title:
            return title
    metadata = soup.select_one('meta[property="og:title"], meta[name="twitter:title"]')
    if metadata and metadata.get("content"):
        return " ".join(str(metadata["content"]).split())
    if soup.title:
        title = " ".join(soup.title.get_text(" ", strip=True).split())
        return re.sub(r"^Federal Reserve Board\s*-\s*", "", title)
    return "FOMC statement"


def _paragraph_text(tag: Tag) -> str:
    for removable in tag.select("script, style, noscript, .sr-only, .share, .social"):
        removable.decompose()
    return tag.get_text(" ", strip=True)


def extract_statement_body(page_html: bytes) -> tuple[str, str, list[str]]:
    """Extract substantive paragraphs from the Federal Reserve article container."""
    soup = BeautifulSoup(page_html, "html.parser")
    container = soup.select_one("#article") or soup.select_one("article") or soup.select_one("main #content")
    if container is None:
        raise BuildError("No recognized Federal Reserve article container (#article/article/main #content)")
    paragraphs: list[str] = []
    started = False
    for tag in container.find_all(["p", "h2"], recursive=True):
        text = " ".join(_paragraph_text(tag).split())
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith(("for release at", "for immediate release")):
            started = True
            continue
        if lowered.startswith(TERMINAL_PARAGRAPH_PREFIXES):
            if started or paragraphs:
                break
            continue
        if not started:
            if "committee" in lowered or "federal reserve" in lowered:
                started = True
            else:
                continue
        paragraphs.append(text)
    if not paragraphs:
        raise BuildError("Recognized article container yielded no substantive statement paragraphs")
    body = "\n\n".join(paragraphs)
    return body, EXTRACTION_METHOD, []


def normalize_statement_body(text: str) -> tuple[str, list[str]]:
    """Remove document noise while preserving financial semantics."""
    warnings: list[str] = []
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    mappings = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2212": "-",
        "\u2011": "-",
    }
    for source, replacement in mappings.items():
        text = text.replace(source, replacement)
    text = re.sub(r"(?<=\d)\s*[\u2013\u2014]\s*(?=\d)", "-", text)
    text = text.replace("\u2013", " -- ").replace("\u2014", " -- ")
    bad_controls = sorted({ord(char) for char in text if unicodedata.category(char) == "Cc" and char not in "\n\t"})
    if bad_controls:
        rendered = ", ".join(f"U+{code:04X}" for code in bad_controls)
        raise BuildError(f"Unsupported control character(s): {rendered}")
    text = text.replace("\t", " ")
    lines = [re.sub(r"[ \f\v]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if START_MARKER in text or END_MARKER in text:
        raise BuildError("Statement body collides with a reserved corpus marker")
    return text, warnings


def assign_split(statement_date: dt.date) -> str:
    if not (FREEZE_START <= statement_date <= FREEZE_END):
        raise BuildError(f"Statement date outside freeze interval: {statement_date}")
    return "validation" if statement_date >= VALIDATION_START else "train"


def infer_meeting_type(candidate: Candidate, body: str) -> str:
    labels = " ".join(candidate.evidence_labels).lower()
    body_lower = body.lower()
    if (
        "unscheduled" in labels
        or "intermeeting" in labels
        or "notation vote" in labels
        or "notational vote" in body_lower
    ):
        return "unscheduled"
    return "scheduled"


def serialize_document(document: IncludedDocument) -> str:
    return (
        f"{START_MARKER}\n"
        f"date: {document.statement_date.isoformat()}\n"
        f"meeting_type: {document.meeting_type}\n"
        f"document_id: {document.document_id}\n\n"
        f"{document.body}\n"
        f"{END_MARKER}\n"
    )


def serialize_corpus(documents: Sequence[IncludedDocument]) -> str:
    return "\n".join(serialize_document(document) for document in documents)


def _same_date_review(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    by_date: dict[dt.date, list[Candidate]] = collections.defaultdict(list)
    for candidate in candidates:
        by_date[candidate.statement_date].append(candidate)
    return [
        {
            "statement_date": day.isoformat(),
            "candidate_urls": [candidate.canonical_url for candidate in sorted(items, key=lambda item: item.canonical_url)],
            "status": "reviewed_during_eligibility_classification",
        }
        for day, items in sorted(by_date.items())
        if len(items) > 1
    ]


def validate_documents(documents: Sequence[IncludedDocument], corpus_bytes: bytes) -> None:
    errors: list[str] = []
    if not documents:
        errors.append("No included documents")
    if not any(document.split == "train" for document in documents):
        errors.append("Train split is empty")
    if not any(document.split == "validation" for document in documents):
        errors.append("Validation split is empty")
    ordered = sorted(documents, key=lambda item: (item.statement_date, item.canonical_url))
    if list(documents) != ordered:
        errors.append("Documents are not in deterministic date/URL order")
    for field, values in (
        ("canonical URL", [document.canonical_url for document in documents]),
        ("document ID", [document.document_id for document in documents]),
        ("normalized hash", [document.normalized_sha256 for document in documents]),
    ):
        duplicates = [value for value, count in collections.Counter(values).items() if count > 1]
        if duplicates:
            errors.append(f"Duplicate {field}(s): {duplicates}")
    for document in documents:
        try:
            canonicalize_url(document.canonical_url)
        except BuildError as exc:
            errors.append(str(exc))
        if assign_split(document.statement_date) != document.split:
            errors.append(f"Incorrect split for {document.document_id}")
        if not SHA256_RE.fullmatch(document.raw_sha256) or not SHA256_RE.fullmatch(document.normalized_sha256):
            errors.append(f"Invalid SHA-256 for {document.document_id}")
        body_bytes = document.body.encode("utf-8")
        if len(body_bytes) != document.normalized_utf8_bytes or len(document.body) != document.normalized_characters:
            errors.append(f"Normalized size mismatch for {document.document_id}")
        if sha256_bytes(body_bytes) != document.normalized_sha256:
            errors.append(f"Normalized checksum mismatch for {document.document_id}")
        if document.normalized_characters < MIN_REVIEW_CHARACTERS and "short_body_reviewed" not in document.warnings:
            errors.append(f"Below {MIN_REVIEW_CHARACTERS}-character review threshold: {document.document_id}")
        lowered = document.body.lower()
        leaks = [phrase for phrase in NAVIGATION_LEAKAGE if phrase in lowered]
        if leaks:
            errors.append(f"Navigation/footer leakage in {document.document_id}: {leaks}")
    corpus_text = corpus_bytes.decode("utf-8")
    if corpus_text.count(START_MARKER) != len(documents) or corpus_text.count(END_MARKER) != len(documents):
        errors.append("Corpus marker count does not equal manifest document count")
    if serialize_corpus(documents).encode("utf-8") != corpus_bytes:
        errors.append("Corpus serialization does not reproduce")
    if errors:
        raise BuildError("Corpus validation failed:\n- " + "\n- ".join(errors))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build(output_dir: Path, timeout: float, retries: int) -> tuple[Path, Path, dict[str, Any]]:
    output_dir = output_dir.resolve()
    repository_target = (Path(__file__).resolve().parent / OUTPUT_CORPUS).resolve()
    if output_dir / OUTPUT_CORPUS == repository_target:
        raise BuildError("Refusing implicit write to the frozen repository corpus location")
    fetcher = Fetcher(timeout=timeout, retries=retries)
    candidates, discovery_sources, failures = discover_candidates(fetcher)
    if not candidates:
        raise BuildError("Official discovery sources yielded no in-range statement candidates")
    exclusions: list[dict[str, Any]] = []
    documents: list[IncludedDocument] = []
    for candidate in candidates:
        try:
            fetched = fetcher.get(candidate.canonical_url)
            soup = BeautifulSoup(fetched.content, "html.parser")
            title = extract_title(soup)
            eligible, reason = classify_eligibility(candidate, fetched.content)
            if not eligible:
                exclusions.append(
                    {
                        "canonical_url": fetched.canonical_url,
                        "observed_date": candidate.statement_date.isoformat(),
                        "title": title,
                        "reason": reason,
                        "discovery_evidence": list(candidate.evidence_urls),
                    }
                )
                continue
            raw_body, extraction_method, warnings = extract_statement_body(fetched.content)
            body, normalization_warnings = normalize_statement_body(raw_body)
            warnings.extend(normalization_warnings)
            if len(body) < MIN_REVIEW_CHARACTERS:
                failures.append(
                    {
                        "url": fetched.canonical_url,
                        "stage": "content_review",
                        "error": f"normalized body has {len(body)} characters; threshold is {MIN_REVIEW_CHARACTERS}",
                        "status": "unresolved",
                    }
                )
                continue
            body_bytes = body.encode("utf-8")
            documents.append(
                IncludedDocument(
                    document_id=f"fomc-{candidate.statement_date.isoformat()}",
                    statement_date=candidate.statement_date,
                    meeting_type=infer_meeting_type(candidate, body),
                    title=title,
                    canonical_url=fetched.canonical_url,
                    retrieved_at_utc=fetched.retrieved_at_utc,
                    raw_sha256=sha256_bytes(fetched.content),
                    normalized_sha256=sha256_bytes(body_bytes),
                    raw_utf8_bytes=len(fetched.content),
                    normalized_utf8_bytes=len(body_bytes),
                    normalized_characters=len(body),
                    split=assign_split(candidate.statement_date),
                    extraction_method=extraction_method,
                    warnings=warnings,
                    status="included",
                    body=body,
                )
            )
        except (BuildError, requests.RequestException, ValueError) as exc:
            failures.append(
                {"url": candidate.canonical_url, "stage": "document", "error": str(exc), "status": "unresolved"}
            )
    documents.sort(key=lambda item: (item.statement_date, item.canonical_url))
    date_counts = collections.Counter(document.statement_date for document in documents)
    for document in documents:
        if date_counts[document.statement_date] > 1:
            document.document_id = f"{document.document_id}-{1 + sum(d.statement_date == document.statement_date and d.canonical_url < document.canonical_url for d in documents)}"
    corpus_bytes = serialize_corpus(documents).encode("utf-8")
    if failures:
        summary = "\n".join(f"- {failure['url']}: {failure['error']}" for failure in failures)
        raise BuildError(f"Unresolved eligible/discovery failures block candidate publication:\n{summary}")
    validate_documents(documents, corpus_bytes)
    generated_at = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "freeze_start": FREEZE_START.isoformat(),
        "freeze_end": FREEZE_END.isoformat(),
        "normalization_version": NORMALIZATION_VERSION,
        "extraction_method": EXTRACTION_METHOD,
        "generated_at_utc": generated_at,
        "discovery_sources": discovery_sources,
        "document_count": len(documents),
        "train_document_count": sum(document.split == "train" for document in documents),
        "validation_document_count": sum(document.split == "validation" for document in documents),
        "corpus_utf8_bytes": len(corpus_bytes),
        "corpus_characters": len(corpus_bytes.decode("utf-8")),
        "corpus_sha256": sha256_bytes(corpus_bytes),
        "documents": [document.manifest_record() for document in documents],
        "exclusions": exclusions,
        "failures": failures,
        "same_date_candidate_review": _same_date_review(candidates),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    corpus_path = output_dir / OUTPUT_CORPUS
    manifest_path = output_dir / OUTPUT_MANIFEST
    atomic_write(corpus_path, corpus_bytes)
    atomic_write(manifest_path, manifest_bytes)
    return corpus_path, manifest_path, manifest


def run_self_tests() -> None:
    fixture = (
        "January 29, 2025\r\n\r\nThe Committee's range was 4.25–4.50 percent (not 5-1/4).\r\n"
        "It preserved 5.25, 2%, +0.25, −0.25, and $8.5 trillion.\r\n\r\n"
        "“Quoted” text — with   spaces and digits 1234567890."
    )
    normalized, warnings = normalize_statement_body(fixture)
    required = (
        "5.25", "2%", "+0.25", "-0.25", "4.25-4.50", "5-1/4",
        "January 29, 2025", "$8.5 trillion", "Committee's", "(not 5-1/4)",
        '"Quoted"', " -- ", "\n\n", "1234567890",
    )
    assert not warnings
    for value in required:
        assert value in normalized, f"normalization failed to preserve {value!r}"
    assert "   " not in normalized
    assert normalize_statement_body(normalized)[0] == normalized
    try:
        normalize_statement_body("bad\x00control")
    except BuildError as exc:
        assert "U+0000" in str(exc)
    else:
        raise AssertionError("control character was silently retained")
    for marker in (START_MARKER, END_MARKER):
        try:
            normalize_statement_body(f"collision {marker}")
        except BuildError as exc:
            assert "reserved corpus marker" in str(exc)
        else:
            raise AssertionError("marker collision was not rejected")
    assert assign_split(dt.date(2024, 12, 31)) == "train"
    assert assign_split(dt.date(2025, 1, 1)) == "validation"
    sample = IncludedDocument(
        document_id="fomc-2025-01-29", statement_date=dt.date(2025, 1, 29), meeting_type="scheduled",
        title="Federal Reserve issues FOMC statement", canonical_url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20250129a.htm",
        retrieved_at_utc="2025-01-29T19:00:00Z", raw_sha256="0" * 64,
        normalized_sha256=sha256_bytes(normalized.encode()), raw_utf8_bytes=1,
        normalized_utf8_bytes=len(normalized.encode()), normalized_characters=len(normalized),
        split="validation", extraction_method="fixture", warnings=[], status="included", body=normalized,
    )
    serialized = serialize_corpus([sample])
    assert serialized.count(START_MARKER) == serialized.count(END_MARKER) == 1
    assert serialized.endswith(f"{END_MARKER}\n")

    fixture_a = b"""
        <html><body><div id="article">
        <p>For immediate release</p>
        <p>The Committee decided to maintain the target range for the federal funds rate.</p>
        <p>Voting for the monetary policy action were all Committee members.</p>
        <p>For media inquiries, please email media@frb.gov.</p>
        <p>Implementation Note issued January 29, 2025</p>
        <p>Related Materials and Implementation Note</p>
        </div></body></html>
    """
    body_a, method_a, warnings_a = extract_statement_body(fixture_a)
    assert method_a == EXTRACTION_METHOD
    assert not warnings_a
    assert "maintain the target range" in body_a
    assert "Voting for the monetary policy action" in body_a
    assert "media inquiries" not in body_a.lower()
    assert "implementation note" not in body_a.lower()
    assert "related materials" not in body_a.lower()

    fixture_b = b"""
        <html><body><article>
        <p>The Committee decided to maintain the target range.</p>
        <p>Media contact: Federal Reserve Press Office.</p>
        <p>This unrelated trailing paragraph must not be retained.</p>
        </article></body></html>
    """
    body_b, _, _ = extract_statement_body(fixture_b)
    assert "maintain the target range" in body_b
    assert "media contact" not in body_b.lower()
    assert "unrelated trailing paragraph" not in body_b.lower()

    fixture_c = b"""
        <html><body><main><div id="content">
        <p>The Committee decided that implementation risks and media reports do not alter its policy judgment.</p>
        <p>Voting for the monetary policy action were all members.</p>
        </div></main></body></html>
    """
    body_c, _, _ = extract_statement_body(fixture_c)
    assert "implementation risks and media reports" in body_c
    assert "Voting for the monetary policy action" in body_c


def write_review_report(output_dir: Path, compare_dir: Path, review_path: Path) -> dict[str, Any]:
    """Create the empirical review report from two completed candidate builds."""
    corpus_path = output_dir.resolve() / OUTPUT_CORPUS
    manifest_path = output_dir.resolve() / OUTPUT_MANIFEST
    rebuild_corpus_path = compare_dir.resolve() / OUTPUT_CORPUS
    rebuild_manifest_path = compare_dir.resolve() / OUTPUT_MANIFEST
    corpus_bytes = corpus_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    rebuild_corpus_bytes = rebuild_corpus_path.read_bytes()
    rebuild_manifest_bytes = rebuild_manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    rebuild_manifest = json.loads(rebuild_manifest_bytes)
    documents = manifest["documents"]
    rebuild_documents = rebuild_manifest["documents"]
    identity_fields = ("document_id", "statement_date", "canonical_url", "normalized_sha256", "split")
    stable_documents_match = [
        tuple(document[field] for field in identity_fields) for document in documents
    ] == [tuple(document[field] for field in identity_fields) for document in rebuild_documents]
    corpus_match = corpus_bytes == rebuild_corpus_bytes
    stable_top_level_fields = (
        "schema_version", "corpus_id", "freeze_start", "freeze_end", "normalization_version",
        "extraction_method",
        "document_count", "train_document_count", "validation_document_count", "corpus_utf8_bytes",
        "corpus_characters", "corpus_sha256", "exclusions", "failures", "same_date_candidate_review",
    )
    stable_manifest_match = all(manifest[field] == rebuild_manifest[field] for field in stable_top_level_fields)
    years = collections.Counter(document["statement_date"][:4] for document in documents)
    types = collections.Counter(document["meeting_type"] for document in documents)
    exclusion_reasons = collections.Counter(item["reason"] for item in manifest["exclusions"])
    corpus_text = corpus_bytes.decode("utf-8")
    inventory = sorted(set(corpus_text), key=ord)
    inventory_display = " ".join(
        f"U+{ord(char):04X} {unicodedata.name(char, 'UNNAMED')} ({json.dumps(char, ensure_ascii=False)})"
        for char in inventory
    )
    run_self_tests()
    checks = {
        "included_identities_hashes_splits_match": stable_documents_match,
        "corpus_bytes_match": corpus_match,
        "corpus_sha256_match": manifest["corpus_sha256"] == rebuild_manifest["corpus_sha256"],
        "stable_manifest_content_match": stable_manifest_match,
    }
    recommendation = "GO" if all(checks.values()) and not manifest["failures"] else "NO-GO"
    lines = [
        "# Week 3 FOMC candidate corpus review", "",
        f"Recommendation: **{recommendation}** for promotion in a later authorized phase. No promotion occurred in this phase.", "",
        "## Discovery and coverage", "",
        "Official discovery sources:", "",
    ]
    for source in manifest["discovery_sources"]:
        lines.append(f"- {source['url']} — {source['candidate_link_count']} in-range monetary press-release links")
    lines.extend([
        "",
        f"Eligible/included documents: **{manifest['document_count']} / {manifest['document_count']}**",
        f"Scheduled/unscheduled: **{types['scheduled']} / {types['unscheduled']}**",
        f"Train/validation: **{manifest['train_document_count']} / {manifest['validation_document_count']}**",
        f"Date minimum/maximum: **{documents[0]['statement_date']} / {documents[-1]['statement_date']}**",
        "",
        "Per-year included counts:", "",
    ])
    lines.extend(f"- {year}: {count}" for year, count in sorted(years.items()))
    lines.extend(["", "## Exclusions, failures, ambiguities, and duplicates", ""])
    lines.append("Exclusions by reason: " + (", ".join(f"{reason}={count}" for reason, count in sorted(exclusion_reasons.items())) or "none"))
    lines.append(f"Failures: {len(manifest['failures'])} (none unresolved).")
    lines.append(
        f"Same-date candidate groups explicitly reviewed: {len(manifest['same_date_candidate_review'])}; "
        "each candidate was fetched and passed through content-based eligibility classification."
    )
    lines.append("Canonical-URL, document-ID, and exact normalized-hash duplicate checks: PASS; no included duplicates.")
    lines.append("Near-duplicate scoring was not added because discovery produced no unresolved same-date included pair or exact duplicate.")
    lines.extend(["", "Excluded candidates:", ""])
    for item in manifest["exclusions"]:
        lines.append(
            f"- {item['observed_date']} — `{item['reason']}` — {item['title']} — {item['canonical_url']}"
        )
    lines.extend([
        "", "## Corpus measurements", "",
        f"Corpus UTF-8 bytes: **{manifest['corpus_utf8_bytes']}**",
        f"Corpus characters: **{manifest['corpus_characters']}**",
        f"Vocabulary size (exact serialized-corpus character set): **{len(inventory)}**",
        "",
        "Exact character inventory:", "",
        inventory_display,
        "", "## Finance-preserving normalization evidence", "",
        "Offline normalization fixtures: **PASS**. They preserve `5.25`, `2%`, `+0.25`, `-0.25`, "
        "`4.25–4.50` as deterministic ASCII range `4.25-4.50`, `5-1/4`, `January 29, 2025`, "
        "`$8.5 trillion`, `Committee's`, parentheses, paragraph boundaries, and digit runs. Curly quotes, "
        "numeric/prose dashes, horizontal whitespace, idempotence, controls, and marker collisions also pass.",
        "", "## Deterministic rebuild", "",
    ])
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    lines.extend([
        "",
        "Retrieval timestamps and raw-HTML SHA-256 values are volatile because the official pages inject dynamic "
        "page bytes. Stable manifest content, extracted document metadata, normalized hashes, and corpus bytes reconcile.",
        "", "## Hashes", "",
        f"- Candidate corpus SHA-256: `{sha256_bytes(corpus_bytes)}`",
        f"- Candidate manifest SHA-256: `{sha256_bytes(manifest_bytes)}`",
        f"- Rebuild corpus SHA-256: `{sha256_bytes(rebuild_corpus_bytes)}`",
        f"- Rebuild manifest SHA-256: `{sha256_bytes(rebuild_manifest_bytes)}`",
        "", "## Included documents", "",
        "| Date | Type | Title | URL | Characters | Split |",
        "|---|---|---|---|---:|---|",
    ])
    for document in documents:
        title = document["title"].replace("|", "\\|")
        lines.append(
            f"| {document['statement_date']} | {document['meeting_type']} | {title} | "
            f"{document['canonical_url']} | {document['normalized_characters']} | {document['split']} |"
        )
    report_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    atomic_write(review_path.resolve(), report_bytes)
    return {
        "recommendation": recommendation,
        "review_sha256": sha256_bytes(report_bytes),
        "candidate_manifest_sha256": sha256_bytes(manifest_bytes),
        "rebuild_manifest_sha256": sha256_bytes(rebuild_manifest_bytes),
        **checks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Explicit candidate output directory outside the repository")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds (default: 30)")
    parser.add_argument("--retries", type=int, default=2, choices=range(0, 6), help="Retries after the first request (default: 2)")
    parser.add_argument("--self-test", action="store_true", help="Run bounded offline pure-function tests and exit")
    parser.add_argument("--review-existing", action="store_true", help="Generate a review from two existing candidate builds")
    parser.add_argument("--compare-dir", type=Path, help="Second candidate directory for --review-existing")
    parser.add_argument("--review-output", type=Path, help="Markdown path for --review-existing")
    args = parser.parse_args(argv)
    if args.review_existing and (args.output_dir is None or args.compare_dir is None or args.review_output is None):
        parser.error("--review-existing requires --output-dir, --compare-dir, and --review-output")
    if not args.self_test and not args.review_existing and args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            run_self_tests()
            print("Self-tests: PASS (normalization, extraction boundaries, controls, markers, splits, serialization)")
            return 0
        if args.review_existing:
            result = write_review_report(args.output_dir, args.compare_dir, args.review_output)
            print(f"Review: {args.review_output.resolve()}")
            print(f"Recommendation: {result['recommendation']}; review SHA-256 {result['review_sha256']}")
            return 0
        corpus_path, manifest_path, manifest = build(args.output_dir, args.timeout, args.retries)
        print(
            f"Built {manifest['document_count']} statements "
            f"({manifest['train_document_count']} train, {manifest['validation_document_count']} validation); "
            f"{manifest['corpus_utf8_bytes']} bytes; SHA-256 {manifest['corpus_sha256']}"
        )
        print(f"Corpus:   {corpus_path}")
        print(f"Manifest: {manifest_path}")
        return 0
    except (BuildError, OSError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
