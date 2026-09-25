"""Corpus handling for the Week 3 v2 capstone (DESIGN.md v0.4, §2 and §3).

Single source of truth for everything between the static corpus snapshot and
model input: SHA-256 checks, §3 normalization, the body parser, manifest and
split loading, document records, serialization as BOS + genre token + body,
the character vocabulary, and training-window sampling. Every check fails
closed with ValueError.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol, Sequence

if TYPE_CHECKING:
    import torch

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "fomc_training_corpus.txt"
MANIFEST = HERE / "fomc_training_corpus_manifest.json"
SPLIT_MANIFEST = HERE / "split_manifest.json"

# Frozen snapshot (§2, §9): any mismatch stops the pipeline.
CORPUS_SHA256 = (
    "3318b478ddcfaa0dd6292212d7f5b69d0578bd56350099306501ac40edb9ce93")
MANIFEST_SHA256 = (
    "63c967167fed33f9755da65b8da0538471b8436d2082be02522c02d842cfa168")
SPLIT_MANIFEST_SHA256 = (
    "f15633e9a53d311dd866e196c017a0f1bc87fd5ea020813fe1fce829e8d82bd7")

# §3 normalization and corpus format
FIXED_MAP = str.maketrans(
    {"ø": "o", "Ø": "O", "®": None, "™": None, "©": None})
DOC_RE = re.compile(
    r"<\|fomc_(statement|minutes)\|>\n(.*?)<\|end_fomc_\1\|>", re.S)
HEADER_RE = re.compile(r"^(date|document_id|meeting_type): ")
GENRES = {"statements": "statement", "minutes": "minutes"}

# §3 serialization: reserved special tokens, never produced from body text.
BOS = "<BOS>"
GENRE_TOKENS = {"statement": "<stmt>", "minutes": "<min>"}
SPECIAL_TOKENS = (BOS, GENRE_TOKENS["statement"], GENRE_TOKENS["minutes"])
PREFIX_LEN = 2  # BOS + genre token

SPLITS = ("T", "N", "F", "excluded")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sha256(data: bytes, expected: str, label: str) -> None:
    actual = sha256_bytes(data)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 {actual} does not match expected {expected}")


def normalize(text: str) -> str:
    """§3: (1) NFD, (2) drop Mn, (3) drop Cf, (4) fixed map, (5) NFC."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text if unicodedata.category(c) not in ("Mn", "Cf"))
    text = text.translate(FIXED_MAP)
    return unicodedata.normalize("NFC", text)


def parse_documents(corpus_text: str) -> list[dict[str, Any]]:
    """§3 body parser: strip special tokens and the metadata header.

    The body is everything after the header's blank line, up to the newline
    before the end token.
    """
    docs = []
    for m in DOC_RE.finditer(corpus_text):
        genre, lines = m.group(1), m.group(2).split("\n")
        header: dict[str, str] = {}
        i = 0
        while i < len(lines) and HEADER_RE.match(lines[i]):
            key, value = lines[i].split(": ", 1)
            header[key] = value
            i += 1
        if (i >= len(lines) or lines[i] != ""
                or "document_id" not in header or "date" not in header):
            raise ValueError(f"malformed header near {header}")
        doc_id = header["document_id"]
        # Format: header, one blank line, body, one newline before end token.
        if len(lines) < i + 3 or lines[-1] != "":
            raise ValueError(
                f"missing newline before end token in {doc_id}")
        body_lines = lines[i + 1:-1]
        if body_lines[0] == "" or body_lines[-1] == "":
            raise ValueError(
                f"unexpected leading or trailing blank line in body of "
                f"{doc_id}")
        body = "\n".join(body_lines)
        if "<|" in body or any(HEADER_RE.match(ln) for ln in body_lines):
            raise ValueError(
                f"body of {doc_id} is empty or contains metadata")
        docs.append({"document_id": doc_id, "date": header["date"],
                     "genre": genre, "body": body})
    if DOC_RE.sub("", corpus_text).strip():
        raise ValueError("non-whitespace text outside document delimiters")
    return docs


def load_corpus(corpus_path: Path, manifest_path: Path
                ) -> tuple[str, dict[str, Any], bytes]:
    """Read the corpus and its manifest; the corpus hash must match.

    Returns (corpus text, manifest, manifest bytes).
    """
    raw = corpus_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if sha256_bytes(raw) != manifest["corpus_sha256"]:
        raise ValueError("corpus SHA-256 does not match manifest")
    return raw.decode("utf-8"), manifest, manifest_bytes


def load_split_manifest(path: Path = SPLIT_MANIFEST,
                        expected_sha256: str | None = SPLIT_MANIFEST_SHA256
                        ) -> dict[str, Any]:
    data = path.read_bytes()
    if expected_sha256 is not None:
        verify_sha256(data, expected_sha256, path.name)
    return json.loads(data)


@dataclasses.dataclass(frozen=True)
class Document:
    id: str
    genre: str  # "statement" or "minutes"
    meeting: str  # meeting_id (ISO date)
    split: str  # "T", "N", "F" or "excluded"
    body: str  # normalized (§3)


def load_documents(corpus_path: Path = CORPUS,
                   manifest_path: Path = MANIFEST,
                   split_path: Path = SPLIT_MANIFEST,
                   check_frozen: bool = True) -> list[Document]:
    """All documents, normalized and joined with the split manifest.

    With check_frozen, the corpus, manifest and split manifest must match
    the hardcoded snapshot hashes (§2, §9). Sorted by (meeting, genre).
    """
    text, manifest, manifest_bytes = load_corpus(corpus_path, manifest_path)
    split = load_split_manifest(
        split_path, SPLIT_MANIFEST_SHA256 if check_frozen else None)
    if check_frozen:
        verify_sha256(text.encode("utf-8"), CORPUS_SHA256, corpus_path.name)
        verify_sha256(manifest_bytes, MANIFEST_SHA256, manifest_path.name)
    if split["source"]["corpus_sha256"] != manifest["corpus_sha256"] or (
            split["source"]["manifest_sha256"]
            != sha256_bytes(manifest_bytes)):
        raise ValueError("split manifest was built from a different snapshot")

    bodies = {d["document_id"]: d for d in parse_documents(normalize(text))}
    docs = []
    for meeting in split["meetings"]:
        for entry in meeting["documents"]:
            parsed = bodies.pop(entry["document_id"], None)
            if parsed is None or parsed["genre"] != entry["genre"]:
                raise ValueError(
                    f"split document {entry['document_id']} not in corpus")
            if len(parsed["body"]) != entry["body_code_points"]:
                raise ValueError(
                    f"body length mismatch for {entry['document_id']}")
            docs.append(Document(
                id=entry["document_id"], genre=entry["genre"],
                meeting=meeting["meeting_id"], split=meeting["split"],
                body=parsed["body"]))
    if bodies:
        raise ValueError(f"corpus documents missing from split: {bodies}")
    return sorted(docs, key=lambda d: (d.meeting, d.genre))


def check_characters(docs: Iterable[Document]) -> dict[str, dict[str, int]]:
    """§3: characters in N and F bodies that are absent from T (fail closed
    if the result is non-empty)."""
    docs = list(docs)
    t_chars = {c for d in docs if d.split == "T" for c in d.body}
    offending: dict[str, dict[str, int]] = {}
    for d in docs:
        if d.split in ("N", "F"):
            for c in d.body:
                if c not in t_chars:
                    key = f"U+{ord(c):04X}"
                    counts = offending.setdefault(d.split, {})
                    counts[key] = counts.get(key, 0) + 1
    return offending


def _require_t(docs: Iterable[Document], what: str) -> list[Document]:
    """Fail closed unless every document is in split T (§2, §5)."""
    docs = list(docs)
    leaked = sorted({d.split for d in docs if d.split != "T"})
    if leaked:
        raise ValueError(f"{what} is fit on T documents only; "
                         f"got splits {leaked}")
    return docs


# Guards private constructors: tokenizers are built from T documents via
# from_documents(), or from toy input via the internal _from_* helpers.
_PRIVATE = object()


class Tokenizer(Protocol):
    token_to_id: dict[str, int]

    def encode(self, body: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


def serialize(tokenizer: Tokenizer, genre: str, body: str) -> list[int]:
    """§3: <BOS> + genre token + body. The body is encoded on its own, so
    token boundaries never cross the genre token (§4)."""
    return [tokenizer.token_to_id[BOS],
            tokenizer.token_to_id[GENRE_TOKENS[genre]],
            *tokenizer.encode(body)]


class CharVocab:
    """Characters of normalized T bodies plus the special tokens (§5).

    IDs: special tokens first (0..2), then characters in code-point order.
    Encoding fails closed on any character outside the vocabulary. Build
    it with from_documents(); _from_chars() is for tests only.
    """

    def __init__(self, chars: Iterable[str], *, _token: object = None):
        if _token is not _PRIVATE:
            raise TypeError("use CharVocab.from_documents()")
        chars = sorted(set(chars))
        if any(len(c) != 1 for c in chars):
            raise ValueError("vocabulary entries must be single characters")
        self.tokens = list(SPECIAL_TOKENS) + chars
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self._char_to_id = {c: self.token_to_id[c] for c in chars}

    @classmethod
    def from_documents(cls, docs: Iterable[Document]) -> CharVocab:
        docs = _require_t(docs, "the character vocabulary")
        return cls._from_chars(c for d in docs for c in d.body)

    @classmethod
    def _from_chars(cls, chars: Iterable[str]) -> CharVocab:
        """Internal: no provenance check. Tests and toy input only."""
        return cls(chars, _token=_PRIVATE)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, body: str) -> list[int]:
        try:
            return [self._char_to_id[c] for c in body]
        except KeyError:
            unknown = sorted({f"U+{ord(c):04X}" for c in body
                              if c not in self._char_to_id})
            raise ValueError(
                f"characters outside the vocabulary: {unknown}") from None

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.tokens[i] for i in ids)


class WindowSampler:
    """Training windows that each lie inside a single T document.

    A document is drawn with probability proportional to its serialized
    length in tokens, then a start position uniformly among the windows of
    block_size + 1 tokens that fit inside it. x is the first block_size
    tokens of the window, y the last block_size.
    """

    def __init__(self, sequences: Sequence[Sequence[int]], block_size: int):
        import torch  # lazy: make_split.py uses this module without torch

        if not sequences:
            raise ValueError("no training documents")
        short = [i for i, s in enumerate(sequences)
                 if len(s) < block_size + 1]
        if short:
            raise ValueError(
                f"{len(short)} documents shorter than block_size + 1 = "
                f"{block_size + 1} tokens (indices {short[:10]})")
        self.block_size = block_size
        lengths = torch.tensor([len(s) for s in sequences])
        self.data = torch.cat(
            [torch.as_tensor(s, dtype=torch.long) for s in sequences])
        self.offsets = torch.cumsum(lengths, 0) - lengths
        self.n_starts = lengths - block_size
        self.weights = lengths.double()

    @classmethod
    def from_documents(cls, docs: Iterable[Document], tokenizer: Tokenizer,
                       block_size: int) -> WindowSampler:
        docs = _require_t(docs, "the training sampler")
        return cls([serialize(tokenizer, d.genre, d.body) for d in docs],
                   block_size)

    def sample(self, batch_size: int,
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x, y, doc_index), each of leading dimension batch_size."""
        import torch

        doc = torch.multinomial(self.weights, batch_size, replacement=True,
                                generator=generator)
        u = torch.rand(batch_size, generator=generator, dtype=torch.float64)
        start = (u * self.n_starts[doc]).long()
        idx = (self.offsets[doc] + start)[:, None] + torch.arange(
            self.block_size + 1)
        window = self.data[idx]
        return window[:, :-1], window[:, 1:], doc
