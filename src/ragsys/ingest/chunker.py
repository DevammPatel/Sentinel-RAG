"""Structure-aware chunking.

Regulatory circulars are highly structured (numbered clauses, "Annexure",
paragraph markers). We first split on those natural boundaries, then pack the
pieces into ~chunk_size windows with overlap so no single clause is orphaned.

Cuts always land on a sentence boundary where one exists, and on a word boundary
otherwise, so a passage shown to a user never starts or ends mid-word.
"""
from __future__ import annotations

import re

from ragsys.config import settings
from ragsys.models import Chunk, Document

# Boundaries: numbered clauses (1., 2.3, (a)), "Annexure" / "Chapter" headings.
_BOUNDARY = re.compile(
    r"(?m)^(?:\s*(?:\d+(?:\.\d+)*\.?|\([a-z0-9]+\)|Annexure|ANNEXURE|Chapter|CHAPTER)\s)"
)
_SENTENCE_END = re.compile(r"(?<=[.;:?!])\s+")


def _split_units(text: str) -> list[str]:
    idxs = [m.start() for m in _BOUNDARY.finditer(text)]
    if not idxs:
        # fall back to paragraph splitting
        return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    idxs = [0] + idxs if idxs[0] != 0 else idxs
    units: list[str] = []
    for i, start in enumerate(idxs):
        end = idxs[i + 1] if i + 1 < len(idxs) else len(text)
        piece = text[start:end].strip()
        if piece:
            units.append(piece)
    return units


def _split_long(unit: str, size: int) -> list[str]:
    """Break a unit longer than `size` into pieces of at most `size` characters,
    preferring sentence boundaries and never cutting inside a word."""
    pieces: list[str] = []
    buf = ""
    for sentence in _SENTENCE_END.split(unit):
        sentence = sentence.strip()
        while len(sentence) > size:  # one enormous sentence: cut at a space
            if buf:
                pieces.append(buf)
                buf = ""
            cut = sentence.rfind(" ", 0, size)
            if cut < size // 2:  # no usable space (e.g. a long URL): hard cut
                cut = size
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if len(buf) + len(sentence) + 1 <= size:
            buf = f"{buf} {sentence}" if buf else sentence
        else:
            pieces.append(buf)
            buf = sentence
    if buf:
        pieces.append(buf)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Up to `overlap - 1` trailing characters of `text` to carry into the next
    chunk, starting at a sentence boundary if there is one, else at a word."""
    if overlap <= 1:
        return ""
    window = text[-(overlap - 1):]
    if len(text) <= overlap - 1:
        return window
    sentence_starts = [m.end() for m in _SENTENCE_END.finditer(window)]
    if sentence_starts:
        return window[sentence_starts[0]:].strip()
    space = window.find(" ")
    return window[space + 1:].strip() if space != -1 else ""


def _pack(units: list[str], size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    for unit in units:
        pieces.extend(_split_long(unit, size) if len(unit) > size else [unit])

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if not buf:
            buf = piece
        elif len(buf) + len(piece) + 1 <= size:
            buf = f"{buf}\n{piece}"
        else:
            chunks.append(buf)
            tail = _tail(buf, overlap)
            buf = f"{tail}\n{piece}" if tail else piece
    if buf.strip():
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def chunk_document(
    doc: Document,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    pieces = _pack(_split_units(doc.text), size, overlap)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{i}",
            doc_id=doc.doc_id,
            title=doc.title,
            source=doc.source,
            url=doc.url,
            ordinal=i,
            text=piece,
            meta={"circular_no": doc.circular_no, "issued_on": doc.issued_on},
        )
        for i, piece in enumerate(pieces)
    ]
