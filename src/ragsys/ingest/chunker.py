"""Structure-aware chunking.

Regulatory circulars are highly structured (numbered clauses, "Annexure",
paragraph markers). We first split on those natural boundaries, then pack the
pieces into ~chunk_size windows with overlap so no single clause is orphaned.
"""
from __future__ import annotations

import re

from ragsys.config import settings
from ragsys.models import Chunk, Document

# Boundaries: numbered clauses (1., 2.3, (a)), ALL-CAPS headings, "Annexure".
_BOUNDARY = re.compile(
    r"(?m)^(?:\s*(?:\d+(?:\.\d+)*\.?|\([a-z0-9]+\)|Annexure|ANNEXURE|Chapter|CHAPTER)\s)"
)


def _split_units(text: str) -> list[str]:
    idxs = [m.start() for m in _BOUNDARY.finditer(text)]
    if not idxs:
        # fall back to paragraph splitting
        return [p for p in re.split(r"\n{2,}", text) if p.strip()]
    idxs = [0] + idxs if idxs[0] != 0 else idxs
    units: list[str] = []
    for i, start in enumerate(idxs):
        end = idxs[i + 1] if i + 1 < len(idxs) else len(text)
        piece = text[start:end].strip()
        if piece:
            units.append(piece)
    return units


def _pack(units: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for unit in units:
        if len(unit) > size:
            # a single oversized unit -> hard window it
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for i in range(0, len(unit), size - overlap):
                chunks.append(unit[i : i + size].strip())
            continue
        if len(buf) + len(unit) + 1 <= size:
            buf = f"{buf}\n{unit}" if buf else unit
        else:
            if buf:
                chunks.append(buf.strip())
            # carry an overlap tail for continuity
            tail = buf[-overlap:] if overlap and buf else ""
            buf = f"{tail}\n{unit}".strip() if tail else unit
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def chunk_document(
    doc: Document,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    units = _split_units(doc.text)
    pieces = _pack(units, size, overlap)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        chunks.append(
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
        )
    return chunks
