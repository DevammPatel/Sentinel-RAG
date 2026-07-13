"""Turn raw files (PDF / txt) into `Document` objects with cleaned text + metadata."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ragsys.models import Document

# SEBI circular numbers look like:  SEBI/HO/MIRSD/DOP/P/CIR/2023/12
# RBI  circular numbers look like:  RBI/2023-24/45  DOR.STR.REC.20/...
_SEBI_CIRC = re.compile(r"SEBI/[A-Z0-9/\-]+CIR/\d{4}/\d+", re.IGNORECASE)
_RBI_CIRC = re.compile(r"RBI/\d{4}-\d{2,4}/\d+", re.IGNORECASE)
_DATE = re.compile(
    r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4})",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    # collapse hyphenation at line breaks: "regula-\ntion" -> "regulation"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # normalise whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_source(name: str, text: str) -> str:
    head = f"{name}\n{text[:600]}".lower()
    if "reserve bank of india" in head or "rbi/" in head:
        return "RBI"
    if "securities and exchange board" in head or "sebi/" in head:
        return "SEBI"
    return "UNKNOWN"


def _doc_id(source: str, name: str, text: str) -> str:
    h = hashlib.sha1(f"{name}:{text[:2000]}".encode()).hexdigest()[:12]
    return f"{source.lower()}-{h}"


def _extract_meta(text: str) -> dict:
    meta: dict = {}
    if m := (_SEBI_CIRC.search(text) or _RBI_CIRC.search(text)):
        meta["circular_no"] = m.group(0)
    if d := _DATE.search(text[:1500]):
        meta["issued_on_raw"] = d.group(1)
    return meta


def _build(name: str, text: str, url: str | None) -> Document:
    text = _clean(text)
    source = _guess_source(name, text)
    meta = _extract_meta(text)
    # first non-empty line makes a decent title fallback
    title = next((ln.strip() for ln in text.splitlines() if ln.strip()), name)[:180]
    return Document(
        doc_id=_doc_id(source, name, text),
        title=title,
        source=source,
        url=url,
        circular_no=meta.get("circular_no"),
        issued_on=meta.get("issued_on_raw"),
        text=text,
        meta={"filename": name, **meta},
    )


def load_pdf(path: str | Path, url: str | None = None) -> Document:
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _build(path.name, "\n\n".join(pages), url)


def load_text_file(path: str | Path, url: str | None = None) -> Document:
    path = Path(path)
    return _build(path.name, path.read_text(encoding="utf-8", errors="ignore"), url)


def load_any(path: str | Path, url: str | None = None) -> Document:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path, url)
    return load_text_file(path, url)
