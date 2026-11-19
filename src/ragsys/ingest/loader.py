"""Turn raw files (PDF / HTML / txt) into `Document` objects with clean text + metadata.

If the folder being ingested has a `manifest.json` (written by
`scripts/fetch_circulars.py`), its official title, regulator and URL override
what we would otherwise guess from the text. That is what lets every citation
link back to the regulator's own page.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ragsys.models import Document

SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".txt", ".md"}

# Reference numbers, as they appear at the top of the document:
#   SEBI (2026 format):  HO/19/34/11(6)2026-AFD-POD1/I/25244/2026
#   SEBI (older format): SEBI/HO/MIRSD/DOP/P/CIR/2023/12
#   RBI:                 RBI/2023-24/45   or   RBI/DOR/2025-26/155
_CIRCULAR_NO = [
    re.compile(r"\bHO/\S{3,50}?/ ?I/\d{3,6}/\d{4}\b"),
    re.compile(r"\bSEBI/HO/[A-Za-z0-9/\-_()]+/\d{4}/\d+\b"),
    re.compile(r"\bRBI/(?:[A-Za-z]+/)?\d{4}-\d{2,4}/\d+\b"),
]
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November"
    r"|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?"
)
_DATE = re.compile(
    rf"\b({_MONTH}\s+\d{{1,2}},?\s*\d{{4}}"        # November 05, 2026
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{4}}"  # 5 November 2026
    r"|\d{1,2}\.\d{1,2}\.\d{4})\b",                 # 24.08.2026
    re.IGNORECASE,
)
_PAGE_FURNITURE = re.compile(r"(?im)^.*\bpage\s+\d+\s+of\s+\d+\s*$")
# Site chrome that RBI pages put around the document text.
_SITE_CHROME = re.compile(r"(?im)^\s*(\(\s*\d+\s*kb\s*\)|previous versions)\s*$")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_BLOCK_TAGS = ["p", "div", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]


def _is_english(line: str) -> bool:
    """False for Hindi lines. SEBI issues some circulars bilingually and PDF
    extraction turns the Hindi half into mojibake, which only adds noise to an
    English index."""
    if _DEVANAGARI.search(line):
        return False
    letters = [ch for ch in line if ch.isalpha()]
    return not letters or sum(ch.isascii() for ch in letters) / len(letters) >= 0.7


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\xa0", " ").replace("​", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PAGE_FURNITURE.sub("", text)
    text = _SITE_CHROME.sub("", text)
    text = "\n".join(ln for ln in text.split("\n") if _is_english(ln))
    # collapse hyphenation at line breaks: "regula-\ntion" -> "regulation"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # normalise whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_source(name: str, text: str) -> str:
    head = f"{name}\n{text[:800]}".lower()
    if "reserve bank of india" in head or "rbi/" in head or name.lower().startswith("rbi"):
        return "RBI"
    if "securities and exchange board" in head or "sebi" in head:
        return "SEBI"
    return "UNKNOWN"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-") or "doc"


def _extract_meta(text: str) -> dict:
    """Reference number and date, read from the top of the document only: the
    body routinely cites *other* circulars, and those must not be picked up."""
    meta: dict = {}
    head = text[:700]
    for pattern in _CIRCULAR_NO:
        if m := pattern.search(head):
            meta["circular_no"] = re.sub(r"\s+", "", m.group(0))
            break
    if d := _DATE.search(head):
        meta["issued_on_raw"] = re.sub(r"\s+", " ", d.group(1)).strip()
    return meta


def _build(name: str, text: str, url: str | None, info: dict | None = None) -> Document:
    """`info` is this file's manifest entry, if there is one."""
    info = info or {}
    text = _clean(text)
    meta = _extract_meta(text)
    source = info.get("regulator") or _guess_source(name, text)
    # first non-empty line makes a decent title fallback
    title = info.get("title") or next(
        (ln.strip() for ln in text.splitlines() if ln.strip()), name
    )
    return Document(
        doc_id=info.get("id") or _slug(name),
        title=title[:220],
        source=source,
        url=url or info.get("url"),
        circular_no=meta.get("circular_no"),
        issued_on=meta.get("issued_on_raw"),
        text=text,
        meta={"filename": name, **meta},
    )


def html_to_text(html: str) -> str:
    """Extract readable text from a regulator HTML page.

    RBI pages wrap the document in the largest `table.tablebg`; anything else
    falls back to <body>. Block-level tags become line breaks so numbered
    clauses stay on their own lines (the chunker splits on those).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    # NB: never drop <form>: ASP.NET pages (RBI) wrap the whole body in one.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    candidates = soup.select("table.tablebg")
    root = max(candidates, key=lambda el: len(el.get_text())) if candidates else (soup.body or soup)

    # In HTML a newline inside a paragraph is just whitespace. Collapse it first,
    # so the only line breaks left are the ones we add for <br> and block tags.
    for node in root.find_all(string=True):
        node.replace_with(re.sub(r"\s+", " ", str(node)))
    for br in root.find_all("br"):
        br.replace_with("\n")
    for cell in root.find_all(["td", "th"]):
        cell.append(" ")
    for tag in root.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.append("\n")
    return root.get_text("")


_NEW_ITEM = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s|\(?[a-z0-9]{1,4}\)\s|[-•*▪➢]\s|Annexure\b|ANNEXURE\b|Chapter\b|CHAPTER\b"
    r"|Sub(?:ject)?\s*:|To,|Sir|Madam|Dear\b|Yours\b)"
)


def unwrap_lines(text: str) -> str:
    """Rejoin lines that a PDF wrapped in the middle of a sentence.

    PDF extraction returns one line per *visual* line, so a clause arrives
    broken at the column width. A line is treated as wrapped, and glued to the
    next one, when it is about as long as the document's full lines, does not
    end a sentence, and the next line does not open a new clause, bullet or
    heading. Short lines (addressee lists, headings, table cells) are left alone.

    "Long" is relative to the document: a two-column bilingual circular wraps at
    ~40 characters, a normal one at ~90.
    """
    lines = [ln.strip() for ln in text.split("\n")]
    lengths = sorted(len(ln) for ln in lines if ln)
    full_width = lengths[int(len(lengths) * 0.75)] if lengths else 0
    min_wrapped = max(20, int(full_width * 0.6))
    out: list[str] = []
    for line in lines:
        prev = out[-1] if out else ""
        wrapped = (
            prev
            and line
            and len(prev) >= min_wrapped
            and not prev.endswith((".", ":", ";", "?", "”", '"'))
            and not _NEW_ITEM.match(line)
        )
        if wrapped:
            out[-1] = f"{prev} {line}"
        else:
            out.append(line)
    return "\n".join(out)


def load_pdf(path: str | Path, url: str | None = None, info: dict | None = None) -> Document:
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    pages = [_PAGE_FURNITURE.sub("", page.extract_text() or "") for page in reader.pages]
    text = "\n".join(ln for ln in "\n".join(pages).split("\n") if _is_english(ln))
    return _build(path.name, unwrap_lines(text), url, info)


def load_html(path: str | Path, url: str | None = None, info: dict | None = None) -> Document:
    path = Path(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    return _build(path.name, html_to_text(html), url, info)


def load_text_file(path: str | Path, url: str | None = None, info: dict | None = None) -> Document:
    path = Path(path)
    return _build(path.name, path.read_text(encoding="utf-8", errors="ignore"), url, info)


def load_manifest(directory: str | Path) -> dict:
    """filename -> {id, regulator, title, url} for a folder, or {} if absent."""
    path = Path(directory) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_any(path: str | Path, url: str | None = None, info: dict | None = None) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, url, info)
    if suffix in {".html", ".htm"}:
        return load_html(path, url, info)
    return load_text_file(path, url, info)
