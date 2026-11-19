#!/usr/bin/env python
"""Download the official SEBI / RBI documents listed in data/sources.json.

Each manifest entry points at the regulator's own public page for a circular or
direction.

  * SEBI pages embed a PDF, so the PDF is resolved and saved as data/raw/<id>.pdf.
  * RBI pages carry the full text as HTML, so the page itself is saved as
    data/raw/<id>.html (RBI's separate PDF host is not reachable from every
    network, and the HTML is the same document).

data/raw/manifest.json records the official title and URL of every file so the
ingest step can attach them to each chunk, which is what makes the citations in
the UI link back to the regulator.

It is deliberately slow and polite: one request at a time, a pause between
requests, and files already on disk are never fetched again.

  python scripts/fetch_circulars.py                # fetch everything missing
  python scripts/fetch_circulars.py --limit 5      # just the first five
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
_SEBI_PDF = re.compile(r"(https://www\.sebi\.gov\.in/sebi_data/attachdocs/[^'\"&\s]+\.pdf)", re.I)


def resolve_pdf_url(page_url: str, html: str) -> str | None:
    """Find the PDF a SEBI circular page embeds."""
    m = _SEBI_PDF.search(html)
    return urljoin(page_url, m.group(1)) if m else None


def fetch(sources: list[dict], dest: Path, delay: float) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest: dict = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    ok = skipped = failed = 0

    with httpx.Client(
        timeout=httpx.Timeout(45, connect=15), follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        for src in sources:
            as_html = src["regulator"] == "RBI"
            filename = f"{src['id']}.{'html' if as_html else 'pdf'}"
            out = dest / filename
            if out.exists() and filename in manifest:
                skipped += 1
                continue
            try:
                page = client.get(src["url"])
                page.raise_for_status()
                entry = {
                    "id": src["id"],
                    "regulator": src["regulator"],
                    "title": src["title"],
                    "url": src["url"],
                }
                if as_html:
                    if len(page.text) < 5000:
                        raise RuntimeError("page is too short to be the document")
                    out.write_text(page.text, encoding="utf-8")
                    size = len(page.content)
                else:
                    pdf_url = resolve_pdf_url(src["url"], page.text)
                    if not pdf_url:
                        raise RuntimeError("no PDF link found on the page")
                    time.sleep(delay)
                    pdf = client.get(pdf_url)
                    pdf.raise_for_status()
                    if not pdf.content.startswith(b"%PDF"):
                        raise RuntimeError("response is not a PDF")
                    out.write_bytes(pdf.content)
                    entry["pdf_url"] = pdf_url
                    size = len(pdf.content)
                manifest[filename] = entry
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                ok += 1
                print(f"[fetch] ok   {src['id']:<12} {size / 1024:7.0f} KB  {src['title'][:70]}",
                      flush=True)
            except Exception as exc:  # keep going; report at the end
                failed += 1
                print(f"[fetch] FAIL {src['id']:<12} {type(exc).__name__}: {exc}", flush=True)
            time.sleep(delay)
    return {"downloaded": ok, "already_present": skipped, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch official SEBI/RBI documents.")
    ap.add_argument("--sources", default="data/sources.json")
    ap.add_argument("--dest", default="data/raw")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    if args.limit:
        sources = sources[: args.limit]
    stats = fetch(sources, Path(args.dest), args.delay)
    print(f"\n[fetch] {stats}")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
