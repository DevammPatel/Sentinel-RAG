#!/usr/bin/env python
"""CLI for ingesting documents into the hybrid index.

Examples
--------
  python scripts/ingest_cli.py --sample
  python scripts/ingest_cli.py --path data/pdfs
  python scripts/ingest_cli.py --urls https://host/circular1.pdf https://host/circular2.pdf
"""
from __future__ import annotations

import argparse
import sys

from ragsys.ingest import ingest_path
from ragsys.ingest.sources import download_urls, sample_corpus
from ragsys.telemetry import init_tracing


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest SEBI/RBI circulars.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="use the offline sample corpus")
    g.add_argument("--path", help="file or directory of PDFs/txt to ingest")
    g.add_argument("--urls", nargs="+", help="explicit PDF URLs to download then ingest")
    args = ap.parse_args()

    init_tracing()

    if args.sample:
        paths = sample_corpus()
        stats = ingest_path(paths[0].parent)
    elif args.urls:
        saved = download_urls(args.urls)
        if not saved:
            print("No files downloaded.")
            return 1
        stats = ingest_path(saved[0].parent)
    else:
        stats = ingest_path(args.path)

    print("\nIngestion complete:")
    for k, v in stats.items():
        print(f"  {k:>14}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
