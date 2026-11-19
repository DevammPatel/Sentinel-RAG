#!/usr/bin/env python
"""CLI for building the hybrid index.

Examples
--------
  python scripts/ingest_cli.py                    # ingest data/raw (the fetched corpus)
  python scripts/ingest_cli.py --path my/folder   # ingest your own PDFs / HTML / txt
  python scripts/ingest_cli.py --sample           # tiny offline sample corpus
  python scripts/ingest_cli.py --reset            # empty the index first
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest SEBI/RBI documents into the index.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sample", action="store_true", help="use the offline sample corpus")
    g.add_argument("--path", default="data/raw", help="file or directory to ingest")
    ap.add_argument("--reset", action="store_true", help="remove everything indexed so far")
    args = ap.parse_args()

    from ragsys.ingest import ingest_path
    from ragsys.ingest.sources import sample_corpus
    from ragsys.store import get_vector_store
    from ragsys.telemetry import init_tracing

    init_tracing()

    if args.sample:
        path = sample_corpus()[0].parent
    else:
        path = Path(args.path)
        if not path.exists():
            print(f"{path} does not exist. Run `python scripts/fetch_circulars.py` first, "
                  "or pass --sample.")
            return 1

    if args.reset:
        store = get_vector_store()
        store.delete_docs(sorted({c.doc_id for c in store.all_chunks()}))
        print("[ingest] index emptied")

    t0 = time.perf_counter()
    stats = ingest_path(path, verbose=True)

    print(f"\nIngestion complete in {time.perf_counter() - t0:.1f}s:")
    for k, v in stats.items():
        print(f"  {k:>14}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
