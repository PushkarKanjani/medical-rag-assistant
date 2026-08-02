"""
rebuild_index_bm25.py
─────────────────────
Zero-dependency (no torch/GPU) ingestion:
  1. Extract text from PDF using pypdf
  2. Store pages as plain JSON for BM25/keyword retrieval
  3. Update ingestion manifest

Run:
    .venv/Scripts/python scripts/rebuild_index_bm25.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

PDF_PATH     = pathlib.Path("data/The_Gale_Encyclopedia_of_Medicine_3rd_Edition_500.pdf")
PAGES_PATH   = pathlib.Path("data/processed/pages.json")
MANIFEST_PATH = pathlib.Path("ingestion_manifest.json")


def extract_pages(pdf_path: pathlib.Path) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("ERROR: pypdf not installed. Run: pip install pypdf"); sys.exit(1)

    reader = PdfReader(str(pdf_path))
    pages = []
    total = len(reader.pages)
    print(f"Extracting text from {total} pages …")
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        pages.append({"page_number": i + 1, "text": text or f"[Page {i+1}: image-only]"})
        if (i + 1) % 100 == 0:
            print(f"  … {i+1}/{total}")
    print(f"Done. {len(pages)} pages.")
    return pages


if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}"); sys.exit(1)

    t0 = time.perf_counter()
    pages = extract_pages(PDF_PATH)

    PAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGES_PATH.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(pages)} pages to {PAGES_PATH}")

    manifest = {
        "status": "success",
        "collection": "bm25_pages",
        "retrieval_mode": "bm25_json",
        "pages_path": str(PAGES_PATH),
        "processed_files": [str(PDF_PATH)],
        "total_pages": len(pages),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest updated: {MANIFEST_PATH}")
    print(f"Done in {time.perf_counter() - t0:.1f}s")
