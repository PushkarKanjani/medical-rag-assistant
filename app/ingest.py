"""
app/ingest.py
─────────────
Ingestion pipeline for Medical RAG Assistant.
Extracted and ported from Cell 2 of Medical_RAG_Core_Engine.ipynb.

Workflow:
1. Extract & clean text from PDF using PyMuPDF (fitz).
2. Clean noisy lines (headers, index markers, URLs, low alpha ratio).
3. Chunk into ~1500 char blocks with 200 char overlap, tracking page numbers.
4. Generate embeddings with sentence-transformers/all-MiniLM-L6-v2.
5. Persist vectors in ChromaDB (PersistentClient at ./data/indexes/chroma).
6. Build and save BM25 index to ./data/indexes/bm25_index.pkl.

Run via:
    python -m app.ingest
"""

from __future__ import annotations

import bisect
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

# ---- Paths & Config ----
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "pdf" / "The_Gale_Encyclopedia_of_Medicine_3rd_Edition.pdf"
INDEXES_DIR = BASE_DIR / "data" / "indexes"
CHROMA_PATH = INDEXES_DIR / "chroma"
BM25_PATH = INDEXES_DIR / "bm25_index.pkl"

COLLECTION_NAME = "medical_encyclopedia"
TARGET_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Cleaning Patterns ----
HEADER_RE = re.compile(r'(?:[A-Z]\s+){5,}[A-Z]', re.IGNORECASE)  # spaced-out header like "G A L E ..."
INDEX_MARKER_RE = re.compile(r'\b\d+\s*:\s*\d+\b')
URL_RE = re.compile(r'https?://|www\.', re.IGNORECASE)


def clean_medical_text(line: str) -> str:
    """
    Cleans a single line of text from the medical encyclopedia:
    - Strips whitespace.
    - Removes spaced-out Gale headers.
    - Filters short lines (< 50 chars), index markers, URLs, and organization headers.
    - Enforces >= 70% alphabetic character ratio.
    """
    line = line.strip()
    if not line:
        return ""
    # Remove spaced-out "GALE ENCYCLOPEDIA OF MEDICINE" header
    if HEADER_RE.search(line) and "GALE" in re.sub(r'\s+', '', line).upper()[:60]:
        return ""
    if len(line) < 50:
        return ""
    if len(INDEX_MARKER_RE.findall(line)) > 1:
        return ""
    if URL_RE.search(line):
        return ""
    if "ORGANIZATIONS" in line.upper():
        return ""
    alpha_count = sum(c.isalpha() for c in line)
    if alpha_count / max(len(line), 1) < 0.70:
        return ""
    return line


def chunk_text_page(full_text: str, offsets: list[tuple[int, int]], filename: str) -> list[dict[str, Any]]:
    """
    Chunks text by paragraph (\\n\\n), greedily packing to ~1500 chars with 200 char overlap,
    and maps each chunk offset back to its source page.
    """
    offset_starts = [o[0] for o in offsets]

    def find_page(char_offset: int) -> int:
        idx = bisect.bisect_right(offset_starts, char_offset) - 1
        idx = max(0, min(idx, len(offsets) - 1))
        return offsets[idx][1]

    raw_paragraphs = []
    pos = 0
    for part in full_text.split("\n\n"):
        raw_paragraphs.append((pos, part))
        pos += len(part) + 2

    chunks = []  # list of (start_offset, text)
    current = ""
    current_start = 0

    for start, para in raw_paragraphs:
        if not para.strip():
            continue
        if not current:
            current_start = start
        if len(current) + len(para) + 2 <= TARGET_CHUNK_SIZE or not current:
            current += para + "\n\n"
        else:
            chunks.append((current_start, current.strip()))
            overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
            current_start = start
            current = overlap_text + "\n\n" + para + "\n\n"

    if current.strip():
        chunks.append((current_start, current.strip()))

    chunk_records = []
    for idx, (start_off, text) in enumerate(chunks):
        if len(text) < 50:
            continue
        page = find_page(start_off)
        chunk_id = f"{filename}-p{page}-c{idx}"
        chunk_records.append({"id": chunk_id, "text": text, "page": page})

    return chunk_records


def run_ingestion(pdf_path: Path | str | None = None) -> None:
    """Executes the complete ingestion pipeline on the specified PDF."""
    target_pdf = Path(pdf_path) if pdf_path else PDF_PATH

    if not target_pdf.exists():
        # Check fallback to any available PDF in data/pdf
        pdf_dir = BASE_DIR / "data" / "pdf"
        available_pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        if available_pdfs:
            # Pick non-part PDF first or largest
            complete = [p for p in available_pdfs if "_part" not in p.stem.lower()]
            target_pdf = complete[0] if complete else max(available_pdfs, key=lambda p: p.stat().st_size)
        else:
            raise FileNotFoundError(f"PDF not found at {target_pdf}")

    filename = target_pdf.stem
    print("=" * 65)
    print("🩺 MEDICAL RAG ASSISTANT — INGESTION PIPELINE")
    print("=" * 65)
    print(f"Source PDF       : {target_pdf.name}")
    print(f"Full Path        : {target_pdf}")
    print(f"Embedding Device : {DEVICE}")
    print(f"Chroma Path      : {CHROMA_PATH}")
    print(f"BM25 Index Path  : {BM25_PATH}")
    print("=" * 65)

    start_time = time.time()

    # 1. Extract + clean per page
    print("\n📖 Extracting and cleaning text from PDF...")
    doc = fitz.open(str(target_pdf))
    full_text = ""
    offsets: list[tuple[int, int]] = []  # (start_char_offset_in_full_text, page_num)

    for page_num, page in enumerate(doc, start=1):
        raw_lines = page.get_text().split("\n")
        cleaned_lines = [clean_medical_text(l) for l in raw_lines]
        cleaned_lines = [l for l in cleaned_lines if l]
        page_clean = "\n".join(cleaned_lines)
        if page_clean.strip():
            offsets.append((len(full_text), page_num))
            full_text += page_clean + "\n\n"

    doc.close()
    print(f"✅ Extracted {len(full_text):,} clean characters across {len(offsets)} non-empty pages.")

    # 2. Chunking
    print("\n✂️ Chunking text into ~1500 char blocks with 200 char overlap...")
    chunk_records = chunk_text_page(full_text, offsets, filename)
    total_chunks = len(chunk_records)
    print(f"✅ Created {total_chunks} chunks.")

    if total_chunks == 0:
        print("⚠️ No valid chunks generated. Aborting ingestion.")
        return

    # 3. Dense Embeddings
    print(f"\n🧠 Loading embedding model '{EMBED_MODEL_NAME}' on {DEVICE}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    texts = [c["text"] for c in chunk_records]
    print(f"⚡ Encoding {total_chunks} chunks...")
    batch_size = 128 if DEVICE == "cuda" else 32
    embeddings = embed_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # 4. ChromaDB (Local Persistent Storage)
    print(f"\n💾 Initializing ChromaDB PersistentClient at {CHROMA_PATH}...")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Reset or get collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"📦 Upserting {total_chunks} vectors into collection '{COLLECTION_NAME}'...")
    batch_upsert = 500
    for i in range(0, total_chunks, batch_upsert):
        batch = chunk_records[i:i + batch_upsert]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings[i:i + batch_upsert].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"]} for c in batch],
        )
        print(f"  -> Added vectors {i + 1} to {min(i + batch_upsert, total_chunks)} / {total_chunks}")

    print(f"✅ Chroma collection ready with {collection.count()} vectors.")

    # 5. BM25 Index
    print("\n🔍 Building BM25 index...")
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    tokenized_corpus = [re.findall(r'\w+', c["text"].lower()) for c in chunk_records]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_payload = {
        "bm25": bm25,
        "chunks": chunk_records,
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved BM25 index to {BM25_PATH}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("🎉 INGESTION COMPLETE")
    print(f"Total Chunks Indexed : {total_chunks}")
    print(f"Total Time Taken     : {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print("=" * 65)


if __name__ == "__main__":
    run_ingestion()
