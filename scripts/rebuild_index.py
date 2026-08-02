"""
rebuild_index.py
────────────────
One-shot script: extract text from the Gale Encyclopedia PDF,
embed every page with sentence-transformers (all-MiniLM-L6-v2, 384-dim),
and persist a fresh Qdrant local collection at ./qdrant_db_text/
with full text payloads so the RAG pipeline returns real answers.

Run:
    .venv/Scripts/python scripts/rebuild_index.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

PDF_PATH = pathlib.Path("data/The_Gale_Encyclopedia_of_Medicine_3rd_Edition_500.pdf")
DB_PATH  = pathlib.Path("qdrant_db_text")
COLLECTION = "gale_text"
MANIFEST_PATH = pathlib.Path("ingestion_manifest.json")
BATCH_SIZE = 32


def extract_pages(pdf_path: pathlib.Path) -> list[dict]:
    """Extract text from each PDF page using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("ERROR: pypdf not installed. Run: pip install pypdf"); sys.exit(1)

    reader = PdfReader(str(pdf_path))
    pages = []
    print(f"Extracting text from {len(reader.pages)} pages …")
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            text = f"[Page {i+1}: visual/image content – no extractable text]"
        pages.append({"page_number": i + 1, "text": text})
        if (i + 1) % 50 == 0:
            print(f"  … extracted {i+1}/{len(reader.pages)} pages")
    print(f"Done. {len(pages)} pages extracted.")
    return pages


def embed_batches(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    print("Loading embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(texts)} pages in batches of {BATCH_SIZE} …")
    all_vecs: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.extend(vecs.tolist())
        done = min(start + BATCH_SIZE, len(texts))
        print(f"  … embedded {done}/{len(texts)}")
    return all_vecs


def build_qdrant(pages: list[dict], vectors: list[list[float]]) -> None:
    """Create fresh local Qdrant collection with text payloads."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    dim = len(vectors[0])
    print(f"\nCreating Qdrant collection '{COLLECTION}' (dim={dim}) at {DB_PATH} …")

    # Remove old db if it exists to avoid schema conflicts
    if DB_PATH.exists():
        import shutil
        shutil.rmtree(DB_PATH)
        print(f"  Removed old {DB_PATH}/")

    client = QdrantClient(path=str(DB_PATH))
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=p["page_number"],
            vector=v,
            payload={
                "page_number": p["page_number"],
                "text": p["text"],
                "source_uri": f"local://gale_encyclopedia/page_{p['page_number']}",
                "authority_level": "textbook",
            },
        )
        for p, v in zip(pages, vectors)
    ]

    print(f"Upserting {len(points)} points …")
    for start in range(0, len(points), 100):
        batch = points[start : start + 100]
        client.upsert(collection_name=COLLECTION, points=batch)
        print(f"  … upserted {min(start+100, len(points))}/{len(points)}")

    info = client.get_collection(COLLECTION)
    print(f"\n✅ Collection '{COLLECTION}' ready: {info.points_count} points, dim={dim}")
    client.close()


def update_manifest() -> None:
    data = {
        "status": "success",
        "collection": COLLECTION,
        "db_path": str(DB_PATH),
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embed_dim": 384,
        "processed_files": [str(PDF_PATH)],
    }
    MANIFEST_PATH.write_text(json.dumps(data, indent=2))
    print(f"✅ Manifest updated at {MANIFEST_PATH}")


if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}"); sys.exit(1)

    t0 = time.perf_counter()
    pages   = extract_pages(PDF_PATH)
    vectors = embed_batches([p["text"] for p in pages])
    build_qdrant(pages, vectors)
    update_manifest()
    print(f"\n🏁 Total time: {time.perf_counter() - t0:.1f}s")
