"""
src/pipeline/bm25_retriever.py
──────────────────────────────
Pure-Python BM25 retrieval over the extracted PDF page text.
No torch, no GPU, no Qdrant needed — works everywhere.

The retriever loads pages.json once (lazy, cached) and ranks
pages by BM25 score against a query.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

_PAGES_PATH = pathlib.Path("data/processed/pages.json")

# BM25 hyperparameters
K1 = 1.5
B  = 0.75

_CACHE: list[dict] | None = None


def _load_pages() -> list[dict]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not _PAGES_PATH.exists():
        logger.warning("pages.json not found at %s – returning empty corpus", _PAGES_PATH)
        _CACHE = []
        return _CACHE
    _CACHE = json.loads(_PAGES_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d pages from %s", len(_CACHE), _PAGES_PATH)
    return _CACHE


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(corpus_tokens: list[list[str]], vocab: set[str]) -> dict[str, float]:
    N = len(corpus_tokens)
    idf: dict[str, float] = {}
    for term in vocab:
        df = sum(1 for doc in corpus_tokens if term in doc)
        idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
    return idf


def bm25_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Rank all pages by BM25 score for *query* and return top_k evidence dicts.
    Each dict is compatible with GraphState["candidate_evidence"].
    """
    pages = _load_pages()
    if not pages:
        return []

    q_terms = _tokenize(query)
    if not q_terms:
        return []

    # Tokenise corpus (compute on each call — fast enough for 500 pages)
    corpus_tokens = [set(_tokenize(p["text"])) for p in pages]
    doc_lengths   = [len(_tokenize(p["text"])) for p in pages]
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)

    vocab = set(q_terms)
    idf   = _build_idf(corpus_tokens, vocab)  # type: ignore[arg-type]

    scores: list[tuple[float, dict]] = []
    for i, page in enumerate(pages):
        doc_terms = _tokenize(page["text"])
        tf_map: dict[str, int] = {}
        for t in doc_terms:
            tf_map[t] = tf_map.get(t, 0) + 1

        dl   = doc_lengths[i]
        score = 0.0
        for term in q_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            numerator   = tf * (K1 + 1)
            denominator = tf + K1 * (1 - B + B * dl / avgdl)
            score += idf.get(term, 0.0) * numerator / denominator

        if score > 0:
            scores.append((score, page))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:top_k]

    evidence: list[dict] = []
    for rank, (score, page) in enumerate(top):
        text = page["text"]
        # Return a sensible snippet (first 800 chars) rather than full page
        snippet = text[:800].strip()
        evidence.append({
            "chunk_id": f"bm25-page-{page['page_number']}",
            "text": snippet,
            "full_text": text,          # kept for Groq context
            "page_number": page["page_number"],
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "source_uri": f"local://gale_encyclopedia/page_{page['page_number']}",
            "score": round(score, 4),
            "channel": "local",
            "authority_level": "textbook",
        })

    return evidence


def invalidate_cache() -> None:
    """Clear the loaded pages cache (useful for testing)."""
    global _CACHE
    _CACHE = None
