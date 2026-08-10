"""
app/chunking.py
───────────────
Markdown text chunking with structural paragraph preservation,
sentence-level splitting for oversized sections, and context overlap.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any
from app.config import settings


def split_paragraph_into_sentences(paragraph: str, max_chars: int) -> list[str]:
    """
    Splits a large paragraph into sentences if it exceeds max_chars.
    If an individual sentence still exceeds max_chars, it is split on word boundaries.
    """
    if len(paragraph) <= max_chars:
        return [paragraph]

    # Split on sentence-ending punctuation followed by whitespace
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    raw_sentences = sentence_pattern.split(paragraph)

    units: list[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue

        # If a single sentence is exceptionally large, split by words
        if len(s_clean) > max_chars:
            words = s_clean.split()
            current_unit: list[str] = []
            current_len = 0
            for w in words:
                word_len = len(w) + (1 if current_unit else 0)
                if current_len + word_len > max_chars and current_unit:
                    units.append(" ".join(current_unit))
                    current_unit = [w]
                    current_len = len(w)
                else:
                    current_unit.append(w)
                    current_len += word_len
            if current_unit:
                units.append(" ".join(current_unit))
        else:
            units.append(s_clean)

    return units


def chunk_markdown_page(
    page_text: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Chunks a single page of Markdown text:
    1. Splits by double newlines (\\n\\n) to preserve paragraph structure.
    2. Splits paragraphs exceeding chunk_target_chars into sentences.
    3. Aggregates paragraphs/sentences into chunks up to chunk_target_chars with chunk_overlap_chars.
    4. Attaches flat metadata including source_file, part, page, chunk_id, and content_hash.

    Args:
        page_text: Markdown text extracted from the PDF page.
        metadata: Page-level metadata (e.g. source_file, part, page).

    Returns:
        List of chunk dictionaries: [{'text': str, 'metadata': dict}].
    """
    if not page_text or not page_text.strip():
        return []

    meta = dict(metadata or {})
    target_chars = int(settings.get("chunk_target_chars", 2000))
    overlap_chars = int(settings.get("chunk_overlap_chars", 300))

    # 1. Split page into paragraphs by double newlines
    raw_paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    # 2. Break any large paragraph down to sentence/word level units
    base_units: list[str] = []
    for p in raw_paragraphs:
        if len(p) > target_chars:
            base_units.extend(split_paragraph_into_sentences(p, target_chars))
        else:
            base_units.append(p)

    if not base_units:
        return []

    # 3. Accumulate units into chunks with overlap
    raw_chunks: list[str] = []
    current_units: list[str] = []
    current_len = 0

    for unit in base_units:
        unit_len = len(unit)
        sep_len = 2 if current_units else 0

        if current_len + unit_len + sep_len > target_chars and current_units:
            chunk_str = "\n\n".join(current_units).strip()
            if chunk_str:
                raw_chunks.append(chunk_str)

            # Build overlap from trailing units
            overlap_units: list[str] = []
            overlap_len = 0
            for u in reversed(current_units):
                u_cost = len(u) + (2 if overlap_units else 0)
                if overlap_len + u_cost <= overlap_chars:
                    overlap_units.insert(0, u)
                    overlap_len += u_cost
                else:
                    break

            # Avoid infinite loop if overlap equals the full set of units
            if len(overlap_units) >= len(current_units):
                overlap_units = overlap_units[-1:]

            current_units = list(overlap_units)
            current_len = sum(len(u) for u in current_units) + max(0, (len(current_units) - 1) * 2)

        current_units.append(unit)
        current_len += len(unit) + (2 if len(current_units) > 1 else 0)

    if current_units:
        chunk_str = "\n\n".join(current_units).strip()
        if chunk_str:
            raw_chunks.append(chunk_str)

    # 4. Resolve metadata identifiers
    # Extract / format part identifier
    part_val = meta.get("part")
    if part_val is not None:
        part_str = str(part_val) if str(part_val).startswith("part") else f"part{part_val}"
    else:
        # Infer part from source_file if present (e.g. "..._part2.pdf")
        source_name = str(meta.get("source_file", ""))
        part_match = re.search(r"part(\d+)", source_name, re.IGNORECASE)
        part_str = f"part{part_match.group(1)}" if part_match else "part1"

    # Extract page identifier
    page_num = meta.get("page", 1)
    try:
        page_int = int(page_num)
    except (ValueError, TypeError):
        page_int = 1

    # Format part as string or int cleanly for metadata
    part_clean = int(re.search(r"\d+", part_str).group(0)) if re.search(r"\d+", part_str) else 1

    # 5. Build output dictionaries with flat metadata
    results: list[dict[str, Any]] = []
    for idx, chunk_text in enumerate(raw_chunks, start=1):
        chunk_id = f"{part_str}-p{page_int}-c{idx:02d}"
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        # Build flat metadata for ChromaDB compatibility
        chunk_meta: dict[str, Any] = {
            "source_file": str(meta.get("source_file", "")),
            "part": part_clean,
            "page": page_int,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "content_hash": content_hash,
        }

        # Add any other primitive metadata from incoming meta
        for k, v in meta.items():
            if k not in chunk_meta and isinstance(v, (str, int, float, bool)):
                chunk_meta[k] = v

        results.append({
            "text": chunk_text,
            "metadata": chunk_meta,
        })

    return results
