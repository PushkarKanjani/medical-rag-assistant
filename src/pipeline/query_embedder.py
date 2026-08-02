from __future__ import annotations

import hashlib
import math
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_EMBED_DIM = 128


class QueryEmbedder(Protocol):
    """Interface expected by search nodes that need to vectorise a query."""

    async def embed_query(self, text: str) -> list[float]: ...


class DeterministicPseudoEmbedder:
    """Fast, dependency-free query embedder suitable when the heavyweight ColPali
    vision-language model is not available (e.g. unit tests, dev environments
    without GPU).

    Produces L2-normalised ``dim``-dimensional vectors using a seeded hash of
    the input text.  The embeddings are **not** semantically meaningful but
    are deterministic (identical queries → identical vectors) which is enough
    for a Qdrant lookup against a collection indexed with the same scheme, and
    for exercising end-to-end retrieval pipelines in tests.
    """

    def __init__(self, dim: int = DEFAULT_EMBED_DIM) -> None:
        if dim < 8 or dim % 4 != 0:
            raise ValueError("dim must be a multiple of 4 and >= 8")
        self.dim = dim

    async def embed_query(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dim

        raw: list[float] = []
        seed_bytes = text.encode("utf-8")
        counter = 0
        while len(raw) < self.dim:
            h = hashlib.sha256(seed_bytes + counter.to_bytes(4, "little"))
            digest = h.digest()
            for i in range(0, min(32, (self.dim - len(raw)) * 4), 4):
                u32 = int.from_bytes(digest[i:i + 4], "little", signed=False)
                value = (u32 / 0xFFFFFFFF) * 2.0 - 1.0
                raw.append(value)
            counter += 1

        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


_EMBEDDER: DeterministicPseudoEmbedder | None = None


def get_default_embedder() -> DeterministicPseudoEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = DeterministicPseudoEmbedder()
    return _EMBEDDER
