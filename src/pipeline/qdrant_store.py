"""src/pipeline/qdrant_store.py
Async wrapper around Qdrant for vector storage and retrieval.
All methods are async and raise informative RuntimeError on failure.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import settings

logger = logging.getLogger(__name__)


class QdrantStore:
    """High‑level async interface for a Qdrant collection.

    The client is created lazily and cached for the process lifetime.
    """

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None

    async def _get_client(self) -> AsyncQdrantClient:
        """Instantiate (or retrieve) the AsyncQdrantClient.

        Performs a quick health‑check to surface connection problems early.
        """
        if self._client is None:
            try:
                self._client = AsyncQdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    api_key=(settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None),
                )
                await self._client.get_collections()
            except Exception as exc:
                logger.error(
                    "Unable to connect to Qdrant at %s:%s – %s",
                    settings.QDRANT_HOST,
                    settings.QDRANT_PORT,
                    exc,
                )
                raise RuntimeError("Failed to connect to Qdrant") from exc
        return self._client

    async def init_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a collection if it does not already exist, using cosine similarity."""
        client = await self._get_client()
        try:
            await client.get_collection(collection_name)
            logger.debug("Qdrant collection %s already exists", collection_name)
        except Exception:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )
            logger.info("Created Qdrant collection %s (dim=%d)", collection_name, vector_size)

    async def upsert_vectors(
        self,
        collection_name: str,
        ids: List[int],
        vectors: List[List[float]],
        payload: List[Dict[str, Any]],
    ) -> None:
        """Insert or replace a batch of vectors with associated payload metadata."""
        if not (len(ids) == len(vectors) == len(payload)):
            raise ValueError("ids, vectors, and payload must have the same length")
        client = await self._get_client()
        points = [
            qmodels.PointStruct(id=uid, vector=vec, payload=pl)
            for uid, vec, pl in zip(ids, vectors, payload)
        ]
        try:
            await client.upsert(collection_name=collection_name, points=points)
            logger.debug("Upserted %d points into %s", len(points), collection_name)
        except Exception as exc:
            logger.error("Failed to upsert vectors to %s – %s", collection_name, exc)
            raise RuntimeError("Qdrant upsert failed") from exc

    async def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top‑``limit`` most similar payloads for ``query_vector``.

        Each result dict contains ``id``, ``score`` and the original ``payload``.
        """
        client = await self._get_client()
        try:
            hits = await client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vector=False,
            )
        except Exception as exc:
            logger.error("Qdrant search failed on %s – %s", collection_name, exc)
            raise RuntimeError("Qdrant search error") from exc
        results: List[Dict[str, Any]] = []
        for hit in hits:
            results.append({"id": hit.id, "score": hit.score, "payload": hit.payload})
        return results

# Export a module‑level singleton for convenient reuse.
qdrant_store = QdrantStore()
