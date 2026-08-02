from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTION_MANIFEST_PATH = PROJECT_ROOT / "ingestion_manifest.json"
LOCAL_QDRANT_PATH = PROJECT_ROOT / "qdrant_db_text"  # text-indexed collection
LOCAL_QDRANT_PATH_LEGACY = PROJECT_ROOT / "qdrant_db"  # ColPali legacy (fallback)


@dataclass
class IngestionManifest:
    """Parsed contents of ``ingestion_manifest.json`` describing the pre-built
    Qdrant collection produced by the off-line embedding step."""

    status: str
    collection: str
    processed_files: list[str]
    embed_dim: int = 384
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    db_path: str = ""

    @classmethod
    def load(cls, path: Path = INGESTION_MANIFEST_PATH) -> "IngestionManifest":
        if not path.exists():
            logger.warning("Ingestion manifest not found at %s – using defaults", path)
            return cls(
                status="missing",
                collection="gale_text",
                processed_files=[],
                embed_dim=384,
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            status=str(raw.get("status", "unknown")),
            collection=str(raw.get("collection", "gale_text")),
            processed_files=list(raw.get("processed_files", [])),
            embed_dim=int(raw.get("embed_dim", 384)),
            embed_model=str(raw.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")),
            db_path=str(raw.get("db_path", "")),
        )


class IngestionManager:
    """Loads the pre-existing Qdrant collection produced by the off-line ingestion
    pipeline and exposes helpers for building evidence retrieval for the orchestration graph.

    The manager lazily connects to either a remote Qdrant server (when ``QDRANT_URL`` is
    configured) or falls back to the on-disk ``qdrant_db/`` local folder when the
    on-disk ``qdrant_db/`` folder.  A single instance is cached per-process.
    """

    _instance: "IngestionManager | None" = None

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._manifest: IngestionManifest | None = None
        self._client: AsyncQdrantClient | None = None

    # ── Singleton access ────────────────────────────────────────────────

    @classmethod
    def get_singleton(cls) -> "IngestionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None

    # ── Manifest helpers ────────────────────────────────────────────

    @property
    def manifest(self) -> IngestionManifest:
        if self._manifest is None:
            self._manifest = IngestionManifest.load()
        return self._manifest

    @property
    def collection_name(self) -> str:
        return self.manifest.collection

    # ── Client lifecycle ─────────────────────────────────────────

    async def get_client(self) -> AsyncQdrantClient:
        if self._client is not None:
            return self._client

        settings = self._settings
        qdrant_url = (settings.qdrant_url or "").strip()
        qdrant_api_key = (settings.qdrant_api_key or "").strip()

        try:
            if qdrant_url:
                logger.info("Connecting to remote Qdrant at %s", qdrant_url)
                self._client = AsyncQdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key or None,
                )
            elif LOCAL_QDRANT_PATH.exists():
                logger.info("Using text-indexed local Qdrant at %s", LOCAL_QDRANT_PATH)
                self._client = AsyncQdrantClient(path=str(LOCAL_QDRANT_PATH))
            elif LOCAL_QDRANT_PATH_LEGACY.exists():
                logger.warning("Text DB not found; falling back to legacy ColPali DB at %s", LOCAL_QDRANT_PATH_LEGACY)
                self._client = AsyncQdrantClient(path=str(LOCAL_QDRANT_PATH_LEGACY))
            else:
                logger.warning(
                    "No Qdrant DB found – search will return empty results. "
                    "Run: python scripts/rebuild_index.py"
                )
                self._client = None
        except Exception as exc:  # pragma: no cover – defensive
            logger.exception("Failed to construct Qdrant client: %s", exc)
            self._client = None

        if self._client is not None:
            try:
                await self._client.get_collections()
            except Exception as exc:
                logger.warning("Qdrant health check failed: %s", exc)

        return self._client  # type: ignore[return-value]

    async def ensure_collection(self) -> bool:
        """Ensure the expected collection is accessible.  Returns ``True`` when the
        collection already exists, ``False`` otherwise (callers can continue
        with mock / fallback behaviour)."""
        client = await self.get_client()
        if client is None:
            return False
        try:
            collections = await client.get_collections()
            names = {c.name for c in collections.collections}
            if self.collection_name in names:
                return True
            logger.warning(
                "Collection %s not found (have %s)",
                self.collection_name,
                sorted(names),
            )
            return False
        except Exception as exc:
            logger.warning("ensure_collection failed: %s", exc)
            return False

    async def collection_info(self) -> dict[str, Any]:
        """Return a small dict describing the collection status of the collection info for
        diagnostic / health endpoints."""
        client = await self.get_client()
        info: dict[str, Any] = {
            "manifest_status": self.manifest.status,
            "collection": self.collection_name,
            "processed_files": self.manifest.processed_files,
            "connected": client is not None,
        }
        if client is not None:
            try:
                col = await client.get_collection(self.collection_name)
                info["points_count"] = getattr(col, "points_count", None)
                info["vectors_count"] = getattr(col, "vectors_count", None)
            except Exception as exc:
                info["error"] = str(exc)
        return info

    # ── Evidence retrieval ────────────────────────────────────────

    async def search_evidence(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Run a similarity search on the ingestion-managed collection, returning
        evidence dicts compatible with :class:`GraphState["candidate_evidence"].

        When the Qdrant client is unavailable or the call raises, an empty
        list is returned so that the orchestration graph can degrade gracefully.
        """
        client = await self.get_client()
        if client is None or not query_embedding:
            return []

        try:
            hits = await client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                with_vector=False,
            )
        except Exception as exc:
            logger.warning("search_evidence failed: %s", exc)
            return []

        evidence: list[dict[str, Any]] = []
        for idx, hit in enumerate(hits):
            payload = hit.payload or {}
            evidence.append({
                "chunk_id": f"qdrant-{hit.id}",
                "text": str(payload.get("text", "") or ""),
                "page_number": int(payload.get("page_number", 0)),
                "bbox": tuple(payload.get("bbox", [0.0, 0.0, 1.0, 1.0])),
                "source_uri": str(payload.get("source_uri", payload.get("source", ""))),
                "score": float(hit.score),
                "channel": "local",
                "authority_level": str(payload.get("authority_level", "reference")),
            })
        return evidence


async def load_store() -> IngestionManager:
    """Module-level factory matching the hand-off note: load the pre-built store
    and return the singleton :class:`IngestionManager` ready for use by
    the orchestrator nodes."""
    manager = IngestionManager.get_singleton()
    await manager.ensure_collection()
    return manager
