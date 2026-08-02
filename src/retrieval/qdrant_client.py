from __future__ import annotations

from functools import lru_cache

try:
    from qdrant_client import AsyncQdrantClient, models
except ImportError:
    class _Distance:
        COSINE = "Cosine"


    class _VectorParams:
        def __init__(self, size: int, distance: str) -> None:
            self.size = size
            self.distance = distance


    class _Models:
        Distance = _Distance
        VectorParams = _VectorParams


    class AsyncQdrantClient:
        def __init__(self, url: str = "", api_key: str = "") -> None:
            self.url = url
            self.api_key = api_key

        async def collection_exists(self, collection_name: str) -> bool:
            return False

        async def create_collection(self, collection_name: str, vectors_config: object) -> None:
            return None

        async def upsert(self, collection_name: str, points: list) -> None:
            return None


    models = _Models()

from src.settings import get_settings


@lru_cache(maxsize=1)
def get_async_qdrant() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


async def ensure_collection(client: AsyncQdrantClient, name: str, dim: int = 128) -> None:
    exists = await client.collection_exists(collection_name=name)
    if exists:
        return

    await client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )


async def upsert_pages(client: AsyncQdrantClient, name: str, points: list) -> None:
    await ensure_collection(client, name)
    await client.upsert(collection_name=name, points=points)