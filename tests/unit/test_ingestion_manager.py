from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.ingestion_manager import (
    INGESTION_MANIFEST_PATH,
    IngestionManifest,
    IngestionManager,
    load_store,
)


class TestIngestionManifest:
    def test_load_manifest_parse_existing_file(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "status": "success",
            "collection": "test_pages",
            "processed_files": ["a.pdf", "b.pdf"],
        }))
        manifest = IngestionManifest.load(manifest_path)
        assert manifest.status == "success"
        assert manifest.collection == "test_pages"
        assert manifest.processed_files == ["a.pdf", "b.pdf"]

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.json"
        manifest = IngestionManifest.load(missing)
        assert manifest.status == "missing"
        assert manifest.collection == "medassist_pages"
        assert manifest.processed_files == []


class TestIngestionManager:
    def setup_method(self) -> None:
        IngestionManager.reset_singleton()

    def teardown_method(self) -> None:
        IngestionManager.reset_singleton()

    def test_singleton_returns_same_instance(self) -> None:
        a = IngestionManager.get_singleton()
        b = IngestionManager.get_singleton()
        assert a is b

    def test_reset_singleton_clears_cache(self) -> None:
        a = IngestionManager.get_singleton()
        IngestionManager.reset_singleton()
        b = IngestionManager.get_singleton()
        assert a is not b

    def test_manifest_property_is_cached(self) -> None:
        manager = IngestionManager()
        m1 = manager.manifest
        m2 = manager.manifest
        assert m1 is m2

    def test_collection_name_from_manifest(self) -> None:
        manager = IngestionManager()
        assert manager.collection_name == manager.manifest.collection
        assert manager.collection_name == "medassist_pages"

    @pytest.mark.asyncio
    async def test_get_client_no_env_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeSettings:
            qdrant_url = ""
            qdrant_api_key = ""

        manager = IngestionManager(settings=FakeSettings())  # type: ignore[arg-type]
        client = await manager.get_client()
        assert client is None

    @pytest.mark.asyncio
    async def test_ensure_collection_no_client_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeSettings:
            qdrant_url = ""
            qdrant_api_key = ""

        manager = IngestionManager(settings=FakeSettings())  # type: ignore[arg-type]
        ok = await manager.ensure_collection()
        assert ok is False

    @pytest.mark.asyncio
    async def test_collection_info_without_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeSettings:
            qdrant_url = ""
            qdrant_api_key = ""

        manager = IngestionManager(settings=FakeSettings())  # type: ignore[arg-type]
        info = await manager.collection_info()
        assert info["connected"] is False
        assert info["collection"] == "medassist_pages"
        assert "processed_files" in info

    @pytest.mark.asyncio
    async def test_search_evidence_no_client_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeSettings:
            qdrant_url = ""
            qdrant_api_key = ""

        manager = IngestionManager(settings=FakeSettings())  # type: ignore[arg-type]
        result = await manager.search_evidence([0.1, 0.2], limit=3)
        assert result == []


@pytest.mark.asyncio
async def test_load_store_returns_manager() -> None:
    IngestionManager.reset_singleton()
    result = await load_store()
    assert isinstance(result, IngestionManager)
    IngestionManager.reset_singleton()
