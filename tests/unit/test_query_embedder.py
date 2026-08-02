from __future__ import annotations

import math

import pytest

from src.pipeline.query_embedder import (
    DEFAULT_EMBED_DIM,
    DeterministicPseudoEmbedder,
    get_default_embedder,
)


class TestDeterministicPseudoEmbedder:
    @pytest.mark.asyncio
    async def test_embed_query_returns_correct_dimension(self) -> None:
        embedder = DeterministicPseudoEmbedder(dim=64)
        result = await embedder.embed_query("hello world")
        assert len(result) == 64
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_query_default_dim(self) -> None:
        embedder = DeterministicPseudoEmbedder()
        result = await embedder.embed_query("test")
        assert len(result) == DEFAULT_EMBED_DIM

    @pytest.mark.asyncio
    async def test_embed_query_is_l2_normalized(self) -> None:
        embedder = DeterministicPseudoEmbedder(dim=128)
        result = await embedder.embed_query("hypertension guidelines")
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_embed_query_is_deterministic(self) -> None:
        embedder = DeterministicPseudoEmbedder(dim=64)
        a = await embedder.embed_query("fever and rash")
        b = await embedder.embed_query("fever and rash")
        assert a == b

    @pytest.mark.asyncio
    async def test_embed_query_empty_string(self) -> None:
        embedder = DeterministicPseudoEmbedder(dim=32)
        result = await embedder.embed_query("")
        assert result == [0.0] * 32

    @pytest.mark.asyncio
    async def test_embed_query_different_inputs_differ(self) -> None:
        embedder = DeterministicPseudoEmbedder(dim=128)
        a = await embedder.embed_query("aspirin dosage")
        b = await embedder.embed_query("warfarin interaction")
        assert a != b

    def test_invalid_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            DeterministicPseudoEmbedder(dim=5)
        with pytest.raises(ValueError):
            DeterministicPseudoEmbedder(dim=0)

    def test_get_default_embedder_returns_cached(self) -> None:
        a = get_default_embedder()
        b = get_default_embedder()
        assert a is b
        assert isinstance(a, DeterministicPseudoEmbedder)
        assert a.dim == DEFAULT_EMBED_DIM
