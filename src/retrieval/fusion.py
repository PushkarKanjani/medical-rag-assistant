from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Evidence(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    bbox: tuple[float, float, float, float]
    source_uri: str
    score: float
    channel: str


_DEFAULT_WEIGHTS = {"bm25": 0.25, "maxsim": 0.45, "graph": 0.30}
_DEFAULT_K = 60
_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "hybrid_fusion.yaml"


def _load_rrf_params() -> tuple[dict[str, float], int]:
    weights = dict(_DEFAULT_WEIGHTS)
    rrf_k = _DEFAULT_K

    if _WEIGHTS_PATH.exists():
        loaded = yaml.safe_load(_WEIGHTS_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            candidate_weights = loaded.get("weights")
            if isinstance(candidate_weights, dict):
                for key, value in candidate_weights.items():
                    if key in weights:
                        weights[key] = float(value)
            if "k" in loaded:
                rrf_k = int(loaded["k"])

    return weights, rrf_k


async def _bm25_search(query: str, silos: list[str], top_k: int) -> list[Evidence]:
    return []


async def _maxsim_search(query: str, silos: list[str], top_k: int) -> list[Evidence]:
    return []


async def _graph_search(query: str, silos: list[str], top_k: int) -> list[Evidence]:
    return []


def _rrf_combine(results_by_channel: dict[str, list[Evidence]], weights: dict[str, float], rrf_k: int, top_k: int) -> list[Evidence]:
    ranked: dict[str, Evidence] = {}
    combined_scores: defaultdict[str, float] = defaultdict(float)

    for channel, results in results_by_channel.items():
        weight = weights.get(channel, 0.0)
        for rank, evidence in enumerate(results, start=1):
            combined_scores[evidence.chunk_id] += weight / (rrf_k + rank)
            if evidence.chunk_id not in ranked:
                ranked[evidence.chunk_id] = evidence.model_copy(update={"score": 0.0, "channel": channel})

    combined: list[Evidence] = []
    for chunk_id, score in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        evidence = ranked[chunk_id]
        combined.append(evidence.model_copy(update={"score": score}))

    return combined


async def hybrid_search(query: str, silos: list[str], top_k: int = 10) -> list[Evidence]:
    weights, rrf_k = _load_rrf_params()

    bm25_results = await _bm25_search(query, silos, top_k)
    maxsim_results = await _maxsim_search(query, silos, top_k)
    graph_results = await _graph_search(query, silos, top_k)

    return _rrf_combine(
        {
            "bm25": bm25_results,
            "maxsim": maxsim_results,
            "graph": graph_results,
        },
        weights,
        rrf_k,
        top_k,
    )