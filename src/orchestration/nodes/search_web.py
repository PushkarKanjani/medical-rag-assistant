from __future__ import annotations

from src.orchestration.state import GraphState


async def search_web_node(state: GraphState) -> dict:
    query = state.get("query", "")
    results = [
        {
            "title": f"Mock web result for {query}",
            "url": "https://example.com/mock",
            "snippet": f"Synthetic web snippet for {query}",
        }
    ]

    return {
        "web_results": results,
        "audit_trail": [
            {
                "node": "search_web",
                "status": "completed",
                "latency_ms": 0,
                "details": {"results": len(results)},
            }
        ],
    }