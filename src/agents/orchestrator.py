from __future__ import annotations

import time
from typing import Any

from src.orchestration.main_graph import run_query


async def orchestrate_query(query: str, session_id: str) -> dict[str, Any]:
    """Bridge adapter from the legacy /api/v1/agent/query endpoint to the new
    LangGraph orchestration system.

    Parameters
    ----------
    query:
        The natural-language user question.
    session_id:
        Client session identifier (mapped to user_id in the new system).

    Returns
    -------
    dict with keys ``thought_process``, ``answer`` and ``execution_time_ms``
    matching the legacy QueryResponse schema.
    """
    start = time.perf_counter()

    graph_result = await run_query(
        query=query,
        user_id=session_id,
        abha_id=None,
        max_iterations=3,
    )

    execution_time_ms = int((time.perf_counter() - start) * 1000)

    audit_trail = graph_result.get("audit_trail", [])
    steps = [entry.get("node", "?") for entry in audit_trail]
    thought_process = f"Agent steps executed: {' -> '.join(steps) if steps else 'n/a'}"

    confidence = graph_result.get("confidence_vector", {})
    if confidence:
        details = ", ".join(f"{k}={v}" for k, v in confidence.items())
        thought_process += f" | Confidence: {details}"

    answer = graph_result.get("final_answer", "No answer generated.")

    return {
        "thought_process": thought_process,
        "answer": answer,
        "execution_time_ms": execution_time_ms,
    }
