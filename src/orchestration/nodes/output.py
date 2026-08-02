from __future__ import annotations

from src.orchestration.state import GraphState


async def output_node(state: GraphState) -> dict:
    answer = state.get("candidate_answer") or state.get("final_answer") or "No answer generated."

    return {
        "final_answer": answer,
        "audit_trail": [
            {
                "node": "output",
                "status": "completed",
                "latency_ms": 0,
                "details": {"answer_length": len(answer)},
            }
        ],
    }