from __future__ import annotations

from src.orchestration.state import GraphState


async def critic_node(state: GraphState) -> dict:
    answer = state.get("candidate_answer", "")
    verdict = "accept" if answer else "insufficient"
    report = {
        "verdict": verdict,
        "rationale": "Mock critic evaluation based on presence of a candidate answer.",
        "confidence": 0.5 if answer else 0.0,
    }

    return {
        "critic_report": report,
        "audit_trail": [
            {
                "node": "critic",
                "status": verdict,
                "latency_ms": 0,
                "details": {"verdict": verdict},
            }
        ],
    }