from __future__ import annotations

from src.orchestration.state import GraphState


async def escalate_node(state: GraphState) -> dict:
    message = "Escalated for human review due to safety or confidence concerns."

    return {
        "final_answer": message,
        "audit_trail": [
            {
                "node": "escalate",
                "status": "completed",
                "latency_ms": 0,
                "details": {"reason": state.get("critic_report", {}).get("verdict", "unknown")},
            }
        ],
    }