from __future__ import annotations

from src.orchestration.state import GraphState


async def triage_node(state: GraphState) -> dict:
    query = state.get("query", "")
    lowered = query.lower()

    if any(token in lowered for token in ("abha", "abdm", "health id")):
        intent = "abdm_flow"
    elif any(token in lowered for token in ("drug", "dose", "dosage", "mg")):
        intent = "drug_interaction"
    elif any(token in lowered for token in ("guideline", "recommendation", "protocol")):
        intent = "guideline_search"
    elif query.strip():
        intent = "general_qna"
    else:
        intent = "escalation"

    return {
        "intent": intent,
        "language": state.get("language", "en"),
        "risk_flags": state.get("risk_flags", []),
        "audit_trail": [
            {
                "node": "triage",
                "status": "completed",
                "latency_ms": 0,
                "details": {"intent": intent},
            }
        ],
    }