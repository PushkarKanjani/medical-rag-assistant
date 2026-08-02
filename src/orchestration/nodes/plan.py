from __future__ import annotations

from src.orchestration.state import GraphState


async def plan_node(state: GraphState) -> dict:
    query = state.get("query", "")
    sub_questions = [
        {"question": query, "priority": 1},
    ]

    if state.get("intent") == "drug_interaction":
        sub_questions.append({"question": f"Check dosage and contraindications for: {query}", "priority": 2})

    return {
        "sub_questions": sub_questions,
        "max_iterations": state.get("max_iterations", 3),
        "audit_trail": [
            {
                "node": "plan",
                "status": "completed",
                "latency_ms": 0,
                "details": {"sub_questions": len(sub_questions)},
            }
        ],
    }