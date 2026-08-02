from __future__ import annotations

import pytest

from src.api.v1.chat import ChatRequest, ChatResponse, chat
from src.orchestration.main_graph import run_query


@pytest.mark.asyncio
async def test_clinician_flow_completes_with_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    audit_calls: list[dict] = []

    async def fake_append_audit(*, user_id: str, node: str, status: str, latency_ms: int = 0, error: str | None = None, details: dict | None = None) -> None:
        audit_calls.append({
            "user_id": user_id,
            "node": node,
            "status": status,
            "latency_ms": latency_ms,
            "error": error,
            "details": details,
        })

    monkeypatch.setattr("src.api.v1.chat.append_audit", fake_append_audit)

    request = ChatRequest(
        query="What is the dosage guidance for aspirin?",
        user_id="clinician-1",
        abha_id=None,
        max_iterations=3,
    )

    graph_result = await run_query(request.query, request.user_id, request.abha_id, request.max_iterations)
    final_answer = graph_result["final_answer"]
    assert isinstance(final_answer, str)
    assert len(final_answer) > 20, "Expected a substantive answer, not a trivial placeholder"
    assert "dosage" in final_answer.lower() or "amoxicillin" in final_answer.lower() or "dose" in final_answer.lower()
    assert graph_result["critic_report"]["verdict"] == "accept"
    audit_nodes = [entry["node"] for entry in graph_result.get("audit_trail", [])]
    assert {"triage", "plan", "search_local", "search_web", "synthesise", "critic", "output"}.issubset(set(audit_nodes))

    response = await chat(request)

    assert isinstance(response, ChatResponse)
    assert response.final_answer
    assert isinstance(response.citations, list) and response.citations
    assert isinstance(response.confidence_vector, dict)
    assert response.confidence_vector.get("local") in {0.0, 1.0}
    assert response.confidence_vector.get("web") in {0.0, 1.0}
    assert response.audit_id.startswith("audit-")

    assert audit_calls
    chat_call = audit_calls[-1]
    assert chat_call["node"] == "chat"
    assert chat_call["status"] == "completed"
    assert chat_call["details"]["query"] == request.query
    assert chat_call["details"]["audit_id"] == response.audit_id
