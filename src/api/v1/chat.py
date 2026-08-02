from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from src.observability.audit_log import append_audit
from src.orchestration.main_graph import run_query

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    user_id: str
    abha_id: str | None = None
    max_iterations: int = 3


class ChatResponse(BaseModel):
    final_answer: str
    citations: list
    confidence_vector: dict
    audit_id: str


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result = await run_query(req.query, req.user_id, req.abha_id, req.max_iterations)
    audit_trail = result.get("audit_trail", [])
    audit_id = f"audit-{req.user_id}-{len(audit_trail)}"
    await append_audit(
        user_id=req.user_id,
        node="chat",
        status="completed",
        details={"query": req.query, "audit_id": audit_id},
    )
    return ChatResponse(
        final_answer=result.get("final_answer", ""),
        citations=result.get("citations", []),
        confidence_vector=result.get("confidence_vector", {}),
        audit_id=audit_id,
    )