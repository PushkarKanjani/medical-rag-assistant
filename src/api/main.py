# src/api/main.py
"""FastAPI entrypoint for the Agentic MedAssist service.

Provides:
- ``/health`` – simple health‑check returning timestamps and env info.
- ``/api/v1/agent/query`` – accepts a JSON payload with a user query and a session ID,
  forwards the request to the orchestrator agent, and returns the structured response.
"""

from __future__ import annotations

import datetime
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.config import settings
from src.agents.orchestrator import orchestrate_query

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agentic MedAssist API",
    version="0.1.0",
    description="Asynchronous FastAPI service exposing health checks and an agent‑orchestrated query endpoint.",
    debug=settings.DEBUG,
)

# CORS – allow any origin for development; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models for request / response payloads
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Payload sent by the client to ask the agent a question.

    Attributes
    ----------
    query: str
        The natural‑language question or instruction.
    session_id: str
        Identifier for the user session; useful for traceability and stateful agents.
    """

    query: str = Field(..., description="User question or instruction")
    session_id: str = Field(..., description="Client‑provided session identifier")

class QueryResponse(BaseModel):
    """Structured response returned by the orchestrator.

    Attributes
    ----------
    thought_process: str
        Human‑readable reasoning steps performed by the agent.
    answer: str
        Final answer generated for the user.
    execution_time_ms: int
        Duration of the whole orchestration in milliseconds.
    """

    thought_process: str = Field(..., description="Agent reasoning trace")
    answer: str = Field(..., description="Generated answer for the user")
    execution_time_ms: int = Field(..., description="Total execution time in ms")

# ---------------------------------------------------------------------------
# Health‑check endpoint
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    """Return basic status information.

    The response includes the current UTC timestamp, the process start time, and
    the DEBUG flag from the settings. This endpoint is deliberately cheap and
    async‑friendly.
    """
    now = datetime.datetime.utcnow().isoformat() + "Z"
    return {
        "status": "ok",
        "timestamp": now,
        "debug": settings.DEBUG,
        "pid": time.time(),  # simple uptime indicator
    }

# ---------------------------------------------------------------------------
# Agent query endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/agent/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent"],
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Accept a query, forward it to the orchestrator, and return the result.

    Parameters
    ----------
    request: QueryRequest
        The incoming payload containing ``query`` and ``session_id``.
    """
    try:
        result = await orchestrate_query(request.query, request.session_id)
    except Exception as exc:  # pragma: no cover – defensive generic handling
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestration failed: {exc}",
        ) from exc

    return QueryResponse(**result)

# ---------------------------------------------------------------------------
# Development helper – run with ``python -m src.api.main``
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
    )
