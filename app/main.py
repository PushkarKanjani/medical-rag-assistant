"""
app/main.py
───────────
FastAPI backend for Medical RAG Assistant.
Provides REST API endpoints and serves the static web UI.

Endpoints:
    - GET  /health      : Health check
    - POST /api/chat    : Chat inference through LangGraph RAG agent
    - GET  /            : Static Web UI (static/index.html)

Run via:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import BASE_DIR, get_resources, rag_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context to initialize and pre-warm heavy resources (ChromaDB, BM25,
    and SentenceTransformers) once at server startup to ensure fast request handling.
    """
    print("\n" + "=" * 65)
    print("🚀 Initializing Medical RAG Assistant Backend...")
    try:
        get_resources()
        print("✅ Embeddings & Index resources pre-warmed successfully!")
    except Exception as e:
        print(f"⚠️ Notice during startup resource loading: {e}")
        print("   If indexes are not built yet, run: python -m app.ingest")
    print("=" * 65 + "\n")
    yield


app = FastAPI(
    title="Medical RAG Assistant",
    description="Evidence-grounded medical AI assistant powered by the Gale Encyclopedia of Medicine, Groq, and LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's medical query")
    history: List[Dict[str, str]] = Field(default_factory=list, description="Recent conversation turns (role & content)")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Evidence-grounded medical answer with citations")


@app.get("/health")
async def health_check():
    """Health check probe endpoint."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint:
    Runs query rewrite, dense+BM25 hybrid retrieval with RRF, symptom penalty,
    and grounded generation via LangGraph workflow.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        initial_state = {
            "question": question,
            "history": req.history[-8:] if req.history else [],
            "search_query": "",
            "evidence": [],
            "answer": "",
        }

        # Invoke the pre-compiled LangGraph agent
        result = rag_app.invoke(initial_state)
        answer = result.get("answer", "The system failed to generate a response.")

        return ChatResponse(answer=answer)

    except Exception as exc:
        print(f"❌ Error during chat execution: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# Static frontend mounting
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
