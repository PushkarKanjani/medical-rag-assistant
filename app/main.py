"""
app/main.py
───────────
FastAPI backend for Medical RAG Assistant.
Configured for ephemeral cloud deployment (Render.com) with background
index initialization on startup, non-blocking HTTP endpoints, and live status.

Endpoints:
    - GET  /health       : Health check & real-time index status
    - POST /api/chat     : Non-blocking chat inference through LangGraph RAG agent
    - POST /api/ingest   : Manual trigger for PDF encyclopedia ingestion
    - GET  /             : Static Web UI (static/index.html)

Run via:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import BASE_DIR, CHROMA_PATH, BM25_PATH, get_rag_app, reset_resource_cache

# Track background ingestion status
_is_ingesting: bool = False
_ingest_error: str | None = None


def are_indexes_ready() -> bool:
    """Checks if local ChromaDB and BM25 index files exist on disk without loading them."""
    return CHROMA_PATH.exists() and BM25_PATH.exists()


def run_background_auto_ingest():
    """Executes the ingestion pipeline in a background thread and resets caches when finished."""
    global _is_ingesting, _ingest_error
    if _is_ingesting:
        return

    _is_ingesting = True
    _ingest_error = None

    try:
        print("\n" + "=" * 65)
        print("🔄 [Auto-Ingest] Starting background PDF ingestion from data/pdf/...")
        print("=" * 65 + "\n")

        from app.ingest import run_ingestion
        run_ingestion()

        # Invalidate resource cache so fresh indexes are loaded on the next request
        reset_resource_cache()
        print("\n✅ [Auto-Ingest] Ingestion finished! ChromaDB & BM25 indexes are ready.\n")

    except Exception as exc:
        _ingest_error = str(exc)
        print(f"\n⚠️ [Auto-Ingest] Ingestion failed: {exc}\n")
    finally:
        _is_ingesting = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan:
    - Binds port immediately (< 1s) so Render port checks pass instantly.
    - If indexes are missing (e.g. fresh Render deploy / container restart),
      spawns background thread auto-ingestion concurrently without holding port binding.
    """
    if not are_indexes_ready():
        pdf_dir = BASE_DIR / "data" / "pdf"
        has_pdfs = pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))
        if has_pdfs:
            print("🚀 [Startup] Spawning background auto-ingestion thread...")
            threading.Thread(target=run_background_auto_ingest, daemon=True).start()
        else:
            print("⚠️ [Startup] No PDF files found in data/pdf/. Auto-ingestion waiting for upload.")
    else:
        print("✅ [Startup] Local indexes found and ready.")

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


class IngestResponse(BaseModel):
    status: str
    message: str


@app.get("/health")
async def health_check():
    """
    Instant health check endpoint.
    Returns 200 OK immediately for Render port-checking while reporting live index status.
    """
    ready = are_indexes_ready()
    return {
        "status": "ok",
        "indexes_ready": ready,
        "is_ingesting": _is_ingesting,
        "ingest_error": _ingest_error,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint:
    Non-blocking endpoint that provides immediate status feedback during startup ingestion
    and answers medical questions through the LangGraph RAG agent once ready.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. If currently indexing in the background, return non-blocking informative status
    if _is_ingesting:
        return ChatResponse(
            answer=(
                "⏳ **Indexing in Progress**: The Gale Encyclopedia of Medicine is currently being processed in the background on this server (~2 minutes on startup). "
                "The live status badge at the top right of the page will turn green once indexing completes. Please try your question again in just a moment!"
            )
        )

    # 2. If indexes are not ready and not currently ingesting, trigger background auto-ingestion
    if not are_indexes_ready():
        pdf_dir = BASE_DIR / "data" / "pdf"
        if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
            threading.Thread(target=run_background_auto_ingest, daemon=True).start()
            return ChatResponse(
                answer=(
                    "🔄 **Indexing Initiated**: Medical indexes were missing on this ephemeral container. "
                    "Auto-ingestion has started in the background. Please wait 1-2 minutes and submit your query again."
                )
            )

        return ChatResponse(
            answer=(
                "⚠️ **Indexes Missing**: No vector indexes were found and no source PDF was detected in `data/pdf/`. "
                "Please ensure the source PDF is present in the repository and trigger the `/api/ingest` endpoint."
            )
        )

    # 3. Process question through LangGraph RAG Agent
    try:
        initial_state = {
            "question": question,
            "history": req.history[-8:] if req.history else [],
            "search_query": "",
            "evidence": [],
            "answer": "",
        }

        rag_app = get_rag_app()
        result = rag_app.invoke(initial_state)
        answer = result.get("answer", "The system failed to generate a response.")

        return ChatResponse(answer=answer)

    except Exception as exc:
        print(f"❌ Error during chat execution: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest", response_model=IngestResponse)
async def trigger_ingest():
    """
    Manual trigger fallback for building or rebuilding indexes in the background.
    """
    global _is_ingesting

    if _is_ingesting:
        return IngestResponse(
            status="already_running",
            message="Ingestion is already in progress in the background. Monitor /health for status."
        )

    pdf_dir = BASE_DIR / "data" / "pdf"
    has_pdfs = pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))

    if not has_pdfs:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF files found in {pdf_dir}. Please place your encyclopedia PDF in data/pdf/ first."
        )

    threading.Thread(target=run_background_auto_ingest, daemon=True).start()
    return IngestResponse(
        status="started",
        message="Ingestion process started in background. Monitor /health for readiness."
    )


# Static frontend mounting (Serves static/index.html at root '/')
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
