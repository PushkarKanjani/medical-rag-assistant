"""
app/main.py
───────────
FastAPI backend for Medical RAG Assistant.
Optimized for ephemeral cloud deployment (Render.com).

Features:
- Instant port binding (< 1s) to pass Render port detection.
- Automated startup background auto-ingestion if vector indexes are missing.
- Blocks /api/chat until indexes finish building, then answers user query seamlessly.
- Instant 200 OK /health probe endpoint.
- Manual fallback POST /api/ingest endpoint.

Run via:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import BASE_DIR, CHROMA_PATH, BM25_PATH, get_rag_app, reset_resource_cache

# Ingestion synchronization events
_is_ingesting: bool = False
_ingest_error: str | None = None
_indexes_ready_event = threading.Event()


def are_indexes_ready() -> bool:
    """Checks if local ChromaDB and BM25 index files exist on disk without loading them."""
    return CHROMA_PATH.exists() and BM25_PATH.exists()


def run_background_auto_ingest():
    """Executes the ingestion pipeline in a background thread and signals completion."""
    global _is_ingesting, _ingest_error
    if _is_ingesting:
        return

    _is_ingesting = True
    _ingest_error = None
    _indexes_ready_event.clear()

    try:
        print("\n" + "=" * 65)
        print("🔄 [Auto-Ingest] Checking source PDF in data/pdf/...")
        pdf_dir = BASE_DIR / "data" / "pdf"
        has_pdfs = pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))

        if not has_pdfs:
            _ingest_error = "No PDF files found in data/pdf/"
            print(f"⚠️ [Auto-Ingest] {_ingest_error}")
            return

        print("🔄 [Auto-Ingest] Starting background PDF ingestion from data/pdf/...")
        print("=" * 65 + "\n")

        from app.ingest import run_ingestion
        run_ingestion()

        # Invalidate resource cache and signal readiness
        reset_resource_cache()
        _indexes_ready_event.set()
        print("\n✅ [Auto-Ingest] Ingestion complete! Indexes are now live and ready.\n")

    except Exception as exc:
        _ingest_error = str(exc)
        print(f"\n⚠️ [Auto-Ingest] Ingestion failed: {exc}\n")
    finally:
        _is_ingesting = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan startup:
    - Binds port immediately for Render.
    - Automatically checks indexes and triggers background ingestion if missing.
    """
    if are_indexes_ready():
        _indexes_ready_event.set()
        print("✅ [Startup] Local indexes found and ready.")
    else:
        _indexes_ready_event.clear()
        print("🚀 [Startup] Missing indexes detected on ephemeral storage.")
        print("🚀 [Startup] Spawning background auto-ingestion thread...")
        threading.Thread(target=run_background_auto_ingest, daemon=True).start()

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
    return {
        "status": "ok",
        "indexes_ready": _indexes_ready_event.is_set() or are_indexes_ready(),
        "is_ingesting": _is_ingesting,
        "ingest_error": _ingest_error,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint:
    If background ingestion is in progress, waits for it to complete before answering.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # If indexes are not ready yet:
    if not _indexes_ready_event.is_set() and not are_indexes_ready():
        if _is_ingesting:
            print("⏳ [Chat] Query received while indexing. Waiting for index completion...")
            # Wait up to 180 seconds for the background ingestion to finish
            completed = _indexes_ready_event.wait(timeout=180)
            if not completed:
                return ChatResponse(
                    answer=(
                        "⏳ **Indexing In Progress**: The medical encyclopedia is still compiling on this server. "
                        "Please try your question again in 1-2 minutes."
                    )
                )
        else:
            # Not ingesting and not ready, start it now
            threading.Thread(target=run_background_auto_ingest, daemon=True).start()
            return ChatResponse(
                answer=(
                    "🔄 **Indexing Initiated**: Vector indexes were missing on this ephemeral container. "
                    "Auto-ingestion has started in the background. Please wait 2 minutes and submit your query again."
                )
            )

    # Process question through LangGraph RAG Agent
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
