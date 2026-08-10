"""
app/main.py
───────────
FastAPI backend for Medical RAG Assistant.
Provides REST API endpoints and serves the static web UI.

Endpoints:
    - GET  /health       : Health check & index readiness status
    - POST /api/chat     : Chat inference through LangGraph RAG agent
    - POST /api/ingest   : Manual trigger for encyclopedia ingestion
    - GET  /             : Static Web UI (static/index.html)

Run via:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import BASE_DIR, CHROMA_PATH, BM25_PATH, get_resources, rag_app


def are_indexes_ready() -> bool:
    """Checks if both ChromaDB and BM25 index files exist on disk."""
    return CHROMA_PATH.exists() and BM25_PATH.exists()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context to initialize and pre-warm heavy resources (ChromaDB, BM25,
    and SentenceTransformers) once at server startup to ensure fast request handling.
    """
    print("\n" + "=" * 65)
    print("🚀 Initializing Medical RAG Assistant Backend...")
    if are_indexes_ready():
        try:
            get_resources()
            print("✅ Embeddings & Index resources pre-warmed successfully!")
        except Exception as e:
            print(f"⚠️ Notice during startup resource loading: {e}")
    else:
        print("⚠️ Notice: Local indexes (Chroma / BM25) were not found on disk.")
        print("   To build indexes, run: python -m app.ingest")
        print("   Or send a POST request to /api/ingest once source PDFs are available.")
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


class IngestResponse(BaseModel):
    status: str
    message: str


@app.get("/health")
async def health_check():
    """Health check endpoint indicating server and index readiness status."""
    indexes_present = are_indexes_ready()
    return {
        "status": "ok",
        "indexes_ready": indexes_present,
        "chroma_path": str(CHROMA_PATH),
        "bm25_path": str(BM25_PATH),
    }


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

    # Gracefully check index availability
    if not are_indexes_ready():
        return ChatResponse(
            answer=(
                "⚠️ Medical indexes (ChromaDB / BM25) have not been built on this server instance yet. "
                "Please run ingestion (`python -m app.ingest`) or trigger the `/api/ingest` endpoint to build indexes from the source PDF."
            )
        )

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


@app.post("/api/ingest", response_model=IngestResponse)
async def trigger_ingest(background_tasks: BackgroundTasks):
    """Triggers the PDF ingestion pipeline in the background."""
    from app.ingest import run_ingestion

    pdf_dir = BASE_DIR / "data" / "pdf"
    has_pdfs = pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))

    if not has_pdfs:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF files found in {pdf_dir}. Please place your encyclopedia PDF in data/pdf/ first."
        )

    background_tasks.add_task(run_ingestion)
    return IngestResponse(
        status="started",
        message="Ingestion process started in background. Monitor server logs for progress."
    )


# Static frontend mounting
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
