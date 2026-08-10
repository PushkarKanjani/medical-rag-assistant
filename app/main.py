"""
app/main.py
───────────
FastAPI backend for Medical RAG Assistant.
Optimized for instant port binding on cloud platforms (Render, Railway).

Endpoints:
    - GET  /health       : Health check (instant 200 OK)
    - POST /api/chat     : Lazy-loaded chat inference through LangGraph RAG agent
    - POST /api/ingest   : Background trigger for PDF encyclopedia ingestion
    - GET  /             : Static Web UI (static/index.html)

Run via:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import BASE_DIR, CHROMA_PATH, BM25_PATH, get_rag_app


def are_indexes_ready() -> bool:
    """Checks if local ChromaDB and BM25 index files exist on disk without loading them."""
    return CHROMA_PATH.exists() and BM25_PATH.exists()


app = FastAPI(
    title="Medical RAG Assistant",
    description="Evidence-grounded medical AI assistant powered by the Gale Encyclopedia of Medicine, Groq, and LangGraph.",
    version="1.0.0",
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
    Health check endpoint.
    Returns 200 OK instantly so Render port checks pass immediately without waiting for model loading.
    """
    return {
        "status": "ok",
        "indexes_ready": are_indexes_ready(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint:
    Lazily initializes embedding model and LangGraph workflow on the first request.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Gracefully notify if indexes have not been created yet
    if not are_indexes_ready():
        return ChatResponse(
            answer=(
                "⚠️ Medical indexes (ChromaDB / BM25) have not been built on this server instance yet. "
                "Please trigger the `/api/ingest` endpoint once source PDFs are available."
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

        # Lazily retrieve and invoke the LangGraph agent
        rag_app = get_rag_app()
        result = rag_app.invoke(initial_state)
        answer = result.get("answer", "The system failed to generate a response.")

        return ChatResponse(answer=answer)

    except Exception as exc:
        print(f"❌ Error during chat execution: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest", response_model=IngestResponse)
async def trigger_ingest(background_tasks: BackgroundTasks):
    """
    Triggers the PDF ingestion pipeline as an asynchronous background task.
    Does not block HTTP response.
    """
    from app.ingest import run_ingestion

    pdf_dir = BASE_DIR / "data" / "pdf"
    has_pdfs = pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))

    if not has_pdfs:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF files found in {pdf_dir}. Please ensure source PDFs exist in data/pdf/."
        )

    background_tasks.add_task(run_ingestion)
    return IngestResponse(
        status="started",
        message="Ingestion process started in background. Monitor server logs for progress."
    )


# Static frontend mounting (Serves static/index.html at root '/')
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
