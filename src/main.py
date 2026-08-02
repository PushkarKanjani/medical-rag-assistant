from __future__ import annotations

from contextlib import asynccontextmanager

import asyncpg
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.abha import router as abha_router
from src.api.v1.chat import router as chat_router
from src.orchestration.main_graph import compiled
from src.retrieval.qdrant_client import get_async_qdrant
from src.settings import get_settings


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    qdrant_client = get_async_qdrant()
    try:
        supabase_pool = await asyncpg.create_pool(settings.postgres_dsn, timeout=5.0)
    except Exception as e:
        print(f"Warning: Could not connect to database pool: {e}")
        supabase_pool = None
    app.state.qdrant_client = qdrant_client
    app.state.supabase_pool = supabase_pool
    app.state.compiled_graph = compiled
    try:
        yield
    finally:
        if supabase_pool is not None:
            await supabase_pool.close()
        close_method = getattr(qdrant_client, "aclose", None)
        if callable(close_method):
            await close_method()


configure_logging()
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(abha_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "env": settings.environment}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)