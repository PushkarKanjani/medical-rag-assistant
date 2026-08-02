from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Legacy configuration loaded from environment variables.

    This module is kept for backwards compatibility with the original
    ``/api/v1/agent/query`` endpoint.  New code should prefer
    :mod:`src.settings` which defines the unified ``AppSettings`` class.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Secrets (optional – new-style names are aliased below) ──
    GROQ_API_KEY: Optional[SecretStr] = Field(
        default=None, description="Groq LLM API key (kept secret).")
    HUGGINGFACE_TOKEN: Optional[SecretStr] = Field(
        default=None, description="Token for private Hugging Face model repositories.")
    DATABASE_URL: Optional[SecretStr] = Field(
        default=None, description="Async PostgreSQL connection string (legacy alias).")
    POSTGRES_DSN: Optional[SecretStr] = Field(
        default=None, description="Async PostgreSQL connection string (canonical).")
    QDRANT_API_KEY: Optional[SecretStr] = Field(
        default=None, description="Optional API key for Qdrant Cloud deployments.")
    QDRANT_URL: str = Field(
        default="http://localhost:6333", description="Full Qdrant service URL.")

    # ── General settings ────────────────────────────────────────
    DEBUG: bool = Field(
        default=False, description="If true, FastAPI will expose detailed error traces.")
    PORT: int = Field(
        default=8000, ge=1, le=65535, description="Port on which the FastAPI server listens.")
    ENVIRONMENT: Literal["dev", "staging", "prod", "development", "production"] = Field(
        default="dev", description="Current runtime environment.")
    STARTUP_TIME: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the Settings instance was created.")

    # ── Vector store / embedding settings ────────────────────────
    QDRANT_HOST: str = Field(
        default="localhost", description="Hostname for Qdrant vector DB.")
    QDRANT_PORT: int = Field(
        default=6333, ge=1, le=65535, description="Port for Qdrant service.")
    COLPALI_MODEL_NAME: str = Field(
        default="vidore/colpali-v1.2",
        description="ColPali model identifier for multi‑modal embeddings.")

    @property
    def effective_database_url(self) -> str:
        """Return the PostgreSQL DSN from either legacy or canonical env var."""
        if self.POSTGRES_DSN is not None:
            value = self.POSTGRES_DSN.get_secret_value()
            if value:
                return value
        if self.DATABASE_URL is not None:
            return self.DATABASE_URL.get_secret_value()
        return ""


settings = Settings()

