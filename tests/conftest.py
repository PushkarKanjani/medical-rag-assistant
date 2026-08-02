from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "dev")
os.environ.setdefault("POSTGRES_DSN", "postgresql://example")
os.environ.setdefault("QDRANT_URL", "")
os.environ.setdefault("GROQ_API_KEY", "")


def pytest_configure() -> None:
    """Configure pytest-wide fixtures and settings for the agentic-medassist test suite."""
