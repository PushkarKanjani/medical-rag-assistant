from __future__ import annotations

import asyncpg
import structlog

from src.settings import AppSettings

log = structlog.get_logger(__name__)


async def append_audit(
    user_id: str,
    node: str,
    status: str,
    latency_ms: int = 0,
    error: str | None = None,
    details: dict | None = None,
) -> None:
    settings = AppSettings()
    log.info(
        "audit.append",
        user_id=user_id,
        node=node,
        status=status,
        latency_ms=latency_ms,
        error=error,
    )

    try:
        connection = await asyncpg.connect(settings.postgres_dsn, timeout=3.0)
        try:
            await connection.execute(
                """
                INSERT INTO audit_log (user_id, node, status, latency_ms, error, details)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id,
                node,
                status,
                latency_ms,
                error,
                details,
            )
        finally:
            await connection.close()
    except Exception as e:
        log.warning("audit.database_error", error=str(e))