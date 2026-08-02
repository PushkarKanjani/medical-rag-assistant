from __future__ import annotations

import pytest

from src.observability.audit_log import append_audit


class DummyConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def execute(self, query: str, *args: object) -> None:
        self.executed.append((query, args))

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_append_audit_uses_asyncpg_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_connection = DummyConnection()
    connect_calls: list[str] = []

    async def fake_connect(dsn: str, **kwargs: object) -> DummyConnection:
        connect_calls.append(dsn)
        return dummy_connection

    class DummySettings:
        postgres_dsn = "postgresql://example"

    monkeypatch.setattr("src.observability.audit_log.asyncpg.connect", fake_connect)
    monkeypatch.setattr("src.observability.audit_log.AppSettings", DummySettings)

    await append_audit(
        user_id="user-1",
        node="critic",
        status="ok",
        latency_ms=12,
        error=None,
        details={"source": "unit-test"},
    )

    assert connect_calls == ["postgresql://example"]
    assert dummy_connection.closed is True
    assert len(dummy_connection.executed) == 1
    query, args = dummy_connection.executed[0]
    assert "INSERT INTO audit_log" in query
    assert args == (
        "user-1",
        "critic",
        "ok",
        12,
        None,
        {"source": "unit-test"},
    )