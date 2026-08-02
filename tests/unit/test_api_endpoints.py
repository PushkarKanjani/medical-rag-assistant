from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app as legacy_app
from src.main import app as modern_app


class TestLegacyAPI:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(legacy_app)

    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "timestamp" in payload

    def test_query_endpoint_validation_error(self, client: TestClient) -> None:
        response = client.post("/api/v1/agent/query", json={"query": "only"})
        assert response.status_code == 422

    def test_query_endpoint_success(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/agent/query",
            json={"query": "What is the dosage for aspirin?", "session_id": "test-1"},
        )
        assert response.status_code == 200
        payload = response.json()
        for key in ("thought_process", "answer", "execution_time_ms"):
            assert key in payload
        assert isinstance(payload["execution_time_ms"], int)
        assert payload["execution_time_ms"] >= 0


class TestModernAPI:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(modern_app)

    def test_healthz_endpoint(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_chat_validation_error(self, client: TestClient) -> None:
        response = client.post("/v1/chat", json={"query": "hi"})
        assert response.status_code == 422

    def test_chat_success(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat",
            json={
                "query": "Amoxicillin dose for child",
                "user_id": "doc-1",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        for key in ("final_answer", "citations", "confidence_vector", "audit_id"):
            assert key in payload
        assert len(payload["final_answer"]) > 0
        assert payload["confidence_vector"]["local"] in (0.0, 1.0)
        assert payload["confidence_vector"]["web"] in (0.0, 1.0)

    def test_abha_create_validation_error(self, client: TestClient) -> None:
        response = client.post("/v1/abha/create", json={"aadhaar_number": "123"})
        assert response.status_code == 422

    def test_abha_login_validation_error(self, client: TestClient) -> None:
        response = client.post("/v1/abha/login", json={"abha_id": "x"})
        assert response.status_code == 422
