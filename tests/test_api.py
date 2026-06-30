"""
Lightweight tests that don't require downloading embedding/LLM models,
so they run fast in CI. Full RAG behavior (sync + ask) is best verified
manually/locally since it depends on Ollama or OpenAI being available.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body
    assert "data_source" in body


def test_sync_rejects_unknown_source():
    response = client.post("/sync", json={"source": "carrier_pigeon"})
    assert response.status_code == 400


def test_ask_requires_question_field():
    response = client.post("/ask", json={})
    assert response.status_code == 422  # FastAPI validation error
