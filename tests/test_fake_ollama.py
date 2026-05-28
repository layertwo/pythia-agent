"""Unit tests for the fake-Ollama HTTP stub used by integration tests."""

from fastapi.testclient import TestClient

from tests.integration.fake_ollama import app, EMBEDDING_DIMS


client = TestClient(app)


def test_embeddings_returns_fixed_dim_vector():
    resp = client.post("/api/embeddings", json={"model": "nomic-embed-text", "prompt": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) == EMBEDDING_DIMS
    assert all(isinstance(x, float) for x in data["embedding"])


def test_embeddings_deterministic_by_input():
    """Identical input must produce the identical vector (cosine similarity = 1.0)."""
    r1 = client.post("/api/embeddings", json={"model": "nomic-embed-text", "prompt": "hello"})
    r2 = client.post("/api/embeddings", json={"model": "nomic-embed-text", "prompt": "hello"})
    assert r1.json()["embedding"] == r2.json()["embedding"]


def test_embeddings_differs_by_input():
    """Different input must produce a different vector."""
    r1 = client.post("/api/embeddings", json={"model": "nomic-embed-text", "prompt": "hello"})
    r2 = client.post("/api/embeddings", json={"model": "nomic-embed-text", "prompt": "world"})
    assert r1.json()["embedding"] != r2.json()["embedding"]


def test_chat_returns_canned_assistant_message():
    payload = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"]["role"] == "assistant"
    assert isinstance(data["message"]["content"], str)
    assert data["done"] is True


def test_chat_fact_extraction_returns_json_facts():
    """When the prompt looks like a mem0 fact-extraction prompt, return JSON."""
    payload = {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": "You are a memory extraction assistant. Return JSON."},
            {"role": "user", "content": "Extract facts from: my favorite color is blue"},
        ],
        "stream": False,
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    import json
    content = resp.json()["message"]["content"]
    parsed = json.loads(content)
    assert "facts" in parsed
    assert isinstance(parsed["facts"], list)
    assert len(parsed["facts"]) >= 1


def test_chat_streaming_not_supported():
    payload = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    resp = client.post("/api/chat", json=payload)
    # Streaming returns 400 — we don't support it in the stub
    assert resp.status_code == 400
