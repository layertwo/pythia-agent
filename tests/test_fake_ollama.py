"""Unit tests for the fake-Ollama HTTP stub used by integration tests."""

import json

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


def test_chat_fact_extraction_returns_memory_json():
    """When the prompt looks like a mem0 2.0.4 extraction prompt, return memory JSON."""
    payload = {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": "You are a Memory Extractor. Return JSON."},
            {"role": "user", "content": "## New Messages\nUser: my favorite color is blue\n# Output:"},
        ],
        "stream": False,
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    content = resp.json()["message"]["content"]
    parsed = json.loads(content)
    # mem0 2.0.4 expects {"memory": [{"id": "...", "text": "...", "event": "..."}]}
    assert "memory" in parsed
    assert isinstance(parsed["memory"], list)
    assert len(parsed["memory"]) >= 1
    assert "text" in parsed["memory"][0]
    assert "event" in parsed["memory"][0]


def test_chat_streaming_returns_ndjson():
    payload = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    # Response body is newline-delimited JSON; parse each non-empty line.
    lines = [line for line in resp.text.splitlines() if line.strip()]
    assert len(lines) >= 1
    parsed = [json.loads(line) for line in lines]
    # Last chunk must have done=True.
    assert parsed[-1]["done"] is True
    # All chunks must have a message dict with role.
    assert all("message" in p for p in parsed)
