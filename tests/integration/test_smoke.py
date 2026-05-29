"""Integration smoke tests against a running docker compose stack.

Requires `docker compose -f tests/integration/compose.yaml up --wait` to have
been run first. Tests share a single user_id and run sequentially.
"""

import time

import httpx
import pytest

pytestmark = pytest.mark.integration

USER_ID = "integration-test"
REMEMBER_PROMPT = "remember that my favorite color is blue"


def test_health_endpoint(base_url):
    resp = httpx.get(f"{base_url}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["agent"] == "Pythia"
    assert data["model_provider"] == "ollama"
    assert data["memory_enabled"] is True


def test_chat_stores_memory(base_url):
    resp = httpx.post(
        f"{base_url}/chat",
        json={"prompt": REMEMBER_PROMPT, "user_id": USER_ID},
        timeout=30,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == USER_ID
    assert isinstance(data["response"], str)
    # Give mem0 a moment to flush the write to pgvector before the next test reads it.
    time.sleep(1)


def test_memory_readback(base_url, memory_state):
    resp = httpx.get(f"{base_url}/memory/{USER_ID}", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == USER_ID
    assert isinstance(data["results"], list)
    assert len(data["results"]) >= 1, "expected at least one stored memory"
    memory_state["memory_id"] = data["results"][0]["id"]


def test_memory_search(base_url):
    resp = httpx.post(
        f"{base_url}/memory/search",
        json={"query": REMEMBER_PROMPT, "user_id": USER_ID, "top_k": 5},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == USER_ID
    assert isinstance(data["results"], list)
    assert len(data["results"]) >= 1, "expected search to return the stored memory"


def test_memory_delete(base_url, memory_state):
    memory_id = memory_state.get("memory_id")
    assert memory_id, "test_memory_readback must run before this and populate memory_state"

    resp = httpx.delete(f"{base_url}/memory/{USER_ID}/{memory_id}", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    assert data["memory_id"] == memory_id

    # Pre-existing bug: GET /memory/{user}/{id} returns 200 even when missing.
    # We assert the `result` field is empty/None rather than expecting 404.
    follow = httpx.get(f"{base_url}/memory/{USER_ID}/{memory_id}", timeout=10)
    assert follow.status_code == 200
    assert not follow.json().get("result")
