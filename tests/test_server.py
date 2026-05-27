"""Tests for the FastAPI server endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with fully mocked provider."""
    with patch("pythia_agent.server._provider") as mock_provider:
        from pythia_agent.config import AgentConfig, ModelConfig, MemoryConfig, ServerConfig, Settings

        mock_provider.settings = Settings(
            agent=AgentConfig(name="TestAgent", system_prompt="Test."),
            model=ModelConfig(provider="ollama", model_id="test"),
            memory=MemoryConfig(enabled=False),
            server=ServerConfig(),
        )
        mock_provider.start = MagicMock()
        mock_provider.shutdown = MagicMock()

        mock_agent = MagicMock()
        mock_agent.session_manager = None
        mock_agent.invoke.return_value = {"response": "Hello from test"}
        mock_provider.create_agent.return_value = mock_agent

        from pythia_agent.server import app

        yield TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["agent"] == "TestAgent"
    assert data["model_provider"] == "ollama"


def test_chat_endpoint(client):
    resp = client.post("/chat", json={"prompt": "hi", "user_id": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Hello from test"
    assert data["user_id"] == "test"
    assert "timestamp" in data


def test_chat_default_user(client):
    resp = client.post("/chat", json={"prompt": "hello"})
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "default"


def test_memory_search_disabled(client):
    resp = client.post("/memory/search", json={"query": "test", "user_id": "u1"})
    assert resp.status_code == 503


def test_memory_get_disabled(client):
    resp = client.get("/memory/testuser")
    assert resp.status_code == 503
