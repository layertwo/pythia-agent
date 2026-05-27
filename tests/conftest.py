"""Shared test fixtures."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pythia_agent.config import (
    AgentConfig,
    BedrockConfig,
    MemoryConfig,
    MemoryEmbedderConfig,
    MemoryLLMConfig,
    MemoryVectorStoreConfig,
    ModelConfig,
    OllamaConfig,
    OpenAIConfig,
    AnthropicConfig,
    ServerConfig,
    Settings,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        agent=AgentConfig(name="TestAgent", system_prompt="You are a test agent."),
        model=ModelConfig(provider="ollama", model_id="test-model"),
        ollama=OllamaConfig(host="http://localhost:11434"),
        openai=OpenAIConfig(model_id="gpt-4o"),
        anthropic=AnthropicConfig(model_id="claude-sonnet-4-20250514"),
        bedrock=BedrockConfig(model_id="test", region="us-east-1"),
        memory=MemoryConfig(
            enabled=False,
            auto_inject=True,
            auto_inject_top_k=3,
            auto_store=True,
            llm=MemoryLLMConfig(),
            embedder=MemoryEmbedderConfig(),
            vector_store=MemoryVectorStoreConfig(),
        ),
        server=ServerConfig(host="0.0.0.0", port=8080),
    )


@pytest.fixture
def mock_model():
    model = MagicMock()
    return model


@pytest.fixture
def mock_agent_result():
    result = MagicMock()
    result.message = "Test response from agent"
    return result


@pytest.fixture
def mock_strands_agent(mock_agent_result):
    with patch("pythia_agent.agent.Agent") as mock_cls:
        instance = MagicMock()
        instance.return_value = mock_agent_result
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def test_client():
    """Create a test client with mocked dependencies."""
    with patch("pythia_agent.server._provider") as mock_provider:
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

        client = TestClient(app)
        yield client
