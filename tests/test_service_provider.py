"""Tests for ServiceProvider."""

from unittest.mock import MagicMock, patch

import pytest

from pythia_agent.config import Settings, AgentConfig, ModelConfig, MemoryConfig, ServerConfig
from pythia_agent.environment.service_provider import ServiceProvider


@pytest.fixture
def provider(settings):
    return ServiceProvider(settings=settings)


def test_settings_loaded(provider, settings):
    assert provider.settings is settings
    assert provider.settings.agent.name == "TestAgent"


def test_settings_default_when_none():
    with patch("pythia_agent.environment.service_provider.load_settings") as mock_load:
        mock_load.return_value = Settings(
            agent=AgentConfig(name="Default"),
            model=ModelConfig(),
            memory=MemoryConfig(enabled=False),
            server=ServerConfig(),
        )
        p = ServiceProvider(settings=None)
        assert p.settings.agent.name == "Default"


def test_database_url_from_env(provider, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@db:5432/test")
    p = ServiceProvider(settings=provider.settings)
    assert p.database_url == "postgresql://test:test@db:5432/test"


def test_shared_plugins_count(provider):
    with patch("pythia_agent.environment.service_provider.create_model"):
        plugins = provider.shared_plugins
    assert len(plugins) == 10


def test_create_agent_without_memory(provider):
    with patch("pythia_agent.environment.service_provider.create_model") as mock_model:
        mock_model.return_value = MagicMock()
        with patch("pythia_agent.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            agent = provider.create_agent(user_id="testuser")

    assert agent.session_manager is None


def test_create_session_manager_disabled(provider):
    assert provider.settings.memory.enabled is False
    sm = provider.create_session_manager(user_id="test")
    assert sm is None
