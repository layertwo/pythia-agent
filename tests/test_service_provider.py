"""Tests for ServiceProvider."""

from unittest.mock import MagicMock, patch

import pytest

from pythia_agent.config import (
    Settings,
    AgentConfig,
    ModelConfig,
    MemoryConfig,
    ServerConfig,
)
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
    # ContextPlugin removed in favor of the native conversation_manager; 9 remain.
    assert len(plugins) == 9


def test_context_plugin_not_in_shared_plugins(provider):
    """ContextPlugin is replaced by the native conversation manager and must not be registered."""
    with patch("pythia_agent.environment.service_provider.create_model"):
        plugins = provider.shared_plugins
    assert not any(type(p).__name__ == "ContextPlugin" for p in plugins)


def test_conversation_manager_maps_config_knobs(provider):
    """The provider builds a SummarizingConversationManager mapping the old protect_first/last knobs."""
    from strands.agent.conversation_manager import SummarizingConversationManager

    cm = provider.create_conversation_manager()
    assert isinstance(cm, SummarizingConversationManager)
    # pin_first <- protect_first_n (3), preserve_recent_messages <- protect_last_n (6)
    assert cm.pin_first == provider.settings.context.pin_first
    assert cm.preserve_recent_messages == provider.settings.context.preserve_recent_messages


def test_conversation_manager_none_when_disabled(settings):
    settings.context.enabled = False
    p = ServiceProvider(settings=settings)
    assert p.create_conversation_manager() is None


def test_conversation_manager_is_fresh_per_call(provider):
    """A new manager is built per call — it holds per-conversation state and is
    registered as a per-agent hook, so it must never be a shared singleton."""
    first = provider.create_conversation_manager()
    second = provider.create_conversation_manager()
    assert first is not second


def test_limits_built_from_config(provider):
    """Only non-None limit fields are included in the dict handed to the agent."""
    limits = provider.limits
    assert limits == {"turns": provider.settings.limits.turns}


def test_limits_none_when_disabled(settings):
    settings.limits.enabled = False
    p = ServiceProvider(settings=settings)
    assert p.limits is None


def test_create_agent_wires_context_and_limits(provider):
    with patch("pythia_agent.environment.service_provider.create_model") as mock_model:
        mock_model.return_value = MagicMock()
        with patch("pythia_agent.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            agent = provider.create_agent(user_id="testuser")

    assert agent._limits == {"turns": provider.settings.limits.turns}
    call_kwargs = mock_agent_cls.call_args[1]
    assert "conversation_manager" in call_kwargs


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
