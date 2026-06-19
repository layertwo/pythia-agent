"""Tests for PythiaAgent."""

from unittest.mock import MagicMock, patch

import pytest

from pythia_agent.agent import PythiaAgent


def test_agent_init_without_memory(mock_model):
    with patch("pythia_agent.agent.Agent") as mock_agent_cls:
        mock_agent_cls.return_value = MagicMock()

        agent = PythiaAgent(
            model=mock_model,
            system_prompt="Test prompt",
            plugins=[],
            session_manager=None,
        )

        assert agent.session_manager is None
        mock_agent_cls.assert_called_once()
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["system_prompt"] == "Test prompt"
        assert call_kwargs["tools"] == []
        assert call_kwargs["session_manager"] is None


def test_agent_init_with_memory(mock_model):
    mock_session_manager = MagicMock()
    mock_session_manager.get_tools.return_value = ["tool1", "tool2"]

    with patch("pythia_agent.agent.Agent") as mock_agent_cls:
        mock_agent_cls.return_value = MagicMock()

        agent = PythiaAgent(
            model=mock_model,
            system_prompt="Test",
            plugins=[],
            session_manager=mock_session_manager,
        )

        assert agent.session_manager is mock_session_manager
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["tools"] == ["tool1", "tool2"]


def test_agent_invoke(mock_model):
    mock_result = MagicMock()
    # Strands returns message as a dict with content blocks; invoke()
    # joins text from any blocks that have a "text" key.
    mock_result.message = {
        "role": "assistant",
        "content": [{"text": "Agent says hello"}],
    }

    mock_strands = MagicMock()
    mock_strands.return_value = mock_result

    with patch("pythia_agent.agent.Agent", return_value=mock_strands):
        agent = PythiaAgent(
            model=mock_model,
            system_prompt="Test",
            plugins=[],
            session_manager=None,
        )
        result = agent.invoke("Hello")

    assert result["response"] == "Agent says hello"
    # No limits configured -> invocation called positionally with no limits kwarg.
    mock_strands.assert_called_once_with("Hello")


def test_agent_passes_conversation_manager(mock_model):
    """The conversation_manager (native context window management) is forwarded to Agent."""
    sentinel_cm = object()
    with patch("pythia_agent.agent.Agent") as mock_agent_cls:
        mock_agent_cls.return_value = MagicMock()
        PythiaAgent(
            model=mock_model,
            system_prompt="Test",
            plugins=[],
            session_manager=None,
            conversation_manager=sentinel_cm,
        )

    call_kwargs = mock_agent_cls.call_args[1]
    assert call_kwargs["conversation_manager"] is sentinel_cm


def test_agent_passes_none_conversation_manager_by_default(mock_model):
    """With no conversation_manager, None is passed through (Strands' own default)."""
    with patch("pythia_agent.agent.Agent") as mock_agent_cls:
        mock_agent_cls.return_value = MagicMock()
        PythiaAgent(
            model=mock_model,
            system_prompt="Test",
            plugins=[],
            session_manager=None,
        )

    call_kwargs = mock_agent_cls.call_args[1]
    assert call_kwargs["conversation_manager"] is None


def test_agent_invoke_passes_limits(mock_model):
    """A configured limits dict is forwarded as a keyword arg to the Strands invocation."""
    mock_result = MagicMock()
    mock_result.message = {"role": "assistant", "content": [{"text": "ok"}]}
    mock_strands = MagicMock(return_value=mock_result)

    with patch("pythia_agent.agent.Agent", return_value=mock_strands):
        agent = PythiaAgent(
            model=mock_model,
            system_prompt="Test",
            plugins=[],
            session_manager=None,
            limits={"turns": 50},
        )
        agent.invoke("Hello")

    mock_strands.assert_called_once_with("Hello", limits={"turns": 50})


def test_agent_invoke_omits_limits_when_none(mock_model):
    """With no limits, invoke() must not pass a limits kwarg at all."""
    mock_result = MagicMock()
    mock_result.message = {"role": "assistant", "content": [{"text": "ok"}]}
    mock_strands = MagicMock(return_value=mock_result)

    with patch("pythia_agent.agent.Agent", return_value=mock_strands):
        agent = PythiaAgent(
            model=mock_model,
            system_prompt="Test",
            plugins=[],
            session_manager=None,
        )
        agent.invoke("Hello")

    mock_strands.assert_called_once_with("Hello")


@pytest.mark.asyncio
async def test_agent_stream_passes_limits(mock_model):
    """The streaming path forwards a configured limits dict to stream_async."""

    async def fake_stream(*args, **kwargs):
        # Record the call, then yield one text delta.
        fake_stream.calls.append((args, kwargs))
        yield {"data": "hi"}

    fake_stream.calls = []

    mock_strands = MagicMock()
    mock_strands.stream_async = fake_stream

    with patch("pythia_agent.agent.Agent", return_value=mock_strands):
        agent = PythiaAgent(
            model=mock_model,
            system_prompt="Test",
            plugins=[],
            session_manager=None,
            limits={"turns": 50},
        )
        chunks = [c async for c in agent.stream("Hello")]

    assert chunks == ["hi"]
    args, kwargs = fake_stream.calls[0]
    assert args == ("Hello",)
    assert kwargs == {"limits": {"turns": 50}}
