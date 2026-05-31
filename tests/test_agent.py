"""Tests for PythiaAgent."""

from unittest.mock import MagicMock, patch

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
    mock_strands.assert_called_once_with("Hello")
