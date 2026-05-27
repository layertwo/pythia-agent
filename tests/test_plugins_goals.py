"""Tests for GoalsPlugin with mocked DB."""

from unittest.mock import MagicMock, patch

import pytest

from pythia_agent.plugins.goals import GoalsPlugin


@pytest.fixture
def goals_plugin():
    return GoalsPlugin()


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


def test_create_goal_success(goals_plugin, mock_session):
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()

    with patch("pythia_agent.plugins.goals.get_session", return_value=mock_session):
        result = goals_plugin.create_goal(name="Write posts", target=10, unit="posts", period="monthly")

    assert "Created goal" in result
    assert "10" in result
    assert "posts" in result


def test_create_goal_duplicate(goals_plugin, mock_session):
    from sqlalchemy.exc import IntegrityError

    mock_session.add = MagicMock()
    mock_session.commit = MagicMock(side_effect=IntegrityError("", "", Exception()))
    mock_session.rollback = MagicMock()

    with patch("pythia_agent.plugins.goals.get_session", return_value=mock_session):
        result = goals_plugin.create_goal(name="Write posts", target=10)

    assert "already exists" in result


def test_update_goal_not_found(goals_plugin, mock_session):
    mock_session.get = MagicMock(return_value=None)

    with patch("pythia_agent.plugins.goals.get_session", return_value=mock_session):
        result = goals_plugin.update_goal(goal_id="nonexistent")

    assert "not found" in result


def test_update_goal_increment(goals_plugin, mock_session):
    mock_goal = MagicMock()
    mock_goal.name = "Write posts"
    mock_goal.target = 10.0
    mock_goal.current = 3.0
    mock_goal.unit = "posts"
    mock_session.get = MagicMock(return_value=mock_goal)

    with patch("pythia_agent.plugins.goals.get_session", return_value=mock_session):
        result = goals_plugin.update_goal(goal_id="write-posts", increment=2)

    assert "5.0/10.0" in result or "5/10" in result


def test_delete_goal_not_found(goals_plugin, mock_session):
    mock_session.get = MagicMock(return_value=None)

    with patch("pythia_agent.plugins.goals.get_session", return_value=mock_session):
        result = goals_plugin.delete_goal(goal_id="missing")

    assert "not found" in result
