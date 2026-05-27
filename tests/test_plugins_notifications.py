"""Tests for NotificationPlugin."""

from unittest.mock import patch, MagicMock

import pytest

from pythia_agent.plugins.notifications import NotificationPlugin


@pytest.fixture
def notif_plugin():
    return NotificationPlugin()


def test_telegram_missing_env(notif_plugin, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    result = notif_plugin.notify_telegram(message="test")
    assert "TELEGRAM_BOT_TOKEN" in result


def test_telegram_success(notif_plugin, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("pythia_agent.plugins.notifications.requests.post", return_value=mock_resp) as mock_post:
        result = notif_plugin.notify_telegram(message="hello", title="Test")

    assert "sent" in result.lower()
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "fake-token" in call_kwargs[0][0]


def test_slack_missing_env(notif_plugin, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    result = notif_plugin.notify_slack(message="test")
    assert "SLACK_WEBHOOK_URL" in result


def test_slack_success(notif_plugin, monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("pythia_agent.plugins.notifications.requests.post", return_value=mock_resp):
        result = notif_plugin.notify_slack(message="hello")

    assert "sent" in result.lower()


def test_webhook_missing_url(notif_plugin, monkeypatch):
    monkeypatch.delenv("NOTIFICATION_WEBHOOK_URL", raising=False)
    result = notif_plugin.notify_webhook(message="test")
    assert "no webhook URL" in result.lower() or "NOTIFICATION_WEBHOOK_URL" in result


def test_webhook_explicit_url(notif_plugin):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200

    with patch("pythia_agent.plugins.notifications.requests.post", return_value=mock_resp) as mock_post:
        result = notif_plugin.notify_webhook(message="hello", url="https://example.com/hook", title="Alert")

    assert "sent" in result.lower()
    call_kwargs = mock_post.call_args
    assert call_kwargs[0][0] == "https://example.com/hook"
