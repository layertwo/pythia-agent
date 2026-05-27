"""Notification plugin: multi-channel alerts (Telegram, Slack webhook, generic webhook)."""

import logging

import requests

from strands import tool
from strands.plugins import Plugin

from pythia_agent.utils import get_required_env, utc_now

logger = logging.getLogger(__name__)


GUIDANCE = (
    "\n\nYou can send notifications via Telegram or webhooks. "
    "Use these to alert the user when scheduled jobs produce notable results."
)


class NotificationPlugin(Plugin):
    """Provides notification tools for sending alerts via Telegram or webhooks."""

    name = "notifications"

    def __init__(self):
        super().__init__()

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    @tool
    def notify_telegram(self, message: str, title: str = "") -> str:
        """Send a notification via Telegram.

        Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.

        Args:
            message: The message body to send
            title: Optional title/heading for the message
        """
        bot_token, err = get_required_env("TELEGRAM_BOT_TOKEN")
        if err:
            return err
        chat_id, err = get_required_env("TELEGRAM_CHAT_ID")
        if err:
            return err

        text = f"*{title}*\n{message}" if title else message

        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            resp.raise_for_status()
            return "Telegram notification sent."
        except Exception as e:
            return f"Error sending Telegram notification: {e}"

    @tool
    def notify_webhook(self, message: str, url: str = "", title: str = "") -> str:
        """Send a notification via a generic webhook (POST with JSON body).

        Uses NOTIFICATION_WEBHOOK_URL env var if url not provided.

        Args:
            message: The message body
            url: Webhook URL to POST to (falls back to NOTIFICATION_WEBHOOK_URL env var)
            title: Optional title
        """
        if not url:
            url, err = get_required_env("NOTIFICATION_WEBHOOK_URL")
            if err:
                return "Error: no webhook URL provided and NOTIFICATION_WEBHOOK_URL not set"

        payload = {"title": title, "message": message, "timestamp": utc_now().isoformat()}

        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            return f"Webhook notification sent (status {resp.status_code})."
        except Exception as e:
            return f"Error sending webhook notification: {e}"
