"""Shared utilities for pythia-agent plugins."""

import json
import os
from datetime import datetime, timezone
from typing import Any


def slugify(name: str, max_len: int = 40) -> str:
    """Convert a name to a URL-safe slug."""
    return name.lower().replace(" ", "-").replace("/", "-")[:max_len]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_required_env(var_name: str) -> tuple[str | None, str | None]:
    """Get an env var, returning (value, None) or (None, error_message)."""
    value = os.environ.get(var_name)
    if not value:
        return None, f"Error: {var_name} environment variable not set"
    return value, None


def parse_json_or_none(text: str) -> Any | None:
    """Parse JSON, returning None on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError, TypeError:
        return None


def truncate(text: str, max_len: int, suffix: str = "") -> str:
    """Truncate text to max_len, appending suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + suffix


MAX_TOOL_OUTPUT = 5000
MAX_HTTP_RESPONSE = 10000
