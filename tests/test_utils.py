"""Tests for shared utilities."""

from datetime import timezone

from pythia_agent.utils import (
    MAX_TOOL_OUTPUT,
    get_required_env,
    parse_json_or_none,
    slugify,
    truncate,
    utc_now,
)


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_with_slashes():
    assert slugify("path/to/thing") == "path-to-thing"


def test_slugify_max_len():
    long_name = "a" * 100
    assert len(slugify(long_name)) == 40


def test_slugify_custom_max_len():
    assert slugify("hello world", max_len=5) == "hello"


def test_utc_now_has_timezone():
    now = utc_now()
    assert now.tzinfo == timezone.utc


def test_get_required_env_present(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "test_value")
    value, err = get_required_env("TEST_VAR")
    assert value == "test_value"
    assert err is None


def test_get_required_env_missing(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    value, err = get_required_env("MISSING_VAR")
    assert value is None
    assert "MISSING_VAR" in err


def test_parse_json_or_none_valid():
    assert parse_json_or_none('{"key": "value"}') == {"key": "value"}


def test_parse_json_or_none_invalid():
    assert parse_json_or_none("not json") is None


def test_parse_json_or_none_none_input():
    assert parse_json_or_none(None) is None


def test_truncate_short():
    assert truncate("hello", 10) == "hello"


def test_truncate_long():
    assert truncate("hello world", 5) == "hello"


def test_truncate_with_suffix():
    assert truncate("hello world", 5, suffix="...") == "hello..."


def test_max_tool_output_constant():
    assert MAX_TOOL_OUTPUT == 5000
