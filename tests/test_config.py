"""Tests for config loading."""

import tempfile
from pathlib import Path

from pythia_agent.config import load_settings


def test_load_settings_defaults():
    settings = load_settings(config_path=Path("/nonexistent/path.yaml"))
    assert settings.agent.name == "Pythia"
    assert settings.model.provider == "ollama"
    assert settings.server.port == 8080


def test_load_settings_from_yaml():
    yaml_content = """
agent:
  name: "CustomAgent"
  system_prompt: "Custom prompt"
model:
  provider: "openai"
  model_id: "gpt-4o"
server:
  port: 9090
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        settings = load_settings(config_path=f.name)

    assert settings.agent.name == "CustomAgent"
    assert settings.model.provider == "openai"
    assert settings.model.model_id == "gpt-4o"
    assert settings.server.port == 9090


def test_memory_config_defaults():
    settings = load_settings(config_path=Path("/nonexistent"))
    assert settings.memory.enabled is True
    assert settings.memory.auto_inject is True
    assert settings.memory.auto_inject_top_k == 5
    assert settings.memory.vector_store.provider == "pgvector"
