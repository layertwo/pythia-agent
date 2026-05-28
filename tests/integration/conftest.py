"""Fixtures for integration tests.

These tests require the docker compose stack at tests/integration/compose.yaml
to be running. They are marked @pytest.mark.integration and deselected from
the default pytest run via pyproject.toml's `-m 'not integration'` addopt.
"""

import os

import pytest


@pytest.fixture(scope="session")
def base_url() -> str:
    """URL where the pythia container is exposed. Override with PYTHIA_BASE_URL."""
    return os.environ.get("PYTHIA_BASE_URL", "http://localhost:8080")


# Module-level holder shared across tests in test_smoke.py.
# Test 3 (memory_readback) populates it; Tests 4 and 5 read it.
# Pytest runs tests within a module in declaration order, which is what we rely on.
_state: dict[str, str] = {}


@pytest.fixture(scope="session")
def memory_state() -> dict[str, str]:
    return _state
