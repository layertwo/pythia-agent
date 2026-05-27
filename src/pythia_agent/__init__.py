"""Pythia Agent - Self-hosted Strands agent with mem0 persistent memory."""

__version__ = "0.1.0"

from pythia_agent.agent import PythiaAgent
from pythia_agent.config import Settings, load_settings
from pythia_agent.environment import ServiceProvider
from pythia_agent.memory import Mem0SessionManager

__all__ = [
    "PythiaAgent",
    "Settings",
    "load_settings",
    "ServiceProvider",
    "Mem0SessionManager",
]
