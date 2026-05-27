import logging

from strands import Agent

from pythia_agent.config import Settings
from pythia_agent.memory import Mem0SessionManager

logger = logging.getLogger(__name__)


class PythiaAgent:
    """Strands agent assembled from injected dependencies.

    Does not know about ServiceProvider — receives a model, plugins, and
    optional session manager directly.
    """

    def __init__(
        self,
        model,
        system_prompt: str,
        plugins: list,
        session_manager: Mem0SessionManager | None = None,
    ):
        self.session_manager = session_manager
        tools = session_manager.get_tools() if session_manager else []

        self.agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            plugins=plugins,
            session_manager=session_manager,
        )

    def invoke(self, message: str) -> dict:
        result = self.agent(message)
        return {
            "response": str(result.message),
        }
