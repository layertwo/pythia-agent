"""Service provider — lazy dependency graph for pythia-agent.

The composition root. Only the server entrypoint touches this class.
Components receive their actual dependencies directly — no service
locator pattern leaking into business logic.
"""

import os
from functools import cached_property

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pythia_agent.agent import PythiaAgent
from pythia_agent.config import Settings, load_settings
from pythia_agent.db import Base
from pythia_agent.memory import Mem0SessionManager
from pythia_agent.plugins.agent_tools import AgentToolsPlugin
from pythia_agent.plugins.context import ContextPlugin
from pythia_agent.plugins.goals import GoalsPlugin
from pythia_agent.plugins.notifications import NotificationPlugin
from pythia_agent.plugins.personas import PersonasPlugin
from pythia_agent.plugins.safety import SafetyPlugin
from pythia_agent.plugins.scheduler import SchedulerPlugin
from pythia_agent.plugins.sessions import SessionsPlugin
from pythia_agent.plugins.system_tools import SystemToolsPlugin
from pythia_agent.plugins.tasks import TasksPlugin
from pythia_agent.plugins.web_tools import WebToolsPlugin
from pythia_agent.providers.factory import create_model
import pythia_agent.models.session  # noqa: F401


class ServiceProvider:

    def __init__(self, settings: Settings | None = None):
        self._settings = settings

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @cached_property
    def settings(self) -> Settings:
        return self._settings or load_settings()

    @cached_property
    def database_url(self) -> str:
        return os.environ.get("DATABASE_URL", "postgresql://pythia:pythia@localhost:5432/pythia")

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    @cached_property
    def engine(self):
        return create_engine(self.database_url, pool_size=5, max_overflow=10, pool_pre_ping=True)

    @cached_property
    def session_factory(self):
        return sessionmaker(bind=self.engine)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    @cached_property
    def model(self):
        return create_model(self.settings)

    # ------------------------------------------------------------------
    # Plugins (shared singletons)
    # ------------------------------------------------------------------

    @cached_property
    def system_tools_plugin(self):
        return SystemToolsPlugin()

    @cached_property
    def web_tools_plugin(self):
        return WebToolsPlugin()

    @cached_property
    def agent_tools_plugin(self):
        return AgentToolsPlugin()

    @cached_property
    def scheduler_plugin(self):
        return SchedulerPlugin(on_job_fire=self._handle_job)

    @cached_property
    def notification_plugin(self):
        return NotificationPlugin()

    @cached_property
    def goals_plugin(self):
        return GoalsPlugin()

    @cached_property
    def personas_plugin(self):
        return PersonasPlugin()

    @cached_property
    def tasks_plugin(self):
        return TasksPlugin()

    @cached_property
    def safety_plugin(self):
        return SafetyPlugin()

    @cached_property
    def context_plugin(self):
        return ContextPlugin()

    @cached_property
    def shared_plugins(self) -> list:
        return [
            self.system_tools_plugin,
            self.web_tools_plugin,
            self.agent_tools_plugin,
            self.scheduler_plugin,
            self.notification_plugin,
            self.goals_plugin,
            self.personas_plugin,
            self.tasks_plugin,
            self.safety_plugin,
            self.context_plugin,
        ]

    # ------------------------------------------------------------------
    # Per-user factories
    # ------------------------------------------------------------------

    def create_session_manager(self, user_id: str) -> Mem0SessionManager | None:
        if not self.settings.memory.enabled:
            return None
        return Mem0SessionManager(self.settings.memory, user_id=user_id)

    def create_sessions_plugin(self, user_id: str) -> SessionsPlugin:
        return SessionsPlugin(user_id=user_id)

    def create_agent(self, user_id: str = "default") -> PythiaAgent:
        return PythiaAgent(
            model=self.model,
            system_prompt=self.settings.agent.system_prompt,
            plugins=[*self.shared_plugins, self.create_sessions_plugin(user_id)],
            session_manager=self.create_session_manager(user_id),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def start(self) -> None:
        self.init_schema()
        self.scheduler_plugin.start()

    def shutdown(self) -> None:
        self.scheduler_plugin.stop()
        self.engine.dispose()

    def _handle_job(self, job) -> str:
        agent = self.create_agent(user_id="scheduler")
        prompt = job["prompt"] if isinstance(job, dict) else job.prompt
        return agent.invoke(prompt)["response"]
