"""Service provider — lazy dependency graph for pythia-agent.

The composition root. Only the server entrypoint touches this class.
Components receive their actual dependencies directly — no service
locator pattern leaking into business logic.
"""

import os
from functools import cached_property
from strands.agent.conversation_manager import SummarizingConversationManager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pythia_agent.agent import PythiaAgent
from pythia_agent.dreams import DreamEngine
from pythia_agent.config import Settings, load_settings
from pythia_agent.db import Base
from pythia_agent.memory import Mem0SessionManager
from pythia_agent.plugins.agent_tools import AgentToolsPlugin
from pythia_agent.plugins.dreams import DreamsPlugin
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
    def dreams_engine(self) -> DreamEngine:
        return DreamEngine(
            dreams_config=self.settings.dreams,
            memory_config=self.settings.memory,
            model_settings=self.settings,
        )

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
        ]

    # ------------------------------------------------------------------
    # Context window management (native, replaces the old ContextPlugin)
    # ------------------------------------------------------------------

    def create_conversation_manager(self):
        """Build a fresh Strands conversation manager for context-window compression.

        Maps the old ContextPlugin knobs onto SummarizingConversationManager:
        pin_first <- protect_first_n, preserve_recent_messages <- protect_last_n.
        Returns None when disabled so the agent falls back to Strands' default.

        NOT cached: the manager holds per-conversation state (summary message,
        removed-message count) and is registered as a hook on its agent, so each
        per-user agent needs its own instance to avoid cross-user state bleed.
        """
        ctx = self.settings.context
        if not ctx.enabled:
            return None

        return SummarizingConversationManager(
            summary_ratio=ctx.summary_ratio,
            preserve_recent_messages=ctx.preserve_recent_messages,
            pin_first=ctx.pin_first,
        )

    @cached_property
    def limits(self) -> dict | None:
        """Per-invocation hard backstop dict, or None when disabled.

        Safe to cache: a plain immutable dict shared read-only across agents.
        """
        return self.settings.limits.to_limits()

    # ------------------------------------------------------------------
    # Per-user factories
    # ------------------------------------------------------------------

    def create_session_manager(self, user_id: str) -> Mem0SessionManager | None:
        if not self.settings.memory.enabled:
            return None
        return Mem0SessionManager(self.settings.memory, user_id=user_id)

    def create_sessions_plugin(self, user_id: str) -> SessionsPlugin:
        return SessionsPlugin(user_id=user_id)

    def create_dreams_plugin(self, user_id: str) -> DreamsPlugin | None:
        if not self.settings.memory.enabled:
            return None
        return DreamsPlugin(engine=self.dreams_engine, user_id=user_id)

    def create_agent(self, user_id: str = "default") -> PythiaAgent:
        per_user = [self.create_sessions_plugin(user_id)]
        dreams_plugin = self.create_dreams_plugin(user_id)
        if dreams_plugin is not None:
            per_user.append(dreams_plugin)
        return PythiaAgent(
            model=self.model,
            system_prompt=self.settings.agent.system_prompt,
            plugins=[*self.shared_plugins, *per_user],
            session_manager=self.create_session_manager(user_id),
            conversation_manager=self.create_conversation_manager(),
            limits=self.limits,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def start(self) -> None:
        self.init_schema()
        self.scheduler_plugin.start()
        if self.settings.memory.enabled:
            self.dreams_engine.start_cron()

    def shutdown(self) -> None:
        self.scheduler_plugin.stop()
        # Always drain the dreams engine if it was constructed at any point
        # during this provider's lifetime — `cached_property` won't have
        # built it if no caller touched `dreams_engine`, in which case
        # __dict__ won't contain it and we have nothing to drain. This is
        # also robust to `memory.enabled` toggling between start and
        # shutdown: if cron started, it must be stopped.
        if "dreams_engine" in self.__dict__:
            self.dreams_engine.shutdown()
        self.engine.dispose()

    def _handle_job(self, job) -> str:
        agent = self.create_agent(user_id="scheduler")
        prompt = job["prompt"] if isinstance(job, dict) else job.prompt
        return agent.invoke(prompt)["response"]
