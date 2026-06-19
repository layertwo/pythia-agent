"""Scheduler plugin: cron-based recurring jobs backed by PostgreSQL via SQLAlchemy."""

import logging
import threading
import time
from datetime import datetime
from typing import Callable

from croniter import croniter
from sqlalchemy.exc import IntegrityError

from strands import tool
from strands.plugins import Plugin

from pythia_agent.db import Job, JobRun, get_session
from pythia_agent.utils import slugify, utc_now, MAX_TOOL_OUTPUT

logger = logging.getLogger(__name__)


GUIDANCE = (
    "\n\nYou can schedule autonomous recurring work with cron expressions. "
    "Jobs run in the background on their schedule and persist across restarts."
)


class SchedulerPlugin(Plugin):
    """Cron-based job scheduler with PostgreSQL persistence."""

    name = "scheduler"

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    def __init__(self, on_job_fire: Callable | None = None):
        self._on_job_fire = on_job_fire
        self._running = False
        self._thread: threading.Thread | None = None
        super().__init__()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def _run_loop(self) -> None:
        last_fire: dict[str, datetime] = {}
        while self._running:
            try:
                now = utc_now()
                with get_session() as session:
                    jobs = session.query(Job).filter(Job.enabled.is_(True)).all()  # noqa: E712
                    for job in jobs:
                        base = last_fire.get(job.id, now)
                        cron = croniter(job.cron, base)
                        next_fire = cron.get_next(datetime)
                        if next_fire <= now:
                            last_fire[job.id] = now
                            self._execute_job(job.id, job.name, job.prompt)
            except Exception as e:
                logger.error("Scheduler loop error: %s", e)
            time.sleep(30)

    def _execute_job(self, job_id: str, name: str, prompt: str) -> None:
        logger.info("Firing job: %s (%s)", name, job_id)
        started = utc_now()

        with get_session() as session:
            run = JobRun(job_id=job_id, started_at=started, status="running")
            session.add(run)
            session.commit()
            run_id = run.id

        try:
            output = (
                self._on_job_fire({"id": job_id, "name": name, "prompt": prompt})
                if self._on_job_fire
                else "(no handler)"
            )
            with get_session() as session:
                run = session.get(JobRun, run_id)
                run.status = "completed"
                run.completed_at = utc_now()
                run.output = output[:MAX_TOOL_OUTPUT]
                session.commit()
        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e)
            with get_session() as session:
                run = session.get(JobRun, run_id)
                run.status = "failed"
                run.completed_at = utc_now()
                run.output = str(e)[:MAX_TOOL_OUTPUT]
                session.commit()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool
    def create_job(self, name: str, cron: str, prompt: str) -> str:
        """Create a new scheduled job that runs on a cron schedule.

        Args:
            name: Human-readable name for the job
            cron: Cron expression (e.g., '0 9 * * *' for daily at 9am UTC)
            prompt: The prompt to send to the agent when the job fires
        """
        if not croniter.is_valid(cron):
            return f"Error: invalid cron expression '{cron}'"

        job_id = slugify(name)
        with get_session() as session:
            try:
                session.add(Job(id=job_id, name=name, cron=cron, prompt=prompt))
                session.commit()
            except IntegrityError:
                session.rollback()
                return f"Error: job '{job_id}' already exists"
        return f"Created job '{name}' (id: {job_id}) — schedule: {cron}"

    @tool
    def list_jobs(self) -> str:
        """List all scheduled jobs and their status."""
        with get_session() as session:
            jobs = session.query(Job).order_by(Job.created_at.desc()).all()

        if not jobs:
            return "No scheduled jobs."

        lines = [f"Scheduled jobs ({len(jobs)} total):"]
        for job in jobs:
            status = "enabled" if job.enabled else "disabled"
            lines.append(f"- [{status}] {job.name} (id: {job.id}) — cron: {job.cron}")
            lines.append(f"  prompt: {job.prompt[:80]}")
        return "\n".join(lines)

    @tool
    def delete_job(self, job_id: str) -> str:
        """Delete a scheduled job and its run history.

        Args:
            job_id: The ID of the job to delete
        """
        with get_session() as session:
            job = session.get(Job, job_id)
            if not job:
                return f"Job '{job_id}' not found."
            session.delete(job)
            session.commit()
        return f"Job '{job_id}' deleted."

    @tool
    def toggle_job(self, job_id: str, enabled: bool) -> str:
        """Enable or disable a scheduled job.

        Args:
            job_id: The ID of the job to toggle
            enabled: True to enable, False to disable
        """
        with get_session() as session:
            job = session.get(Job, job_id)
            if not job:
                return f"Job '{job_id}' not found."
            job.enabled = enabled
            session.commit()
        return f"Job '{job_id}' {'enabled' if enabled else 'disabled'}."

    @tool
    def job_history(self, job_id: str, limit: int = 5) -> str:
        """View recent run history for a scheduled job.

        Args:
            job_id: The ID of the job
            limit: Maximum number of runs to show
        """
        with get_session() as session:
            runs = (
                session.query(JobRun)
                .filter(JobRun.job_id == job_id)
                .order_by(JobRun.started_at.desc())
                .limit(limit)
                .all()
            )

        if not runs:
            return f"No run history for job '{job_id}'."

        lines = [f"Recent runs for '{job_id}':"]
        for run in runs:
            output_preview = (run.output or "")[:100]
            lines.append(f"- [{run.status}] {run.started_at} — {output_preview}")
        return "\n".join(lines)
