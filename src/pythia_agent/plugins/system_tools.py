"""System tools plugin: shell, python execution, file I/O, time, calculator."""

import math
import os
import subprocess
import tempfile
from datetime import datetime
from datetime import timezone as tz
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from strands import tool
from strands.plugins import Plugin


GUIDANCE = (
    "\n\nYou can execute shell commands, run Python code, and read/write files. "
    "Prefer shell for quick operations and python_exec for multi-step logic."
)


class SystemToolsPlugin(Plugin):
    """Provides shell, Python REPL, file I/O, time, and calculator tools."""

    name = "system-tools"

    def __init__(self, working_dir: str | None = None, shell_timeout: int = 30):
        self.working_dir = working_dir or os.getcwd()
        self.shell_timeout = shell_timeout
        super().__init__()

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    @tool
    def current_time(self, timezone: str = "UTC") -> str:
        """Get the current date and time in ISO 8601 format.

        Args:
            timezone: Timezone name (e.g., 'UTC', 'US/Pacific', 'Europe/London')
        """
        try:
            if timezone.upper() == "UTC":
                tz_obj: Any = tz.utc
            else:
                tz_obj = ZoneInfo(timezone)
            return datetime.now(tz_obj).isoformat()
        except Exception as e:
            return f"Error: {e}"

    @tool
    def shell(self, command: str, work_dir: str = "", timeout: int = 0) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to execute
            work_dir: Working directory for the command (defaults to agent working dir)
            timeout: Command timeout in seconds (0 uses default)
        """
        cwd = work_dir or self.working_dir
        t = timeout or self.shell_timeout

        try:
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=t,
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\n[stderr]: {result.stderr}" if result.stderr else ""
                output += f"\n[exit code]: {result.returncode}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {t}s"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def python_exec(self, code: str) -> str:
        """Execute Python code and return the output.

        The code runs in an isolated subprocess. Use print() to produce output.

        Args:
            code: Python code to execute
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            script_path = f.name

        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=self.shell_timeout,
                cwd=self.working_dir,
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\n[stderr]: {result.stderr}" if result.stderr else ""
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {self.shell_timeout}s"
        except Exception as e:
            return f"Error: {e}"
        finally:
            os.unlink(script_path)

    @tool
    def file_read(self, path: str, offset: int = 0, limit: int = 0) -> str:
        """Read contents of a file.

        Args:
            path: Path to the file to read
            offset: Line number to start reading from (0-based)
            limit: Maximum number of lines to read (0 = all)
        """
        resolved = Path(self.working_dir) / path if not Path(path).is_absolute() else Path(path)
        try:
            text = resolved.read_text()
            lines = text.splitlines(keepends=True)
            if offset or limit:
                end = offset + limit if limit else len(lines)
                lines = lines[offset:end]
            return "".join(lines)
        except FileNotFoundError:
            return f"Error: file not found: {resolved}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def file_write(self, path: str, content: str, append: bool = False) -> str:
        """Write content to a file.

        Args:
            path: Path to the file to write
            content: Content to write
            append: If true, append to file instead of overwriting
        """
        resolved = Path(self.working_dir) / path if not Path(path).is_absolute() else Path(path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with resolved.open(mode) as f:
                f.write(content)
            return f"Written to {resolved} ({'appended' if append else 'created/overwritten'})"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def calculator(self, expression: str) -> str:
        """Evaluate a mathematical expression safely.

        Supports basic arithmetic, power, modulo, and common math functions.

        Args:
            expression: Mathematical expression to evaluate (e.g., '2 ** 10', 'sqrt(144)')
        """
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"
