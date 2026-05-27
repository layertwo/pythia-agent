"""Tests for SystemToolsPlugin."""

import pytest

from pythia_agent.plugins.system_tools import SystemToolsPlugin


@pytest.fixture
def system_plugin(tmp_path):
    return SystemToolsPlugin(working_dir=str(tmp_path))


def test_current_time_utc(system_plugin):
    result = system_plugin.current_time()
    assert "T" in result
    assert "+" in result or "Z" in result


def test_current_time_pacific(system_plugin):
    result = system_plugin.current_time(timezone="US/Pacific")
    assert "T" in result


def test_current_time_invalid(system_plugin):
    result = system_plugin.current_time(timezone="Invalid/Zone")
    assert "Error" in result


def test_shell_echo(system_plugin):
    result = system_plugin.shell(command="echo hello")
    assert "hello" in result


def test_shell_nonzero_exit(system_plugin):
    result = system_plugin.shell(command="exit 1")
    assert "[exit code]: 1" in result


def test_shell_timeout(system_plugin):
    result = system_plugin.shell(command="sleep 10", timeout=1)
    assert "timed out" in result


def test_python_exec_basic(system_plugin):
    result = system_plugin.python_exec(code="print(2 + 2)")
    assert "4" in result


def test_python_exec_error(system_plugin):
    result = system_plugin.python_exec(code="raise ValueError('boom')")
    assert "boom" in result


def test_file_read_write(system_plugin, tmp_path):
    test_file = tmp_path / "test.txt"
    system_plugin.file_write(path=str(test_file), content="hello world")
    result = system_plugin.file_read(path=str(test_file))
    assert "hello world" in result


def test_file_read_not_found(system_plugin):
    result = system_plugin.file_read(path="/nonexistent/file.txt")
    assert "not found" in result.lower() or "Error" in result


def test_file_write_creates_dirs(system_plugin, tmp_path):
    nested_path = tmp_path / "a" / "b" / "c.txt"
    result = system_plugin.file_write(path=str(nested_path), content="nested")
    assert "Written" in result
    assert nested_path.read_text() == "nested"


def test_calculator_basic(system_plugin):
    assert system_plugin.calculator(expression="2 + 3") == "5"


def test_calculator_math_functions(system_plugin):
    assert system_plugin.calculator(expression="sqrt(16)") == "4.0"


def test_calculator_rejects_builtins(system_plugin):
    result = system_plugin.calculator(expression="__import__('os')")
    assert "Error" in result
