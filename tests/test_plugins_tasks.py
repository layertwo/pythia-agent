"""Tests for TasksPlugin."""

import json

import pytest

from pythia_agent.plugins.tasks import TasksPlugin


@pytest.fixture
def tasks_plugin():
    return TasksPlugin()


def test_get_tasks_empty(tasks_plugin):
    result = tasks_plugin.get_tasks()
    assert "No tasks" in result


def test_update_tasks_replace(tasks_plugin):
    tasks_json = json.dumps(
        [
            {"id": "t1", "content": "Do thing one", "status": "pending"},
            {"id": "t2", "content": "Do thing two", "status": "in_progress"},
        ]
    )
    result = tasks_plugin.update_tasks(tasks=tasks_json, merge=False)
    assert "t1" in result
    assert "t2" in result
    assert "pending" in result or "⏳" in result


def test_update_tasks_merge(tasks_plugin):
    initial = json.dumps(
        [
            {"id": "t1", "content": "First task", "status": "pending"},
        ]
    )
    tasks_plugin.update_tasks(tasks=initial, merge=False)

    update = json.dumps(
        [
            {"id": "t1", "content": "First task", "status": "completed"},
            {"id": "t2", "content": "New task", "status": "pending"},
        ]
    )
    result = tasks_plugin.update_tasks(tasks=update, merge=True)
    assert "t1" in result
    assert "t2" in result
    assert "completed" in result or "✅" in result


def test_update_tasks_invalid_json(tasks_plugin):
    result = tasks_plugin.update_tasks(tasks="not json", merge=False)
    assert "Error" in result or "error" in result.lower()


def test_update_tasks_invalid_status(tasks_plugin):
    tasks_json = json.dumps(
        [
            {"id": "t1", "content": "Task", "status": "invalid_status"},
        ]
    )
    result = tasks_plugin.update_tasks(tasks=tasks_json, merge=False)
    assert "invalid" in result.lower() or "status" in result.lower() or "pending" in result.lower()


def test_clear_tasks(tasks_plugin):
    tasks_json = json.dumps([{"id": "t1", "content": "Task", "status": "pending"}])
    tasks_plugin.update_tasks(tasks=tasks_json, merge=False)
    result = tasks_plugin.clear_tasks()
    assert "cleared" in result.lower() or "No tasks" in tasks_plugin.get_tasks()
