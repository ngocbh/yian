"""Memory defaults and the memory JSON nesting bound come from `yian.defaults`."""

from __future__ import annotations

import json

import pytest

from yian.memory.contracts import MemoryMutation
from yian.memory.episodic import EpisodicMemory
from yian.memory.profile import ProfileMemory
from yian.memory.reflective import ReflectiveMemory
from yian.memory.session import SessionMemory
from yian.memory.storage import FileMarkdownStore
import yian.memory.episodic as episodic
import yian.memory.reflective as reflective
from tests.engine.test_reflective_memory import _context, _install_model, _mutation

_CONFIG = {"provider": "ollama", "model": "qwen"}
_NOOP = {"actions": [{"action": "noop", "lesson_id": None, "text": None}]}


def _nested(containers: int) -> dict:
    value: dict = {}
    for _ in range(containers - 1):
        value = {"nested": value}
    return value


def test_episodic_limits_default_from_the_environment(set_defaults) -> None:
    set_defaults(EPISODIC_MAX_EPISODES=3, EPISODIC_RECALL_LIMIT=2)
    with EpisodicMemory().open_run() as session:
        assert session.snapshot["config"] == {"max_episodes": 3, "recall_limit": 2}
    with EpisodicMemory(max_episodes=7, recall_limit=5).open_run() as session:
        assert session.snapshot["config"] == {"max_episodes": 7, "recall_limit": 5}
        recorded = session.snapshot
    with EpisodicMemory().open_run(snapshot=recorded) as session:
        assert session.snapshot["config"] == {"max_episodes": 7, "recall_limit": 5}


def test_reflective_limits_default_from_the_environment(set_defaults) -> None:
    set_defaults(REFLECTIVE_MAX_ITEMS=3, REFLECTIVE_RECALL_LIMIT=2)
    with ReflectiveMemory(_CONFIG).open_run() as session:
        assert session.snapshot["config"] == {"max_items": 3, "recall_limit": 2}
    with ReflectiveMemory(_CONFIG, max_items=7, recall_limit=5).open_run() as session:
        assert session.snapshot["config"] == {"max_items": 7, "recall_limit": 5}
        recorded = session.snapshot
    with ReflectiveMemory().open_run(snapshot=recorded) as session:
        assert session.snapshot["config"] == {"max_items": 7, "recall_limit": 5}


def _prompt_parts(prompt: str) -> dict[str, str]:
    labels = ("Memory: ", "Limits: ", "Current lessons: ", "Episodes: ")
    lines = prompt.split("\n")
    assert [line[: len(label)] for line, label in zip(lines, labels)] == list(labels)
    return {label: line[len(label):] for line, label in zip(lines, labels)}


def test_reflection_lesson_material_follows_its_budget(monkeypatch, set_defaults) -> None:
    set_defaults(REFLECTIVE_MAX_LESSON_CHARS=300)
    model, _ = _install_model(monkeypatch, _NOOP)
    memory = ReflectiveMemory(_CONFIG)
    with memory.open_run() as session:
        session.commit("big", (_mutation("big", "add", "big", "x" * 5_000),))
    with memory.open_run() as session:
        assert session.learn(_context()) == []
    lessons = _prompt_parts(model.structured.messages[-1].content)["Current lessons: "]
    assert len(lessons) <= 300
    assert json.loads(lessons)["truncated"] is True


def test_reflection_episode_material_follows_its_budget(monkeypatch, set_defaults) -> None:
    set_defaults(REFLECTIVE_MAX_EPISODE_CHARS=400)
    model, _ = _install_model(monkeypatch, _NOOP)
    with ReflectiveMemory(_CONFIG).open_run() as session:
        assert session.learn(_context(text="y" * 5_000)) == []
    episodes = _prompt_parts(model.structured.messages[-1].content)["Episodes: "]
    assert len(episodes) <= 400
    assert json.loads(episodes)["truncated"] is True


def test_reflection_limit_material_follows_its_budget(monkeypatch, set_defaults) -> None:
    # The limit material is a short object that never reaches the 256-char floor, so
    # the budget it is cut to is observed directly.
    set_defaults(REFLECTIVE_MAX_LIMIT_CHARS=500)
    budgets: list[int] = []
    bounded = reflective._bounded_json_material

    def spy(value, limit):
        budgets.append(limit)
        return bounded(value, limit)

    monkeypatch.setattr(reflective, "_bounded_json_material", spy)
    _install_model(monkeypatch, _NOOP)
    with ReflectiveMemory(_CONFIG).open_run() as session:
        assert session.learn(_context()) == []
    assert 500 in budgets


def test_session_memory_settings_default_from_the_environment(set_defaults) -> None:
    set_defaults(SESSION_HISTORY_BUDGET_CHARS=10, SESSION_RECENT_TURNS=1)
    memory = SessionMemory()
    assert (memory._history_budget_chars, memory._recent_turns) == (10, 1)
    explicit = SessionMemory(history_budget_chars=20, recent_turns=3)
    assert (explicit._history_budget_chars, explicit._recent_turns) == (20, 3)


def test_profile_memory_budgets_default_from_the_environment(tmp_path, set_defaults) -> None:
    set_defaults(PROFILE_MAX_CHARS=100, PROFILE_INPUT_BUDGET_CHARS=50)
    memory = ProfileMemory(FileMarkdownStore(tmp_path / "ada.md"))
    assert (memory.max_chars, memory.input_budget_chars) == (100, 50)
    explicit = ProfileMemory(
        FileMarkdownStore(tmp_path / "bob.md"), max_chars=200, input_budget_chars=70
    )
    assert (explicit.max_chars, explicit.input_budget_chars) == (200, 70)


@pytest.mark.parametrize(
    "check",
    [
        lambda value: MemoryMutation("deep", value),
        lambda value: episodic._copy_json(value, "value"),
        lambda value: reflective._copy_json(value, "value"),
    ],
    ids=["contracts", "episodic", "reflective"],
)
def test_memory_nesting_bound_follows_the_value_nesting_limit(check, set_defaults) -> None:
    set_defaults(MAX_VALUE_NESTING=80)
    check(_nested(108))
    with pytest.raises(ValueError, match="memory limit of 108 levels") as caught:
        check(_nested(109))
    assert "YIAN_MAX_VALUE_NESTING" in str(caught.value)
