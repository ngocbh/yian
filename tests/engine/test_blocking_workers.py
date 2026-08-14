"""Blocking node calls consume exactly the configured worker capacity."""

from __future__ import annotations

import threading
import time

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.events import RunSucceeded
from agent_composer.nodes.human_input import HumanInputNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.tools import install_ask_user
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _engine, _graph


def _drive_in_thread(engine: FlowEngine):
    """Run an engine in a joinable thread and retain its events or exception."""
    outcome = {}

    def drive() -> None:
        try:
            outcome["events"] = list(engine.run())
        except BaseException as exception:
            outcome["exception"] = exception

    thread = threading.Thread(target=drive, name="blocking-worker-test")
    thread.start()
    return thread, outcome


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Wait briefly for a dispatcher-owned state transition."""
    deadline = time.monotonic() + timeout
    while not predicate():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        threading.Event().wait(min(0.01, remaining))
    return True


def test_independent_node_completes_while_one_worker_waits_for_human():
    prompt_entered = threading.Event()
    release_prompt = threading.Event()
    sibling_completed = threading.Event()

    def answer(**request):
        prompt_entered.set()
        assert release_prompt.wait(2), "test did not release the blocking prompt"
        return "answer"

    def finish_sibling(inputs):
        assert prompt_entered.wait(2), "sibling ran without the human worker blocking"
        sibling_completed.set()
        return "done"

    graph = _graph(
        [
            HumanInputNode("ask", prompt="Question?"),
            FuncNode("sibling", finish_sibling),
        ],
        [
            (START_ID, "ask"),
            (START_ID, "sibling"),
            ("ask", END_ID),
            ("sibling", END_ID),
        ],
    )

    engine = _engine(graph, num_workers=2)
    with install_ask_user(answer):
        thread, outcome = _drive_in_thread(engine)
        try:
            assert prompt_entered.wait(2), "HUMAN_INPUT did not occupy a worker"
            assert sibling_completed.wait(2), "the free worker did not run ready work"
            assert _wait_until(
                lambda: engine.state.pool.get_value("sibling") is not None
            ), (
                "the dispatcher did not commit the free worker's result while HUMAN_INPUT blocked"
            )
            assert engine.state.pool.get("sibling") == "done"
        finally:
            release_prompt.set()
            thread.join(2)

    assert not thread.is_alive()
    assert "exception" not in outcome
    assert isinstance(outcome["events"][-1], RunSucceeded)


def test_ready_work_waits_while_every_worker_is_blocked():
    entered_lock = threading.Lock()
    entered_count = 0
    all_workers_blocked = threading.Event()
    release_prompts = threading.Event()
    ready_node_completed = threading.Event()

    def answer(**request):
        nonlocal entered_count
        with entered_lock:
            entered_count += 1
            if entered_count == 2:
                all_workers_blocked.set()
        assert release_prompts.wait(2), "test did not release the blocking prompts"
        return request["node_id"]

    graph = _graph(
        [
            HumanInputNode("ask_one", prompt="First?"),
            HumanInputNode("ask_two", prompt="Second?"),
            FuncNode("ready", lambda inputs: ready_node_completed.set() or "done"),
        ],
        [
            (START_ID, "ask_one"),
            (START_ID, "ask_two"),
            (START_ID, "ready"),
            ("ask_one", END_ID),
            ("ask_two", END_ID),
            ("ready", END_ID),
        ],
    )

    with install_ask_user(answer):
        thread, outcome = _drive_in_thread(_engine(graph, num_workers=2))
        try:
            assert all_workers_blocked.wait(2), "both workers did not enter HUMAN_INPUT"
            assert not ready_node_completed.wait(0.05)
        finally:
            release_prompts.set()
            thread.join(2)

    assert not thread.is_alive()
    assert ready_node_completed.is_set()
    assert "exception" not in outcome
    assert isinstance(outcome["events"][-1], RunSucceeded)
