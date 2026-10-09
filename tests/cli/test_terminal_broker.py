"""Concurrency contract for the CLI's main-thread terminal broker."""

from __future__ import annotations

import ast
from concurrent.futures import CancelledError
from pathlib import Path
from queue import Queue
import sys
from threading import Event, Thread, current_thread, main_thread
from typing import Any

import pytest

from yian.cli.run import assemble_question_answers
from yian.cli.terminal import (
    ControllerComplete,
    PromptRequest,
    TerminalBroker,
)


_TIMEOUT = 2.0


def _ask_async(
    broker: TerminalBroker, **kwargs: Any
) -> tuple[Thread, Queue[tuple[str, Any]]]:
    """Call ``ask_user`` on a daemon worker and capture its value or exception."""
    outcome: Queue[tuple[str, Any]] = Queue()

    def call() -> None:
        try:
            outcome.put(("returned", broker.ask_user(**kwargs)))
        except BaseException as exc:
            outcome.put(("raised", exc))

    thread = Thread(target=call, daemon=True)
    thread.start()
    return thread, outcome


def _wait_for_messages(broker: TerminalBroker, count: int) -> list[Any]:
    """Wait until the broker's single FIFO contains at least ``count`` messages."""
    messages = broker._messages
    with messages.not_empty:
        ready = messages.not_empty.wait_for(
            lambda: len(messages.queue) >= count,
            timeout=_TIMEOUT,
        )
        assert ready, f"broker did not queue {count} messages"
        return list(messages.queue)


def _join(thread: Thread) -> None:
    """Join a worker with a bound and fail instead of leaking a blocked thread."""
    thread.join(_TIMEOUT)
    assert not thread.is_alive(), "worker remained blocked"


def _result(outcome: Queue[tuple[str, Any]]) -> Any:
    """Return a captured worker value, failing with its original exception."""
    state, value = outcome.get(timeout=_TIMEOUT)
    if state == "raised":
        raise value
    return value


def _exception(outcome: Queue[tuple[str, Any]]) -> BaseException:
    """Return a captured worker exception and assert that the call did not return."""
    state, value = outcome.get(timeout=_TIMEOUT)
    assert state == "raised"
    assert isinstance(value, BaseException)
    return value


def test_terminal_module_imports_only_the_standard_library() -> None:
    """Keep the broker below Rich, questionary, runtime, and node packages."""
    import yian.cli.terminal as terminal

    tree = ast.parse(Path(terminal.__file__).read_text())
    roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert roots <= sys.stdlib_module_names | {"__future__"}


def test_ask_user_blocks_until_serve_prompts_on_the_main_thread() -> None:
    """A worker receives exactly the value produced by the main-thread prompt."""
    prompt_threads: list[Any] = []
    prompted = Event()

    def prompt(request: PromptRequest) -> str:
        prompt_threads.append(current_thread())
        prompted.set()
        return f"answer to {request.question}"

    broker = TerminalBroker(prompt)
    worker, outcome = _ask_async(broker, question="Continue?")
    queued = _wait_for_messages(broker, 1)

    assert isinstance(queued[0], PromptRequest)
    assert worker.is_alive()
    assert not prompted.is_set()

    complete = broker.controller_complete(result="done")
    observed = broker.serve()
    _join(worker)

    assert observed is complete
    assert _result(outcome) == "answer to Continue?"
    assert prompt_threads == [main_thread()]


def test_concurrent_prompts_are_served_in_fifo_order() -> None:
    """Concurrent callers retain the one queue's deterministic insertion order."""
    prompted: list[str] = []

    def prompt(request: PromptRequest) -> str:
        prompted.append(request.question)
        return request.question.upper()

    broker = TerminalBroker(prompt)
    workers: list[Thread] = []
    outcomes: list[Queue[tuple[str, Any]]] = []
    for index, question in enumerate(("first", "second", "third"), start=1):
        worker, outcome = _ask_async(broker, question=question)
        workers.append(worker)
        outcomes.append(outcome)
        _wait_for_messages(broker, index)

    broker.controller_complete()
    broker.serve()
    for worker in workers:
        _join(worker)

    assert prompted == ["first", "second", "third"]
    assert [_result(outcome) for outcome in outcomes] == [
        "FIRST",
        "SECOND",
        "THIRD",
    ]


def test_render_and_prompt_callbacks_share_fifo_and_main_thread() -> None:
    """Render actions and prompts execute in enqueue order, only inside ``serve``."""
    calls: list[tuple[str, Any]] = []

    def prompt(request: PromptRequest) -> str:
        calls.append(("prompt", current_thread()))
        return "yes"

    broker = TerminalBroker(prompt)
    broker.render(lambda: calls.append(("before", current_thread())))
    worker, outcome = _ask_async(broker, question="Proceed?")
    _wait_for_messages(broker, 2)
    broker.render(lambda: calls.append(("after", current_thread())))
    broker.controller_complete()

    assert calls == []
    broker.serve()
    _join(worker)

    assert _result(outcome) == "yes"
    assert [name for name, _ in calls] == ["before", "prompt", "after"]
    assert all(thread is main_thread() for _, thread in calls)


def test_structured_prompt_metadata_reaches_handler_unchanged() -> None:
    """Structured questions and display metadata survive the worker/main handoff."""
    questions = [
        {
            "question": "Choose areas",
            "header": "Areas",
            "options": [{"label": "API"}, {"label": "UI"}],
            "multi_select": True,
        },
        {
            "question": "Notes?",
            "header": "Notes",
            "options": [],
            "multi_select": False,
        },
    ]
    answer_schema = [
        {"name": "Areas", "type": "list[str]"},
        {"name": "Notes", "type": "str"},
    ]
    scripted = {"Areas": ["API", "UI"], "Notes": "ship it"}
    seen: list[PromptRequest] = []

    def prompt(request: PromptRequest) -> dict[str, Any]:
        seen.append(request)
        assert request.questions is not None
        return assemble_question_answers(
            request.questions,
            ask=lambda item: scripted[item["header"]],
        )

    broker = TerminalBroker(prompt)
    worker, outcome = _ask_async(
        broker,
        question="Complete the review",
        questions=questions,
        answer_schema=answer_schema,
        title="Release review",
        node_id="review/approval",
    )
    _wait_for_messages(broker, 1)
    broker.controller_complete()
    broker.serve()
    _join(worker)

    assert _result(outcome) == scripted
    assert len(seen) == 1
    request = seen[0]
    assert request.question == "Complete the review"
    assert request.questions is questions
    assert request.answer_schema is answer_schema
    assert request.title == "Release review"
    assert request.node_id == "review/approval"


def test_ask_user_options_reach_the_prompt_request() -> None:
    """Model-sent options travel to the main-thread prompt with the question."""
    seen: list[PromptRequest] = []

    def prompt(request: PromptRequest) -> str:
        seen.append(request)
        return "A"

    broker = TerminalBroker(prompt)
    worker, outcome = _ask_async(broker, question="Pick one", options=["A", "B"])
    _wait_for_messages(broker, 1)
    broker.controller_complete()
    broker.serve()
    _join(worker)

    assert _result(outcome) == "A"
    assert [request.options for request in seen] == [["A", "B"]]


def test_generic_eof_fails_current_prompt_and_cancels_queued_prompts() -> None:
    """EOF ends a non-chat interaction without leaving sibling callers blocked."""
    prompted: list[str] = []

    def prompt(request: PromptRequest) -> str:
        prompted.append(request.question)
        raise EOFError("input closed")

    broker = TerminalBroker(prompt)
    first_worker, first_outcome = _ask_async(broker, question="first")
    _wait_for_messages(broker, 1)
    second_worker, second_outcome = _ask_async(broker, question="second")
    _wait_for_messages(broker, 2)
    complete = broker.controller_complete()

    assert broker.serve() is complete
    _join(first_worker)
    _join(second_worker)

    first_error = _exception(first_outcome)
    second_error = _exception(second_outcome)
    assert isinstance(first_error, EOFError)
    assert str(first_error) == "input closed"
    assert isinstance(second_error, CancelledError)
    assert prompted == ["first"]


def test_chat_eof_returns_configured_exit_value() -> None:
    """A chat broker can translate terminal EOF into its ordinary exit command."""
    broker = TerminalBroker(
        lambda request: (_ for _ in ()).throw(EOFError()),
        eof_value="/exit",
    )
    worker, outcome = _ask_async(broker, question="chat")
    _wait_for_messages(broker, 1)
    broker.controller_complete()

    broker.serve()
    _join(worker)

    assert _result(outcome) == "/exit"


def test_stop_event_wakes_queued_waiter_without_serving_prompt() -> None:
    """The engine's shared stop signal releases a worker before controller join."""
    stop_event = Event()
    prompted: list[str] = []
    broker = TerminalBroker(
        lambda request: prompted.append(request.question),
        stop_event=stop_event,
    )
    worker, outcome = _ask_async(broker, question="queued prompt")
    _wait_for_messages(broker, 1)

    stop_event.set()
    _join(worker)

    assert isinstance(_exception(outcome), CancelledError)
    complete = broker.controller_complete()
    assert broker.serve() is complete
    assert prompted == []


def test_stop_event_set_before_prompt_completion_discards_late_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completion wake rechecks a stop that arrived while the handler ran."""
    stop_event = Event()
    waiter_entered = Event()

    def wait_for_completion(request: PromptRequest, timeout: float) -> bool:
        waiter_entered.set()
        return request._completed.wait(_TIMEOUT)

    monkeypatch.setattr(PromptRequest, "_wait", wait_for_completion)

    def prompt(request: PromptRequest) -> str:
        assert waiter_entered.wait(_TIMEOUT)
        stop_event.set()
        return "late answer"

    broker = TerminalBroker(prompt, stop_event=stop_event)
    worker, outcome = _ask_async(broker, question="racing prompt")
    _wait_for_messages(broker, 1)
    complete = broker.controller_complete()

    assert broker.serve() is complete
    _join(worker)

    assert isinstance(_exception(outcome), CancelledError)


def test_stop_event_wakes_active_prompt_and_discards_its_late_answer() -> None:
    """A failed sibling wakes the main widget through its cancellation hook."""
    stop_event = Event()
    active = Event()
    prompt_released = Event()
    hook_called = Event()
    completions: Queue[ControllerComplete] = Queue()

    def prompt(request: PromptRequest) -> str:
        def cancel_prompt() -> None:
            hook_called.set()
            raise RuntimeError("cancellation hook failed")

        request.set_cancel_callback(cancel_prompt)
        active.set()
        if hook_called.wait(_TIMEOUT):
            prompt_released.set()
        return "late answer"

    broker = TerminalBroker(prompt, stop_event=stop_event)
    worker, outcome = _ask_async(broker, question="blocking prompt")
    _wait_for_messages(broker, 1)

    def stop() -> None:
        if active.wait(_TIMEOUT):
            stop_event.set()
        completions.put(broker.controller_complete(result="cancelled"))

    stopper = Thread(target=stop, daemon=True)
    stopper.start()
    observed = broker.serve()
    _join(stopper)
    _join(worker)

    assert observed is completions.get(timeout=_TIMEOUT)
    assert hook_called.is_set()
    assert prompt_released.is_set()
    assert isinstance(_exception(outcome), CancelledError)


def test_cancel_all_invokes_every_hook_and_ignores_callback_failures() -> None:
    """Cleanup hooks cannot strand waiters or replace the initiating failure."""
    broker = TerminalBroker(lambda request: "unused")
    workers: list[Thread] = []
    outcomes: list[Queue[tuple[str, Any]]] = []
    for index in range(3):
        worker, outcome = _ask_async(broker, question=f"hook-{index}")
        workers.append(worker)
        outcomes.append(outcome)
    messages = _wait_for_messages(broker, 3)
    requests = [message for message in messages if isinstance(message, PromptRequest)]
    called: list[int] = []

    for index, request in enumerate(requests):
        def fail(index: int = index) -> None:
            called.append(index)
            raise RuntimeError(f"hook {index} failed")

        request.set_cancel_callback(fail)

    broker.cancel_all()
    for worker in workers:
        _join(worker)

    assert sorted(called) == [0, 1, 2]
    assert all(
        isinstance(_exception(outcome), CancelledError) for outcome in outcomes
    )
    complete = broker.controller_complete()
    assert broker.serve() is complete


def test_cancel_all_wakes_queued_waiters_and_rejects_later_prompt() -> None:
    """Cancellation is terminal for every pending and subsequent ``ask_user`` call."""
    prompted: list[str] = []
    broker = TerminalBroker(lambda request: prompted.append(request.question))
    workers: list[Thread] = []
    outcomes: list[Queue[tuple[str, Any]]] = []
    for index in range(3):
        worker, outcome = _ask_async(broker, question=f"queued-{index}")
        workers.append(worker)
        outcomes.append(outcome)
    _wait_for_messages(broker, 3)

    broker.cancel_all()
    for worker in workers:
        _join(worker)
    with pytest.raises(CancelledError):
        broker.ask_user(question="too late")

    complete = broker.controller_complete()
    assert broker.serve() is complete
    assert prompted == []
    assert all(
        isinstance(_exception(outcome), CancelledError) for outcome in outcomes
    )


def test_completion_is_exact_fifo_sentinel_and_duplicate_is_rejected() -> None:
    """The first completion object terminates serving after earlier queued work."""
    rendered: list[str] = []
    token = object()
    broker = TerminalBroker(lambda request: "unused")
    broker.render(lambda: rendered.append("before completion"))
    complete = broker.controller_complete(result=token)

    with pytest.raises(RuntimeError, match="completion"):
        broker.controller_complete()

    observed = broker.serve()

    assert observed is complete
    assert observed.result is token
    assert observed.exception is None
    assert rendered == ["before completion"]


def test_serve_rejects_non_main_thread_without_consuming_completion() -> None:
    """Terminal ownership cannot accidentally migrate to a controller worker."""
    broker = TerminalBroker(lambda request: "unused")
    complete = broker.controller_complete()
    outcome: Queue[BaseException] = Queue()

    def serve_off_thread() -> None:
        try:
            broker.serve()
        except BaseException as exc:
            outcome.put(exc)

    worker = Thread(target=serve_off_thread, daemon=True)
    worker.start()
    _join(worker)

    error = outcome.get(timeout=_TIMEOUT)
    assert isinstance(error, RuntimeError)
    assert "main thread" in str(error)
    assert broker.serve() is complete
