"""Controller-thread execution behind the main-thread terminal broker."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from queue import Queue
from threading import (
    Barrier,
    Event,
    Lock,
    Thread,
    current_thread,
    enumerate as enumerate_threads,
    main_thread,
)
from types import SimpleNamespace
from typing import Any

import pytest

import agent_composer.cli.run as run_cli
from agent_composer.cli.terminal import PromptRequest, TerminalBroker
from agent_composer.compile.model import END_ID, START_ID
from agent_composer.events import NodeStarted
from agent_composer.nodes.base import Node, NodeKind, Output, _HostStop
from agent_composer.run import RunResult
from agent_composer.tools import TOOL_REGISTRY, invoke_tool
from tests.engine._graph_builder import _graph


_TIMEOUT = 2.0


class _Reporter:
    """Record progress calls and the threads that executed them."""

    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, Any]] = []

    def start(self) -> None:
        self.calls.append(("start", current_thread()))

    def stop(self) -> None:
        self.calls.append(("stop", current_thread()))

    def handle(self, event: Any) -> None:
        self.calls.append(("handle", current_thread()))
        if self.failure is not None:
            raise self.failure


class _PooledAskNode(Node):
    """Block one real engine worker in the host-installed terminal tool."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        return Output(invoke_tool("ask_user", {"question": "Blocked sibling"}))


class _PooledFailAfterPrompt(Node):
    """Fail only after the main thread is serving the sibling's prompt."""

    kind = NodeKind.CODE
    prompt_active = Event()

    def _run_impl(self, inputs):
        assert type(self).prompt_active.wait(_TIMEOUT)
        raise RuntimeError("pooled primary failure")


class _PooledConcurrentAskNode(Node):
    """Start two broker calls together so EOF must outrank sibling cancellation."""

    kind = NodeKind.CODE
    barrier = Barrier(2)

    def _run_impl(self, inputs):
        type(self).barrier.wait(timeout=_TIMEOUT)
        return Output(invoke_tool("ask_user", {"question": self.id}))


def _success(output: Any = "done") -> RunResult:
    """Build one terminal result for controller lifecycle tests."""

    return RunResult(input={}, status="succeeded", output=output)


def test_prompt_free_result_transfers_after_controller_join_and_registry_cleanup() -> None:
    """A successful controller returns its exact result and leaves no live owner."""

    controller_threads: list[Thread] = []
    expected = _success()

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        return expected

    observed = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=lambda request: "unused",
        reporter=None,
    )

    assert observed is expected
    assert len(controller_threads) == 1
    assert controller_threads[0] is not main_thread()
    assert not controller_threads[0].is_alive()
    assert "ask_user" not in TOOL_REGISTRY


def test_sigint_deferred_at_thread_launch_still_joins_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending SIGINT delivered on mask restore sees a launched, tracked controller."""

    expected = KeyboardInterrupt()
    calls = []
    registry_seen: list[bool] = []

    def fake_pthread_sigmask(how, mask):
        calls.append((current_thread(), how, mask))
        if current_thread() is main_thread() and how == signal.SIG_BLOCK:
            return {signal.SIGTERM}
        if current_thread() is main_thread() and how == signal.SIG_SETMASK:
            raise expected
        return set()

    monkeypatch.setattr(signal, "pthread_sigmask", fake_pthread_sigmask)

    def controller(stop_event, on_event, ask_user, render):
        registry_seen.append("ask_user" in TOOL_REGISTRY)
        return _success("done")

    with pytest.raises(KeyboardInterrupt) as caught:
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unused",
            reporter=None,
        )

    assert caught.value is expected
    main_calls = [call for call in calls if call[0] is main_thread()]
    assert [call[1] for call in main_calls] == [signal.SIG_BLOCK, signal.SIG_SETMASK]
    assert any(call[0] is not main_thread() and call[1] == signal.SIG_SETMASK for call in calls)
    assert registry_seen == [True]
    assert not any(thread.name == "ac-controller" for thread in enumerate_threads())
    assert "ask_user" not in TOOL_REGISTRY


def test_controller_restores_caller_signal_mask_before_spawning_workers() -> None:
    """Controller and descendants do not inherit the launch-only SIGINT block."""

    caller_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    observed_masks = []

    def controller(stop_event, on_event, ask_user, render):
        observed_masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, set()))

        def inspect_worker_mask() -> None:
            observed_masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, set()))

        worker = Thread(target=inspect_worker_mask, name="test-mask-worker")
        worker.start()
        worker.join()
        return _success("done")

    observed = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=lambda request: "unused",
        reporter=None,
    )

    assert observed.output == "done"
    assert observed_masks == [caller_mask, caller_mask]
    assert signal.SIGINT not in observed_masks[0] or signal.SIGINT in caller_mask
    assert "ask_user" not in TOOL_REGISTRY


def test_controller_exception_identity_transfers_after_join() -> None:
    """A controller exception is re-raised unchanged after its thread exits."""

    expected = RuntimeError("controller failed")
    controller_threads: list[Thread] = []

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        raise expected

    with pytest.raises(RuntimeError) as caught:
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unused",
            reporter=None,
        )

    assert caught.value is expected
    assert not controller_threads[0].is_alive()
    assert "ask_user" not in TOOL_REGISTRY


def test_controller_host_stop_is_hidden_without_inventing_ctrl_c() -> None:
    """A private lifecycle stop becomes a public error, never a fake Ctrl-C."""

    def controller(stop_event, on_event, ask_user, render):
        raise _HostStop("internal lifecycle stop")

    with pytest.raises(RuntimeError, match="terminal controller stopped") as caught:
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unused",
            reporter=None,
        )

    assert isinstance(caught.value.__cause__, _HostStop)
    assert "ask_user" not in TOOL_REGISTRY


def test_prompt_eof_preserves_public_cause_without_leaking_host_stop() -> None:
    """EOF crosses the node boundary privately, then reappears exactly at the CLI host."""

    expected = EOFError("input closed")

    def controller(stop_event, on_event, ask_user, render):
        ask_user(question="Need input")
        return _success("unreachable")

    def prompt(request):
        raise expected

    with pytest.raises(EOFError) as caught:
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=prompt,
            reporter=None,
        )

    assert caught.value is expected
    assert "ask_user" not in TOOL_REGISTRY


def test_pooled_prompt_eof_preserves_exact_initiating_cause() -> None:
    """Worker-to-dispatcher host stop transport retains the active prompt's EOF."""

    expected = EOFError("pooled input closed")
    loaded = SimpleNamespace(
        input=[],
        compiled=_graph(
            [_PooledAskNode("ask")],
            [(START_ID, "ask"), ("ask", END_ID)],
        ),
    )

    def controller(stop_event, on_event, ask_user, render):
        return run_cli._run_flow_controlled(
            loaded,
            {},
            num_workers=1,
            stop_event=stop_event,
        )

    def prompt(request):
        raise expected

    with pytest.raises(EOFError) as caught:
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=prompt,
            reporter=None,
        )

    assert caught.value is expected
    assert not any(
        thread.name.startswith(("ac-worker-", "ac-controller"))
        for thread in enumerate_threads()
    )
    assert "ask_user" not in TOOL_REGISTRY


def test_pooled_prompt_eof_outranks_cancelled_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active EOF remains primary even if a cancelled sibling unwinds first."""

    _PooledConcurrentAskNode.barrier.reset()
    expected = EOFError("active prompt closed")
    both_waiting = Event()
    waiting_lock = Lock()
    waiting_count = 0
    original_wait = TerminalBroker._wait_for_prompt

    def counted_wait(broker, request):
        nonlocal waiting_count
        with waiting_lock:
            waiting_count += 1
            if waiting_count == 2:
                both_waiting.set()
        return original_wait(broker, request)

    monkeypatch.setattr(TerminalBroker, "_wait_for_prompt", counted_wait)
    loaded = SimpleNamespace(
        input=[],
        compiled=_graph(
            [_PooledConcurrentAskNode("ask-a"), _PooledConcurrentAskNode("ask-b")],
            [
                (START_ID, "ask-a"),
                (START_ID, "ask-b"),
                ("ask-a", END_ID),
                ("ask-b", END_ID),
            ],
        ),
    )

    def controller(stop_event, on_event, ask_user, render):
        return run_cli._run_flow_controlled(
            loaded,
            {},
            num_workers=2,
            stop_event=stop_event,
        )

    def prompt(request):
        assert both_waiting.wait(_TIMEOUT)
        raise expected

    with pytest.raises(EOFError) as caught:
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=prompt,
            reporter=None,
        )

    assert caught.value is expected
    assert waiting_count == 2
    assert not any(
        thread.name.startswith(("ac-worker-", "ac-controller"))
        for thread in enumerate_threads()
    )
    assert "ask_user" not in TOOL_REGISTRY


def test_brokered_widget_keeps_main_thread_keyboard_interrupt_visible() -> None:
    """Broker prompts bypass questionary.ask, which would swallow Ctrl-C as None."""

    expected = KeyboardInterrupt()

    class Application:
        def run(self, *, pre_run):
            async def drive():
                pre_run()
                raise expected

            return asyncio.run(drive())

    class Widget:
        should_skip_question = False
        default = None
        application = Application()

        def ask(self):
            raise AssertionError("brokered prompts must not use questionary.ask")

    request = PromptRequest(question="Continue?")
    with pytest.raises(KeyboardInterrupt) as caught:
        run_cli._ask_widget(Widget(), request)

    assert caught.value is expected
    assert request._cancel_callback is None


def test_brokered_widget_cancel_schedules_exit_on_main_prompt_loop() -> None:
    """A worker cancellation wakes prompt-toolkit through its owning asyncio loop."""

    exit_threads: list[Thread] = []
    cancelling_threads: list[Thread] = []

    class Application:
        exception: BaseException | None = None

        def exit(self, *, exception):
            exit_threads.append(current_thread())
            self.exception = exception

        def run(self, *, pre_run):
            async def drive():
                pre_run()

                def cancel_from_worker() -> None:
                    cancelling_threads.append(current_thread())
                    callback = request._cancel()
                    assert callback is not None
                    callback()

                worker = Thread(target=cancel_from_worker)
                worker.start()
                worker.join()
                await asyncio.sleep(0)
                assert self.exception is not None
                raise self.exception

            return asyncio.run(drive())

    class Widget:
        should_skip_question = False
        default = None
        application = Application()

    request = PromptRequest(question="Continue?")
    with pytest.raises(_HostStop):
        run_cli._ask_widget(Widget(), request)

    assert exit_threads == [main_thread()]
    assert len(cancelling_threads) == 1
    assert cancelling_threads[0] is not main_thread()
    assert request._cancel_callback is None


def test_prompt_and_progress_render_only_on_main_thread() -> None:
    """The controller enqueues both effects; the main thread performs them."""

    prompt_threads: list[Thread] = []
    reporter = _Reporter()
    controller_threads: list[Thread] = []

    def prompt(request):
        prompt_threads.append(current_thread())
        return f"answer:{request.question}"

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        assert on_event is not None
        on_event(NodeStarted("work"))
        answer = invoke_tool("ask_user", {"question": "Continue?"})
        return _success(answer)

    observed = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=prompt,
        reporter=reporter,
    )

    assert observed.output == "answer:Continue?"
    assert prompt_threads == [main_thread()]
    assert controller_threads[0] is not main_thread()
    assert all(thread is main_thread() for _, thread in reporter.calls)
    assert [name for name, _ in reporter.calls].count("handle") == 1


def test_ctrl_c_during_active_prompt_preserves_keyboard_interrupt_and_joins() -> None:
    """Prompt Ctrl-C wakes the controller, which exits before the interrupt escapes."""

    cancellation_hook = Event()
    controller_threads: list[Thread] = []

    def prompt(request):
        request.set_cancel_callback(cancellation_hook.set)
        raise KeyboardInterrupt

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        invoke_tool("ask_user", {"question": "Blocking prompt"})
        return _success("unreachable")

    with pytest.raises(KeyboardInterrupt):
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=prompt,
            reporter=None,
        )

    assert cancellation_hook.is_set()
    assert not controller_threads[0].is_alive()
    assert "ask_user" not in TOOL_REGISTRY


def test_second_sigint_during_drain_waits_for_controller_before_registry_cleanup() -> None:
    """Repeated Ctrl-C is deferred until the stopped controller has really finished."""

    controller_entered = Event()
    cleanup_entered = Event()
    release_controller = Event()
    controller_done = Event()
    registry_seen: list[bool] = []

    def controller(stop_event, on_event, ask_user, render):
        controller_entered.set()
        assert stop_event.wait(_TIMEOUT)
        cleanup_entered.set()
        assert release_controller.wait(_TIMEOUT)
        registry_seen.append("ask_user" in TOOL_REGISTRY)
        controller_done.set()
        return _success("stopped")

    def interrupt_twice() -> None:
        assert controller_entered.wait(_TIMEOUT)
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)
        assert cleanup_entered.wait(_TIMEOUT)
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)
        # Leave the old implementation enough time to unwind its registry context
        # before the controller checks ownership; the fixed path keeps waiting inside it.
        time.sleep(0.05)
        release_controller.set()

    interrupter = Thread(target=interrupt_twice, name="test-double-sigint")
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_cli._run_with_terminal_controller(
                controller,
                prompt_handler=lambda request: "unused",
                reporter=None,
            )
    finally:
        release_controller.set()
        interrupter.join(timeout=_TIMEOUT)
        assert controller_done.wait(_TIMEOUT)

    assert registry_seen == [True]
    assert not interrupter.is_alive()
    assert not any(
        thread.name.startswith(("ac-worker-", "ac-controller"))
        for thread in enumerate_threads()
    )
    assert "ask_user" not in TOOL_REGISTRY


def test_observer_failure_wins_over_cancelled_controller_prompt() -> None:
    """A main-thread render failure remains the cause after waking blocked input."""

    expected = LookupError("observer failed")
    reporter = _Reporter(failure=expected)
    controller_threads: list[Thread] = []

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        assert on_event is not None
        on_event(NodeStarted("work"))
        ask_user(question="queued behind failed render")
        return _success("unreachable")

    with pytest.raises(LookupError) as caught:
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unreachable",
            reporter=reporter,
        )

    assert caught.value is expected
    assert not controller_threads[0].is_alive()
    assert "ask_user" not in TOOL_REGISTRY


def test_failed_result_wins_over_sibling_prompt_host_stop() -> None:
    """Drain cancellation never replaces the flow failure that initiated it."""

    active = Event()
    released = Event()
    sibling_errors: Queue[BaseException] = Queue()
    sibling_threads: list[Thread] = []
    failed = RunResult(input={}, status="failed", error="boom")

    def prompt(request):
        request.set_cancel_callback(released.set)
        active.set()
        assert released.wait(_TIMEOUT)
        return "late answer"

    def controller(stop_event, on_event, ask_user, render):
        def sibling() -> None:
            try:
                ask_user(question="blocked sibling")
            except BaseException as exc:
                sibling_errors.put(exc)

        worker = Thread(target=sibling, name="test-ask-sibling")
        sibling_threads.append(worker)
        worker.start()
        assert active.wait(_TIMEOUT)
        stop_event.set()
        worker.join()
        assert not worker.is_alive()
        return failed

    observed = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=prompt,
        reporter=None,
    )

    assert observed is failed
    assert released.is_set()
    assert isinstance(sibling_errors.get(timeout=_TIMEOUT), _HostStop)
    assert not sibling_threads[0].is_alive()


def test_real_pooled_failure_wakes_prompt_and_joins_every_engine_worker() -> None:
    """A real fan-out failure drains its broker-blocked sibling without cause loss."""

    _PooledFailAfterPrompt.prompt_active.clear()
    prompt_released = Event()
    loaded = SimpleNamespace(
        input=[],
        compiled=_graph(
            [_PooledAskNode("ask"), _PooledFailAfterPrompt("fail")],
            [
                (START_ID, "ask"),
                (START_ID, "fail"),
                ("ask", END_ID),
                ("fail", END_ID),
            ],
        ),
    )

    def prompt(request):
        request.set_cancel_callback(prompt_released.set)
        _PooledFailAfterPrompt.prompt_active.set()
        assert prompt_released.wait(_TIMEOUT)
        return "late answer"

    def controller(stop_event, on_event, ask_user, render):
        return run_cli._run_flow_controlled(
            loaded,
            {},
            on_event=on_event,
            num_workers=2,
            stop_event=stop_event,
        )

    observed = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=prompt,
        reporter=None,
    )

    assert observed.status == "failed"
    assert observed.error == "pooled primary failure"
    assert prompt_released.is_set()
    assert not any(
        thread.name.startswith(("ac-worker-", "ac-controller"))
        for thread in enumerate_threads()
    )
    assert "ask_user" not in TOOL_REGISTRY


def test_dormant_pause_resume_stays_on_controller_and_prompts_on_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy Pause remains usable without moving resume or input onto main incorrectly."""

    engine = object()
    reason = SimpleNamespace(
        type="human_input_required",
        prompt="Approve?",
        questions=[],
        answer_schema=[],
        node_title="Approval",
        node_id="approve",
    )
    paused = RunResult(
        input={},
        status="paused",
        engine=engine,
        pause_reasons=[reason],
    )
    resumed = _success("approved")
    prompt_threads: list[Thread] = []
    resume_threads: list[Thread] = []

    monkeypatch.setattr(
        run_cli,
        "resume_command",
        lambda loaded, pause_reason, value: (pause_reason.node_id, value),
    )

    def fake_resume(loaded, *, engine, commands, on_event):
        resume_threads.append(current_thread())
        assert commands == [("approve", "yes")]
        return resumed

    monkeypatch.setattr(run_cli, "resume_flow", fake_resume)

    def prompt(request):
        prompt_threads.append(current_thread())
        assert request.question == "Approve?"
        return "yes"

    def controller(stop_event, on_event, ask_user, render):
        return run_cli._resume_to_terminal(
            object(),
            paused,
            on_event=on_event,
            ask_user=ask_user,
            render=render,
        )

    observed = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=prompt,
        reporter=None,
    )

    assert observed is resumed
    assert prompt_threads == [main_thread()]
    assert len(resume_threads) == 1
    assert resume_threads[0] is not main_thread()
