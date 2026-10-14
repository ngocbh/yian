"""Controller-thread execution behind the main-thread terminal broker."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
import types
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
from typing import Any

import pytest

import yian.cli.run as run_cli
from yian.cli.chat import tools as chat_tools
from yian.cli.terminal import PromptRequest, TerminalBroker
from yian.compose import load_flow
from yian.diagnostics import Failure
from yian.events import NodeStarted
from yian.nodes.base import _HostStop
from yian.run import RunResult
from yian.tools import TOOL_REGISTRY


_TIMEOUT = 2.0
_CONCURRENT_ASK_BARRIER = Barrier(2)


def _pooled_fail_after_prompt(inputs: dict[str, Any]) -> str:
    """Fail only after the main thread begins serving the sibling prompt."""
    prompt_active = TerminalBroker._test_prompt_active
    assert prompt_active.wait(_TIMEOUT)
    raise RuntimeError("pooled primary failure")


def _concurrent_ask_tool(ask_user: Any) -> Any:
    """Return a per-run tool that enters two broker calls together."""

    def concurrent_ask(question: str) -> str:
        """Wait for the sibling, then ask so active EOF outranks cancellation."""
        _CONCURRENT_ASK_BARRIER.wait(timeout=_TIMEOUT)
        return ask_user(question=question)

    return concurrent_ask


_POOLED_ASK_FLOW = """
id: terminal-controller-ask
name: terminal_controller_ask
nodes:
  ask:
    kind: human_input
    prompt: Blocked sibling
    output: str
output: ${ask.output}
"""


_POOLED_CONCURRENT_ASK_FLOW = """
id: terminal-controller-concurrent-ask
name: terminal_controller_concurrent_ask
nodes:
  ask_a:
    kind: tool
    tool_id: concurrent_ask
    input:
      question: ask-a
  ask_b:
    kind: tool
    tool_id: concurrent_ask
    input:
      question: ask-b
output:
  a: ${ask_a.output}
  b: ${ask_b.output}
"""


_POOLED_FAILURE_FLOW = """
id: terminal-controller-failure
name: terminal_controller_failure
nodes:
  ask:
    kind: human_input
    prompt: Blocked sibling
    output: str
  fail:
    kind: code
    code: tests.cli.test_terminal_controller:_pooled_fail_after_prompt
    output: str
output:
  ask: ${ask.output}
  fail: ${fail.output}
"""


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


def _success(output: Any = "done") -> RunResult:
    """Build one terminal result for controller lifecycle tests."""

    return RunResult(
        input={},
        status="succeeded",
        run_id="controller-test",
        log=None,
        output=output,
    )


@pytest.fixture
def default_sigint_handler() -> Any:
    """Run the test with Python's default SIGINT handler and restore the previous one."""

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


def test_prompt_free_result_transfers_after_controller_join() -> None:
    """A successful controller returns its exact result and never touches the registry."""

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


def test_controller_stop_links_to_child_without_reverse_propagation() -> None:
    """A nested lifecycle observes parent stop but cannot poison its parent."""
    parent = run_cli._ControllerStopEvent(lambda: None)

    with parent.linked_child() as local_failure:
        local_failure.set()
        assert local_failure.is_set()
        assert not parent.is_set()

    with parent.linked_child() as host_cancelled:
        assert not host_cancelled.is_set()
        parent.set()
        assert host_cancelled.is_set()


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_sigint_during_thread_launch_stops_and_joins_the_controller(
    monkeypatch: pytest.MonkeyPatch,
    default_sigint_handler: None,
) -> None:
    """Ctrl-C handled inside Thread.start still stops and joins the tracked controller."""

    tripped = Event()
    controller_done = Event()
    saw_stop: list[bool] = []
    real_start = Thread.start

    def start_then_take_sigint(self: Thread) -> None:
        real_start(self)
        if self.name == "ac-controller":
            # The main thread runs the pending handler here, before start returns.
            assert tripped.wait(_TIMEOUT)
            time.sleep(0.01)

    monkeypatch.setattr(Thread, "start", start_then_take_sigint)

    def controller(stop_event, on_event, ask_user, render):
        signal.pthread_kill(current_thread().ident, signal.SIGINT)
        tripped.set()
        saw_stop.append(stop_event.wait(_TIMEOUT))
        # A slow drain, so a helper that skips the join returns before this ends.
        time.sleep(0.1)
        controller_done.set()
        return _success("stopped")

    with pytest.raises(KeyboardInterrupt):
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unused",
            reporter=None,
        )

    assert controller_done.is_set()
    assert saw_stop == [True]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert not any(thread.name == "ac-controller" for thread in enumerate_threads())
    assert "ask_user" not in TOOL_REGISTRY


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_sigint_on_the_controller_thread_stops_and_joins_before_raising(
    default_sigint_handler: None,
) -> None:
    """A SIGINT the OS hands to the controller thread still stops the run gracefully."""

    controller_done = Event()
    saw_stop: list[bool] = []

    def controller(stop_event, on_event, ask_user, render):
        signal.pthread_kill(current_thread().ident, signal.SIGINT)
        saw_stop.append(stop_event.wait(_TIMEOUT))
        time.sleep(0.1)
        controller_done.set()
        return _success("stopped")

    with pytest.raises(KeyboardInterrupt):
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unused",
            reporter=None,
        )

    assert controller_done.is_set()
    assert saw_stop == [True]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_sigint_deferred_at_launch_is_replayed_through_the_callers_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launch-time handler hands a recorded Ctrl-C to the handler it replaced."""

    calls: list[int] = []
    real_start = Thread.start

    def start_then_signal(self: Thread) -> None:
        real_start(self)
        if self.name == "ac-controller":
            signal.raise_signal(signal.SIGINT)
            assert calls == []  # deferred, not yet delivered to the caller

    def caller_handler(signum, frame):
        calls.append(signum)

    monkeypatch.setattr(Thread, "start", start_then_signal)
    previous = signal.signal(signal.SIGINT, caller_handler)
    try:
        try:
            observed = run_cli._run_with_terminal_controller(
                lambda stop_event, on_event, ask_user, render: _success("done"),
                prompt_handler=lambda request: "unused",
                reporter=None,
            )
        except KeyboardInterrupt:
            # Report a wrong replay as one failure instead of aborting the session.
            pytest.fail("the replay bypassed the caller's SIGINT handler")
        assert signal.getsignal(signal.SIGINT) is caller_handler
    finally:
        signal.signal(signal.SIGINT, previous)

    assert observed.output == "done"
    assert calls == [signal.SIGINT]


def test_sigint_deferred_at_launch_stays_ignored_for_an_ignoring_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller with SIGINT ignored gets no KeyboardInterrupt from the replay."""

    real_start = Thread.start

    def start_then_signal(self: Thread) -> None:
        real_start(self)
        if self.name == "ac-controller":
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(Thread, "start", start_then_signal)
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            observed = run_cli._run_with_terminal_controller(
                lambda stop_event, on_event, ask_user, render: _success("done"),
                prompt_handler=lambda request: "unused",
                reporter=None,
            )
        except KeyboardInterrupt:
            # Report a wrong replay as one failure instead of aborting the session.
            pytest.fail("the replay raised KeyboardInterrupt for an ignoring caller")
        restored = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, previous)

    assert observed.output == "done"
    assert restored == signal.SIG_IGN


def test_launch_leaves_a_non_python_sigint_handler_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no restorable SIGINT handler the controller starts directly."""

    def refuse(*args: Any) -> None:
        raise AssertionError("signal.signal must not be called")

    monkeypatch.setattr(signal, "getsignal", lambda signum: None)
    monkeypatch.setattr(signal, "signal", refuse)

    observed = run_cli._run_with_terminal_controller(
        lambda stop_event, on_event, ask_user, render: _success("done"),
        prompt_handler=lambda request: "unused",
        reporter=None,
    )

    assert observed.output == "done"


def test_controller_and_workers_keep_the_callers_signal_mask() -> None:
    """The controller and its workers run with the caller's signal mask."""

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
    loaded = load_flow(_POOLED_ASK_FLOW)

    def controller(stop_event, on_event, ask_user, render):
        return run_cli._run_flow_controlled(
            loaded,
            {},
            num_workers=1,
            stop_event=stop_event,
            tools={"ask_user": ask_user},
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

    _CONCURRENT_ASK_BARRIER.reset()
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
    loaded = load_flow(_POOLED_CONCURRENT_ASK_FLOW)

    def controller(stop_event, on_event, ask_user, render):
        return run_cli._run_flow_controlled(
            loaded,
            {},
            num_workers=2,
            stop_event=stop_event,
            tools={"ask_user": ask_user, "concurrent_ask": _concurrent_ask_tool(ask_user)},
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
        answer = ask_user(question="Continue?")
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


def test_chat_write_flow_approval_crosses_the_main_thread_broker(
    tmp_path, monkeypatch
) -> None:
    """A nested trusted prompt gates the tool while terminal work stays on main."""
    content = "id: g\nname: g\nnodes: {}\noutput: {}\n"
    prompt_threads: list[Thread] = []
    controller_threads: list[Thread] = []
    requests: list[PromptRequest] = []
    monkeypatch.setattr(
        chat_tools,
        "_approval_prompt_available",
        lambda: True,
        raising=False,
    )

    def prompt(request: PromptRequest) -> dict[str, str]:
        prompt_threads.append(current_thread())
        requests.append(request)
        return {"Decision": "Approve"}

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        chat_tools.set_workspace(tmp_path)
        with chat_tools._bind_ask_user(ask_user):
            return _success(chat_tools.write_flow("new.yaml", content))

    observed = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=prompt,
        reporter=None,
    )

    assert observed.output == "WROTE: new.yaml"
    assert (tmp_path / "new.yaml").read_text() == content
    assert prompt_threads == [main_thread()]
    assert controller_threads[0] is not main_thread()
    assert "Create new.yaml" in requests[0].question
    assert requests[0].questions is not None
    assert requests[0].questions[0]["header"] == "Decision"


def test_ctrl_c_during_active_prompt_preserves_keyboard_interrupt_and_joins() -> None:
    """Prompt Ctrl-C wakes the controller, which exits before the interrupt escapes."""

    cancellation_hook = Event()
    controller_threads: list[Thread] = []

    def prompt(request):
        request.set_cancel_callback(cancellation_hook.set)
        raise KeyboardInterrupt

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        ask_user(question="Blocking prompt")
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


def test_second_sigint_during_drain_waits_for_controller_before_returning() -> None:
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
        # Give the second SIGINT time to propagate; the helper must still wait for
        # the controller's cleanup to finish.
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

    assert registry_seen == [False]
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
    failed = RunResult(
        input={},
        status="failed",
        run_id="controller-test",
        log=None,
        failure=Failure("node.execute", "boom", "work"),
    )

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


def test_real_pooled_failure_wakes_prompt_and_joins_every_engine_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real fan-out failure drains its broker-blocked sibling without cause loss."""

    prompt_active = Event()
    monkeypatch.setattr(
        TerminalBroker,
        "_test_prompt_active",
        prompt_active,
        raising=False,
    )
    prompt_released = Event()
    loaded = load_flow(_POOLED_FAILURE_FLOW)

    def prompt(request):
        request.set_cancel_callback(prompt_released.set)
        prompt_active.set()
        assert prompt_released.wait(_TIMEOUT)
        return "late answer"

    def controller(stop_event, on_event, ask_user, render):
        return run_cli._run_flow_controlled(
            loaded,
            {},
            on_event=on_event,
            num_workers=2,
            stop_event=stop_event,
            tools={"ask_user": ask_user},
        )

    observed = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=prompt,
        reporter=None,
    )

    assert observed.status == "failed"
    assert observed.failure is not None
    assert observed.failure.message == "pooled primary failure"
    assert prompt_released.is_set()
    assert not any(
        thread.name.startswith(("ac-worker-", "ac-controller"))
        for thread in enumerate_threads()
    )
    assert "ask_user" not in TOOL_REGISTRY


def _trip_sigint_on_another_thread() -> None:
    """Trip SIGINT on a short-lived thread; the main thread runs the handler afterwards."""

    def trip() -> None:
        signal.pthread_kill(current_thread().ident, signal.SIGINT)

    tripper = Thread(target=trip, name="test-sigint-tripper")
    tripper.start()
    tripper.join()


class _SigintOnStopReporter(_Reporter):
    """A reporter whose stop() first takes a Ctrl-C tripped on another thread."""

    def stop(self) -> None:
        _trip_sigint_on_another_thread()
        super().stop()


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_sigint_on_the_controller_thread_during_drain_waits_for_cleanup() -> None:
    """A Ctrl-C tripped while the controller drains runs only after join and reporter stop."""

    expected = LookupError("observer failed")
    reporter = _Reporter(failure=expected)
    controller_threads: list[Thread] = []
    handled: list[list[str]] = []

    def caller_handler(signum, frame):
        # Snapshot cleanup progress when the caller's handler finally runs.
        handled.append(
            [name for name, _ in reporter.calls if name == "stop"]
            + [f"controller alive={controller_threads[0].is_alive()}"]
        )
        raise KeyboardInterrupt

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        assert on_event is not None
        on_event(NodeStarted("work"))
        assert stop_event.wait(_TIMEOUT)
        signal.pthread_kill(current_thread().ident, signal.SIGINT)
        time.sleep(0.05)
        return _success("stopped")

    previous = signal.signal(signal.SIGINT, caller_handler)
    try:
        with pytest.raises(LookupError) as caught:
            run_cli._run_with_terminal_controller(
                controller,
                prompt_handler=lambda request: "unused",
                reporter=reporter,
            )
        assert signal.getsignal(signal.SIGINT) is caller_handler
    finally:
        signal.signal(signal.SIGINT, previous)

    assert caught.value is expected
    assert handled == [["stop", "controller alive=False"]]
    assert not controller_threads[0].is_alive()


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_sigint_during_reporter_stop_lets_it_finish_and_keeps_the_first_cause(
    default_sigint_handler: None,
) -> None:
    """A Ctrl-C handled in reporter.stop() neither cuts it short nor replaces the cause."""

    expected = LookupError("observer failed")

    class _FailingSigintOnStopReporter(_SigintOnStopReporter):
        pass

    reporter = _FailingSigintOnStopReporter(failure=expected)
    controller_threads: list[Thread] = []

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        assert on_event is not None
        on_event(NodeStarted("work"))
        assert stop_event.wait(_TIMEOUT)
        return _success("stopped")

    with pytest.raises(LookupError) as caught:
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unused",
            reporter=reporter,
        )

    assert caught.value is expected
    assert [name for name, _ in reporter.calls if name == "stop"] == ["stop"]
    assert not controller_threads[0].is_alive()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_sigint_during_a_clean_drain_raises_after_cleanup(
    default_sigint_handler: None,
) -> None:
    """With no earlier cause, a Ctrl-C deferred in cleanup raises once cleanup is done."""

    reporter = _SigintOnStopReporter()

    with pytest.raises(KeyboardInterrupt):
        run_cli._run_with_terminal_controller(
            lambda stop_event, on_event, ask_user, render: _success("done"),
            prompt_handler=lambda request: "unused",
            reporter=reporter,
        )

    assert [name for name, _ in reporter.calls if name == "stop"] == ["stop"]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_sigint_during_a_clean_drain_stays_ignored_for_an_ignoring_caller() -> None:
    """A caller with SIGINT ignored gets the result when Ctrl-C lands in cleanup."""

    reporter = _SigintOnStopReporter()
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        observed = run_cli._run_with_terminal_controller(
            lambda stop_event, on_event, ask_user, render: _success("done"),
            prompt_handler=lambda request: "unused",
            reporter=reporter,
        )
        restored = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, previous)

    assert observed.output == "done"
    assert [name for name, _ in reporter.calls if name == "stop"] == ["stop"]
    assert restored == signal.SIG_IGN


def _helper_code_objects() -> set[types.CodeType]:
    """The helper's code object and every function nested in it."""

    codes = {run_cli._run_with_terminal_controller.__code__}
    pending = list(codes)
    while pending:
        for const in pending.pop().co_consts:
            if isinstance(const, types.CodeType) and const not in codes:
                codes.add(const)
                pending.append(const)
    return codes


def _run_with_sigint_at_opcode(index: int, *, first_cause: str) -> dict[str, Any]:
    """Run the helper, tripping SIGINT at the index-th helper opcode after tracing starts.

    Tracing starts as the first cause is raised: a Ctrl-C the controller sends to the
    main thread (``first_cause="sigint"``) or a render failure (``"render"``). With
    ``"none"`` the run completes, and tracing starts in a render action just before the
    controller returns. Opcodes are counted on the main thread, in the helper and the
    functions nested in it.
    """

    codes = _helper_code_objects()
    helper_code = run_cli._run_with_terminal_controller.__code__
    raised: list[BaseException] = []
    state: dict[str, Any] = {"seen": 0, "trip": None}

    def local_trace(frame, event, arg):
        if event == "opcode" and state["trip"] is None:
            if state["seen"] == index:
                helper_frame = frame
                while helper_frame.f_code is not helper_code:
                    helper_frame = helper_frame.f_back
                state["trip"] = {
                    # The caller's own handler is back: cleanup has finished.
                    "after_restore": signal.getsignal(signal.SIGINT) is caller_handler,
                    "deferring": helper_frame.f_locals["defer_sigint"],
                }
                signal.raise_signal(signal.SIGINT)
            state["seen"] += 1
        return local_trace

    def global_trace(frame, event, arg):
        if frame.f_code in codes:
            frame.f_trace_opcodes = True
            return local_trace
        return None

    def start_tracing() -> None:
        # Tracing from the start would let the first Ctrl-C run inside the trace
        # function, which turns tracing off.
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code is helper_code:
                frame.f_trace = local_trace
                frame.f_trace_opcodes = True
            frame = frame.f_back
        sys.settrace(global_trace)

    def caller_handler(signum, frame):
        exc = KeyboardInterrupt()
        if not raised:
            start_tracing()
        raised.append(exc)
        raise exc

    expected = LookupError("observer failed")

    class _TracingReporter(_Reporter):
        def handle(self, event: Any) -> None:
            start_tracing()
            super().handle(event)

    reporter = _TracingReporter(failure=expected if first_cause == "render" else None)
    controller_threads: list[Thread] = []
    saw_stop: list[bool] = []
    main_ident = main_thread().ident

    def controller(stop_event, on_event, ask_user, render):
        controller_threads.append(current_thread())
        if first_cause == "none":
            render(start_tracing)
            return _success("done")
        if first_cause == "render":
            on_event(NodeStarted("work"))
        else:
            signal.pthread_kill(main_ident, signal.SIGINT)
        saw_stop.append(stop_event.wait(_TIMEOUT))
        return _success("stopped")

    previous_handler = signal.signal(signal.SIGINT, caller_handler)
    previous_trace = sys.gettrace()
    outcome: Any = None
    try:
        outcome = run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unused",
            reporter=reporter,
        )
    except BaseException as exc:
        outcome = exc
    finally:
        sys.settrace(previous_trace)
        restored = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, previous_handler)
    if first_cause == "render":
        first: Any = expected
    elif first_cause == "sigint":
        first = raised[0] if raised else None
    else:
        first = None
    return {
        "trip": state["trip"],
        "outcome": outcome,
        "first": first,
        "joined": not controller_threads[0].is_alive(),
        "saw_stop": saw_stop,
        "reporter_stops": [name for name, _ in reporter.calls].count("stop"),
        "restored": restored is caller_handler,
    }


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
@pytest.mark.skipif(sys.gettrace() is not None, reason="replaces the active tracer")
@pytest.mark.parametrize("first_cause", ["sigint", "render", "none"])
def test_sigint_at_any_cleanup_opcode_still_joins_and_stops(first_cause: str) -> None:
    """A Ctrl-C at any helper bytecode after the first cause never cuts cleanup short."""

    failures = []
    index = 0
    while True:
        result = _run_with_sigint_at_opcode(index, first_cause=first_cause)
        trip = result["trip"]
        if trip is None:
            break
        outcome = result["outcome"]
        if trip["after_restore"]:
            # A fresh Ctrl-C after cleanup finished may replace the cause.
            kept = True
        elif first_cause == "none":
            # This Ctrl-C is the first one: it raises, now or after cleanup.
            kept = isinstance(outcome, KeyboardInterrupt) or outcome.output == "done"
        elif first_cause == "render" and not trip["deferring"]:
            # Handled before the helper's except turned deferral on: it raises once
            # through the caller's handler, as if it came just before the failure.
            kept = outcome is result["first"] or isinstance(outcome, KeyboardInterrupt)
        else:
            kept = outcome is result["first"]
        ok = (
            result["joined"]
            and result["saw_stop"] == ([] if first_cause == "none" else [True])
            and result["reporter_stops"] == 1
            and result["restored"]
            and kept
        )
        if not ok:
            failures.append((index, {k: v for k, v in result.items() if k != "first"}))
        index += 1

    assert index > 20
    assert failures == []

@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_a_non_raising_caller_handler_gets_each_ctrl_c_while_serving() -> None:
    """A caller handler that doesn't raise keeps getting Ctrl-C while the run goes on."""

    calls: list[int] = []
    delivered = [Event(), Event()]
    main_ident = main_thread().ident

    def caller_handler(signum, frame):
        calls.append(signum)
        delivered[len(calls) - 1].set()

    def controller(stop_event, on_event, ask_user, render):
        for event in delivered:
            signal.pthread_kill(main_ident, signal.SIGINT)
            assert event.wait(_TIMEOUT)
        return _success("done")

    previous = signal.signal(signal.SIGINT, caller_handler)
    try:
        observed = run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: "unused",
            reporter=None,
        )
        restored = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, previous)

    assert observed.output == "done"
    assert calls == [signal.SIGINT, signal.SIGINT]
    assert restored is caller_handler
