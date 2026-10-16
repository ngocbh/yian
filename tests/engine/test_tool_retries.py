"""TOOL `retries:`: backoff between attempts, recorded steps, and which errors retry."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import Field

import yian.nodes.tool.node as tool_node
from yian.compose import load_flow
from yian.nodes.base import _HostStop
from yian.nodes.contracts import NodeSteps
from yian.nodes.tool import ToolAttemptsFailed, ToolNode
from yian.run import _run_flow_controlled, run_flow
from yian.tools import FunctionTool, PermanentToolError, ToolSet, TransientToolError
from yian.tools.context import ToolContext


def _flow(retries: int | None, *, extra: str = "") -> Any:
    """Load a one-TOOL flow calling `flaky`, with `retries:` when given."""
    retries_line = "" if retries is None else f"    retries: {retries}\n"
    return load_flow(
        "id: t\nname: t\nnodes:\n  call:\n    kind: tool\n    tool_id: flaky\n"
        f"{retries_line}{extra}output: ${{call.output}}\n"
    )


class _Flaky:
    """A tool raising the scripted errors in order, then returning `"ok"`."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.seen: list[ToolContext] = []

    def tool(self):
        """Return the tool function, which records each call's context."""

        def flaky(ctx: ToolContext) -> str:
            """Do one flaky thing."""
            self.seen.append(ctx)
            if self.errors:
                raise self.errors.pop(0)
            return "ok"

        return flaky


@pytest.fixture
def delays(monkeypatch) -> list[float]:
    """Record every backoff wait instead of sleeping; jitter returns 1.0."""
    waited: list[float] = []
    monkeypatch.setattr(tool_node, "_sleep", lambda stop_event, seconds: waited.append(seconds))
    monkeypatch.setattr(tool_node, "_jitter", lambda: 1.0)
    return waited


def _steps(log: Path) -> list[dict[str, Any]]:
    """Return the node steps recorded in a durable run log, in order."""
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return [record["payload"]["step"] for record in records if record["kind"] == "node_step"]


def _kinds(log: Path) -> list[str]:
    """Return the record kinds in a durable run log, in order."""
    return [json.loads(line)["kind"] for line in log.read_text(encoding="utf-8").splitlines()]


def test_retries_zero_fails_as_before_and_records_no_step(tmp_path, delays):
    log = tmp_path / "run.jsonl"
    flaky = _Flaky(TimeoutError("a"))

    result = run_flow(_flow(None), {}, log=log, tools={"flaky": flaky.tool()})

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.message == "a"
    assert result.failure.cause_type == "TimeoutError"
    assert len(flaky.seen) == 1
    assert _steps(log) == []
    assert delays == []


@pytest.mark.parametrize("num_workers", [0, 8])
def test_two_failures_then_success_records_each_attempt(tmp_path, delays, num_workers):
    log = tmp_path / "run.jsonl"
    flaky = _Flaky(TimeoutError("a"), TimeoutError("b"))

    result = run_flow(
        _flow(2), {}, log=log, tools={"flaky": flaky.tool()}, run_id="r1", num_workers=num_workers
    )

    assert result.status == "succeeded", result.failure
    assert result.output == "ok"
    assert [ctx.attempt for ctx in flaky.seen] == [1, 2, 3]
    assert {ctx.call_id for ctx in flaky.seen} == {"r1/call"}
    assert delays == [1.0, 2.0]
    assert _steps(log) == [
        {"event": "failed", "attempt": 1,
         "error": {"type": "TimeoutError", "message": "a"}, "retry_in": 1.0},
        {"event": "started", "attempt": 2},
        {"event": "failed", "attempt": 2,
         "error": {"type": "TimeoutError", "message": "b"}, "retry_in": 2.0},
        {"event": "started", "attempt": 3},
    ]


def test_jitter_draws_from_half_to_the_full_delay(monkeypatch, delays):
    monkeypatch.setattr(tool_node, "_jitter", lambda: 0.0)
    flaky = _Flaky(TimeoutError("a"), TimeoutError("b"))

    result = run_flow(_flow(2), {}, tools={"flaky": flaky.tool()})

    assert result.status == "succeeded", result.failure
    assert delays == [0.5, 1.0]


def test_backoff_stops_at_the_cap(set_defaults, delays):
    set_defaults(TOOL_RETRY_DELAY_SECONDS=10, TOOL_RETRY_MAX_DELAY_SECONDS=15)
    flaky = _Flaky(TimeoutError("a"), TimeoutError("b"), TimeoutError("c"))

    result = run_flow(_flow(3), {}, tools={"flaky": flaky.tool()})

    assert result.status == "succeeded", result.failure
    assert delays == [10, 15, 15]


@pytest.mark.parametrize(("retry_after", "waited"), [(7, 7), (100, 30)])
def test_retry_after_replaces_the_backoff_up_to_the_cap(monkeypatch, delays, retry_after, waited):
    monkeypatch.setattr(tool_node, "_jitter", lambda: 0.0)
    flaky = _Flaky(TransientToolError("busy", retry_after=retry_after))

    result = run_flow(_flow(1), {}, tools={"flaky": flaky.tool()})

    assert result.status == "succeeded", result.failure
    assert delays == [waited]


def test_permanent_error_on_the_first_attempt_fails_unchanged(delays):
    flaky = _Flaky(PermanentToolError("no"))

    result = run_flow(_flow(3), {}, tools={"flaky": flaky.tool()})

    assert result.status == "failed"
    assert result.failure.message == "no"
    assert result.failure.cause_type == "PermanentToolError"
    assert len(flaky.seen) == 1
    assert delays == []


def test_permanent_error_on_the_second_attempt_stops_the_retries(delays):
    flaky = _Flaky(TimeoutError("a"), PermanentToolError("no"))

    result = run_flow(_flow(3), {}, tools={"flaky": flaky.tool()})

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.message == (
        "tool 'flaky' failed after 2 attempts: "
        "attempt 1: TimeoutError: a; attempt 2: PermanentToolError: no"
    )
    assert len(flaky.seen) == 2
    assert delays == [1.0]


def test_an_error_both_permanent_and_transient_is_not_retried(delays):
    class Both(PermanentToolError, TransientToolError):
        pass

    flaky = _Flaky(Both("both"))

    result = run_flow(_flow(3), {}, tools={"flaky": flaky.tool()})

    assert result.status == "failed"
    assert result.failure.cause_type == "Both"
    assert len(flaky.seen) == 1


def test_an_invalid_argument_is_not_retried(delays):
    calls: list[int] = []

    def flaky(n: Annotated[int, Field(gt=0)]) -> str:
        """Count one call.

        Args:
            n: A positive number.
        """
        calls.append(n)
        return "ok"

    flow = load_flow(
        "id: t\nname: t\ninput:\n  n: !int\nnodes:\n  call:\n    kind: tool\n"
        "    tool_id: flaky\n    retries: 3\n    input:\n      n: ${input.n}\n"
        "output: ${call.output}\n"
    )

    result = run_flow(flow, {"n": -1}, tools={"flaky": flaky})

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.cause_type == "ValidationError"
    assert calls == []
    assert delays == []


def test_an_unknown_argument_is_not_retried(delays):
    flaky = _Flaky()
    node = ToolNode("call", tool_id="flaky", retries=3)

    with pytest.raises(ValueError, match="unsupported argument"):
        node.forward({"bogus": 1}, tools=ToolSet({"flaky": flaky.tool()}))

    assert flaky.seen == []
    assert delays == []


def test_validate_checks_arguments_without_calling_the_tool():
    flaky = _Flaky()
    tools = ToolSet({"flaky": flaky.tool()})

    tools.validate("flaky", {})
    with pytest.raises(ValueError, match="unsupported argument"):
        tools.validate("flaky", {"bogus": 1})

    assert flaky.seen == []


def test_an_output_contract_failure_is_not_retried(delays):
    calls: list[int] = []

    def flaky() -> Any:
        """Return the wrong type."""
        calls.append(1)
        return "bad"

    result = run_flow(_flow(3, extra="    output: int\n"), {}, tools={"flaky": flaky})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert calls == [1]
    assert delays == []


def test_an_awaitable_from_a_function_tool_is_not_retried(delays):
    calls: list[int] = []

    async def later() -> str:
        return "x"

    def on_invoke(ctx, args):
        calls.append(1)
        return later()

    tool = FunctionTool(
        name="flaky",
        description="Do one flaky thing.",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke=on_invoke,
    )

    result = run_flow(_flow(3), {}, tools={"flaky": tool})

    assert result.status == "failed"
    assert result.failure.cause_type == "ToolCheckError"
    assert "awaitable" in result.failure.message
    assert calls == [1]
    assert delays == []


def test_exhausted_retries_list_every_attempt(delays):
    flaky = _Flaky(TimeoutError("a"), TimeoutError("b"))

    result = run_flow(_flow(1), {}, tools={"flaky": flaky.tool()})

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.cause_type == "ToolAttemptsFailed"
    assert result.failure.message == (
        "tool 'flaky' failed after 2 attempts: attempt 1: TimeoutError: a; "
        "attempt 2: TimeoutError: b"
    )
    assert len(flaky.seen) == 2


@pytest.mark.parametrize("num_workers", [0, 8])
def test_a_stop_during_the_backoff_wait_stops_the_run(tmp_path, monkeypatch, num_workers):
    log = tmp_path / "run.jsonl"
    stop = threading.Event()

    def sleep(stop_event, seconds):
        stop_event.set()
        raise _HostStop(primary=False)

    monkeypatch.setattr(tool_node, "_sleep", sleep)
    flaky = _Flaky(TimeoutError("a"))

    with pytest.raises(_HostStop):
        _run_flow_controlled(
            _flow(2), {}, log=log, tools={"flaky": flaky.tool()}, num_workers=num_workers,
            stop_event=stop,
        )

    assert stop.is_set()
    assert len(flaky.seen) == 1
    assert [step["event"] for step in _steps(log)] == ["failed"]
    assert "node_failed" not in _kinds(log)


class _BadStr(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no text")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (_BadStr(), "<_BadStr object>"),
        (TimeoutError("a\ud800b"), "a?b"),
        (TimeoutError("x" * 10_000), "x" * 199 + "…"),
    ],
)
def test_odd_error_text_is_recorded_safely(tmp_path, delays, error, message):
    log = tmp_path / "run.jsonl"
    flaky = _Flaky(error)

    result = run_flow(_flow(1), {}, log=log, tools={"flaky": flaky.tool()})

    assert result.status == "succeeded", result.failure
    [failed, _started] = _steps(log)
    assert failed["error"] == {"type": type(error).__name__, "message": message}


def test_forward_without_steps_retries_in_memory(delays):
    flaky = _Flaky(TimeoutError("a"), TimeoutError("b"))
    node = ToolNode("call", tool_id="flaky", retries=2)

    output = node.forward({}, tools=ToolSet({"flaky": flaky.tool()}))

    assert output.value == "ok"
    assert [ctx.attempt for ctx in flaky.seen] == [1, 2, 3]
    assert delays == [1.0, 2.0]


def test_a_resumed_start_continues_the_numbering_and_rewaits(delays):
    recorded: list[dict[str, Any]] = []
    prior = (
        {"event": "failed", "attempt": 1,
         "error": {"type": "TimeoutError", "message": "a"}, "retry_in": 1.0},
        {"event": "started", "attempt": 2},
        {"event": "failed", "attempt": 2,
         "error": {"type": "TimeoutError", "message": "b"}, "retry_in": 2.0},
    )
    # Starts: the first evaluation, a crash inside attempt 3's start, and this one.
    steps = NodeSteps(starts=3, prior=prior, record=recorded.append)
    flaky = _Flaky(TimeoutError("d"))
    node = ToolNode("call", tool_id="flaky", retries=2)

    with pytest.raises(ToolAttemptsFailed) as caught:
        node.forward({}, tools=ToolSet({"flaky": flaky.tool()}), steps=steps)

    assert [ctx.attempt for ctx in flaky.seen] == [4]
    assert delays == [2.0]
    assert recorded == []
    assert str(caught.value) == (
        "tool 'flaky' failed after 4 attempts: attempt 1: TimeoutError: a; "
        "attempt 2: TimeoutError: b; attempt 3: interrupted; attempt 4: TimeoutError: d"
    )
    assert isinstance(caught.value.__cause__, TimeoutError)


def test_a_host_stop_from_the_tool_is_neither_retried_nor_recorded(delays):
    recorded: list[dict[str, Any]] = []
    flaky = _Flaky(_HostStop(primary=False))
    node = ToolNode("call", tool_id="flaky", retries=2)

    with pytest.raises(_HostStop):
        node.forward(
            {}, tools=ToolSet({"flaky": flaky.tool()}),
            steps=NodeSteps(starts=1, prior=(), record=recorded.append),
        )

    assert len(flaky.seen) == 1
    assert recorded == []
    assert delays == []


def test_the_real_sleep_raises_at_once_on_a_set_event():
    event = threading.Event()
    event.set()

    with pytest.raises(_HostStop) as caught:
        tool_node._sleep(event, 60)

    assert caught.value.primary is False


def test_the_real_sleep_waits_without_an_event():
    started = time.monotonic()

    tool_node._sleep(None, 0.01)

    assert time.monotonic() - started >= 0.01


def test_a_huge_failure_count_gets_the_cap(monkeypatch):
    monkeypatch.setattr(tool_node, "_jitter", lambda: 1.0)

    assert tool_node._backoff(2000, TimeoutError()) == 30


@pytest.mark.parametrize(
    ("retries", "error"), [("2", TypeError), (True, TypeError), (1.0, TypeError), (-1, ValueError)]
)
def test_tool_node_rejects_a_bad_retries_count(retries, error):
    with pytest.raises(error, match="'call'"):
        ToolNode("call", tool_id="flaky", retries=retries)
