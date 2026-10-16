"""TOOL `retries:` across crashes, resume, and replay: attempts, call ids, and recorded steps."""

from __future__ import annotations

import copy
import json
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

import yian.nodes.tool.node as tool_node
import yian.run as run_module
import yian.runtime.engine as engine_module
from yian.compose import load_flow
from yian.run import resume_flow, run_flow
from yian.runtime.log import BackwardFinishedRecord, JsonlRunLog, MemoryRunLog
from yian.tools import TransientToolError
from yian.tools.context import ToolContext

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# The flows that wrote the committed old logs, verbatim: resume needs the identical flow.
_OLD_TOOL_FLOW = """\
id: old_tool_crash
name: old_tool_crash
nodes:
  place:
    kind: tool
    tool_id: place
    input: {symbol: "ACME"}
output: ${place.output}
"""

_OLD_LOOP_FLOW = """\
id: old_loop_tool_crash
name: old_loop_tool_crash
defs:
  turn:
    input:
      count: !int
    nodes:
      place:
        kind: tool
        tool_id: place
        input: {n: "${input.count}"}
    output:
      count: ${input.count + 1}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      count: 0
    while: count < 3
    max: 5
output: ${loop.output}
"""


class _SimulatedCrash(BaseException):
    """Process-loss marker that bypasses node-failure conversion."""


def _flow(retries: int) -> Any:
    """Load a one-TOOL flow calling `x` with `retries:`."""
    return load_flow(
        "id: t\nname: t\nnodes:\n  call:\n    kind: tool\n    tool_id: x\n"
        f"    retries: {retries}\noutput: ${{call.output}}\n"
    )


class _Script:
    """A tool raising the scripted outcomes in order (an exception or a crash), then `"ok"`."""

    def __init__(self, *outcomes: BaseException) -> None:
        self.outcomes = list(outcomes)
        self.seen: list[ToolContext] = []

    def tools(self) -> dict[str, Any]:
        """Return the run's tool mapping with the scripted tool as `x`."""

        def x(ctx: ToolContext) -> str:
            """Do one scripted thing."""
            self.seen.append(ctx)
            if self.outcomes:
                raise self.outcomes.pop(0)
            return "ok"

        return {"x": x}


class _Clock:
    """A fake backoff wait: records each wait and raises a scripted crash on the chosen waits."""

    def __init__(self) -> None:
        self.waited: list[float] = []
        self.crash_on: set[int] = set()

    def sleep(self, stop_event: threading.Event | None, seconds: float) -> None:
        """Record one wait; raise `_SimulatedCrash` when its index is in `crash_on`."""
        self.waited.append(seconds)
        if len(self.waited) - 1 in self.crash_on:
            raise _SimulatedCrash("process lost while waiting")


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    """Replace the backoff wait with a fake clock; jitter returns 1.0."""
    fake = _Clock()
    monkeypatch.setattr(tool_node, "_sleep", fake.sleep)
    monkeypatch.setattr(tool_node, "_jitter", lambda: 1.0)
    return fake


def _records(log: Path) -> list[dict[str, Any]]:
    """Return the records of a durable run log, decoded, in order."""
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _steps(log: Path) -> list[dict[str, Any]]:
    """Return the node steps recorded in a durable run log, in order."""
    return [r["payload"]["step"] for r in _records(log) if r["kind"] == "node_step"]


def _node_steps(log: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return each recorded node step with its node id, in order."""
    return [
        (r["payload"]["node_id"], r["payload"]["step"])
        for r in _records(log) if r["kind"] == "node_step"
    ]


def _failed(
    attempt: int, message: str, retry_in: float, error_type: str = "TimeoutError"
) -> dict[str, Any]:
    """Return the recorded step of a failed attempt that raised `error_type`."""
    return {
        "event": "failed", "attempt": attempt,
        "error": {"type": error_type, "message": message}, "retry_in": retry_in,
    }


def _started(attempt: int) -> dict[str, Any]:
    """Return the recorded step of a retried attempt's start."""
    return {"event": "started", "attempt": attempt}


def _replay(flow: Any, log: Path, length: int | None = None):
    """Replay a durable log (or its first `length` records) and return the reducer state."""
    with JsonlRunLog.resume(log) as durable:
        if length is None:
            return run_module._replay_run_log(flow, durable)
        prefix = MemoryRunLog(durable.run_id)
        for envelope in tuple(durable)[:length]:
            prefix.append(envelope.record)
        return run_module._replay_run_log(flow, prefix)


def _capture_live(monkeypatch) -> list[tuple[dict, dict, tuple]]:
    """Record the live state's node starts, node steps, and in-flight ids after every commit."""
    snapshots: list[tuple[dict, dict, tuple]] = []
    original = engine_module.commit_prepared

    def capture(log, state, prepared):
        observers = original(log, state, prepared)
        snapshots.append(
            (dict(state.node_starts), copy.deepcopy(state.node_steps), tuple(state.in_flight))
        )
        return observers

    monkeypatch.setattr(engine_module, "commit_prepared", capture)
    return snapshots


def test_a_crash_inside_attempt_two_resumes_at_attempt_three_with_one_retry_used(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(3)
    script = _Script(TimeoutError("a"), _SimulatedCrash("process lost"))

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=0)

    assert [ctx.attempt for ctx in script.seen] == [1, 2]
    assert _steps(log) == [_failed(1, "a", 1.0), _started(2)]

    # Failures used = 1, so two retries remain: attempts 3 and 4 fail, attempt 5 succeeds.
    script.outcomes = [TimeoutError("c"), TimeoutError("d")]
    result = resume_flow(flow, log=log, tools=script.tools())

    assert result.status == "succeeded", result.failure
    assert [ctx.attempt for ctx in script.seen] == [1, 2, 3, 4, 5]
    assert {ctx.call_id for ctx in script.seen} == {"r1/call"}
    assert clock.waited == [1.0, 2.0, 4.0]
    assert _steps(log) == [
        _failed(1, "a", 1.0), _started(2),
        _failed(3, "c", 2.0), _started(4),
        _failed(4, "d", 4.0), _started(5),
    ]


def test_a_crash_inside_attempt_two_with_no_retry_left_fails_on_resume(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(1)
    script = _Script(TimeoutError("a"), _SimulatedCrash("process lost"))

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=0)

    script.outcomes = [TimeoutError("c")]
    result = resume_flow(flow, log=log, tools=script.tools())

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.message == (
        "tool 'x' failed after 3 attempts: attempt 1: TimeoutError: a; "
        "attempt 2: interrupted; attempt 3: TimeoutError: c"
    )


def test_a_crash_during_the_backoff_wait_rewaits_and_resumes_at_attempt_two(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(3)
    # `retry_after` makes the recorded delay 7, where a recomputed backoff would be 1.0.
    script = _Script(TransientToolError("a", retry_after=7))
    clock.crash_on = {0}

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=0)

    assert _steps(log) == [_failed(1, "a", 7, "TransientToolError")]

    result = resume_flow(flow, log=log, tools=script.tools())

    assert result.status == "succeeded", result.failure
    assert [ctx.attempt for ctx in script.seen] == [1, 2]
    assert {ctx.call_id for ctx in script.seen} == {"r1/call"}
    assert clock.waited == [7, 7]
    # The resumed attempt 2 is the node's own start, so it records no `started` step.
    assert _steps(log) == [_failed(1, "a", 7, "TransientToolError")]


def test_a_crash_during_the_second_backoff_rewaits_the_last_recorded_delay(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(3)
    # Recorded delays 7 then 5; a recomputed second backoff would be 2.0.
    script = _Script(
        TransientToolError("a", retry_after=7), TransientToolError("b", retry_after=5)
    )
    clock.crash_on = {1}

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=0)

    result = resume_flow(flow, log=log, tools=script.tools())

    assert result.status == "succeeded", result.failure
    assert [ctx.attempt for ctx in script.seen] == [1, 2, 3]
    assert {ctx.call_id for ctx in script.seen} == {"r1/call"}
    assert clock.waited == [7, 5, 5]
    assert _steps(log) == [
        _failed(1, "a", 7, "TransientToolError"), _started(2),
        _failed(2, "b", 5, "TransientToolError"),
    ]


def test_a_retries_zero_crash_resumes_at_attempt_two_without_steps(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(0)
    script = _Script(_SimulatedCrash("process lost"))

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=0)

    result = resume_flow(flow, log=log, tools=script.tools())

    assert result.status == "succeeded", result.failure
    assert [ctx.attempt for ctx in script.seen] == [1, 2]
    assert {ctx.call_id for ctx in script.seen} == {"r1/call"}
    assert _steps(log) == []
    assert clock.waited == []


def test_a_finished_log_with_retries_replays_to_the_live_steps(tmp_path, monkeypatch, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(2)
    script = _Script(TimeoutError("a"), TimeoutError("b"))
    live = _capture_live(monkeypatch)

    result = run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=0)

    assert result.status == "succeeded", result.failure
    records = _records(log)
    # `RunStarted` is written before the engine runs; every later record is one live commit.
    assert len(live) == len(records) - 1
    # Every prefix replays to the live node starts and steps after that commit.
    for length, (starts, steps, in_flight) in enumerate(live, start=2):
        replayed = _replay(flow, log, length)
        assert replayed.node_starts == starts, length
        assert replayed.node_steps == steps, length
        assert tuple(replayed.in_flight) == in_flight, length
    assert any(steps.get("call") for _, steps, _ in live)
    assert live[-1][:2] == ({}, {})

    before = log.read_bytes()
    first = resume_flow(flow, log=log, tools=script.tools())
    after_first = log.read_bytes()
    second = resume_flow(flow, log=log, tools=script.tools())

    assert first.status == second.status == "succeeded"
    assert first.output == second.output == "ok"
    assert after_first == before
    assert log.read_bytes() == after_first
    assert len(script.seen) == 3
    assert _replay(flow, log).node_steps == {}


def test_two_crashes_rewait_the_recorded_delay_each_resume(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(3)
    script = _Script(TransientToolError("a", retry_after=7))
    clock.crash_on = {0}

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=0)

    script.outcomes = [_SimulatedCrash("process lost")]
    with pytest.raises(_SimulatedCrash):
        resume_flow(flow, log=log, tools=script.tools())

    assert [ctx.attempt for ctx in script.seen] == [1, 2]
    assert clock.waited == [7, 7]

    result = resume_flow(flow, log=log, tools=script.tools())

    assert result.status == "succeeded", result.failure
    assert [ctx.attempt for ctx in script.seen] == [1, 2, 3]
    assert {ctx.call_id for ctx in script.seen} == {"r1/call"}
    # The first wait, then the recorded delay again on each of the two resumes.
    assert clock.waited == [7, 7, 7]
    assert _steps(log) == [_failed(1, "a", 7, "TransientToolError")]


@pytest.mark.parametrize(("first", "then"), [(8, 0), (0, 8)], ids=["pooled-then-serial",
                                                                  "serial-then-pooled"])
def test_a_crash_mid_retry_resumes_across_worker_modes(tmp_path, clock, first, then):
    log = tmp_path / "run.jsonl"
    flow = _flow(3)
    script = _Script(TimeoutError("a"), _SimulatedCrash("process lost"))

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=first)

    script.outcomes = [TimeoutError("c")]
    result = resume_flow(flow, log=log, tools=script.tools(), num_workers=then)

    assert result.status == "succeeded", result.failure
    assert [ctx.attempt for ctx in script.seen] == [1, 2, 3, 4]
    assert {ctx.call_id for ctx in script.seen} == {"r1/call"}
    assert clock.waited == [1.0, 2.0]
    assert _steps(log) == [_failed(1, "a", 1.0), _started(2), _failed(3, "c", 2.0), _started(4)]


def test_an_interrupted_first_attempt_shows_in_the_final_message(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(1)
    script = _Script(_SimulatedCrash("process lost"))

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), num_workers=0)

    script.outcomes = [TimeoutError("b"), TimeoutError("c")]
    result = resume_flow(flow, log=log, tools=script.tools())

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.cause_type == "ToolAttemptsFailed"
    assert result.failure.message == (
        "tool 'x' failed after 3 attempts: attempt 1: interrupted; "
        "attempt 2: TimeoutError: b; attempt 3: TimeoutError: c"
    )
    assert [ctx.attempt for ctx in script.seen] == [1, 2, 3]
    assert clock.waited == [1.0]


def test_a_learnable_run_with_a_retried_tool_replays_and_runs_backward(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = _flow(2)
    script = _Script(TimeoutError("a"))

    result = run_flow(flow, {}, log=log, run_id="r1", tools=script.tools(), learnable=True)

    assert result.status == "succeeded", result.failure
    assert _steps(log) == [_failed(1, "a", 1.0), _started(2)]

    resumed = resume_flow(flow, log=log, tools=script.tools())

    assert resumed.status == "succeeded"
    assert resumed.output == "ok"
    assert len(script.seen) == 2
    outcome = resumed.backward("Review.")
    assert outcome.status == "succeeded"
    with JsonlRunLog.resume(log) as durable:
        assert type(tuple(durable)[-1].record) is BackwardFinishedRecord


_MAP_FLOW = """\
id: map_tools
name: map_tools
defs:
  one:
    input:
      n: !int
    nodes:
      place:
        kind: tool
        tool_id: x
        retries: 3
        input: {n: "${input.n}"}
    output: ${place.output}
nodes:
  each:
    kind: map
    call: one
    over: ${[0, 1]}
    input:
      n: ${item}
output: ${each.output}
"""


def test_two_tools_retrying_at_once_in_a_pooled_map_resume_with_their_own_steps(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    flow = load_flow(_MAP_FLOW)
    seen: list[tuple[int, ToolContext]] = []
    together = threading.Barrier(2, timeout=10)
    crashed = [False]

    def x(ctx: ToolContext, n: int) -> str:
        """Fail attempt 1, crash in attempt 2 once, then fail once more before succeeding."""
        seen.append((n, ctx))
        if ctx.attempt == 1:
            together.wait()
            raise TimeoutError(f"a{n}")
        if ctx.attempt == 2 and not crashed[0]:
            # Both nodes have recorded their failure and the retry's start.
            together.wait()
            raise _SimulatedCrash("process lost")
        if ctx.attempt == 3:
            raise TimeoutError(f"c{n}")
        return f"ok{n}"

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, run_id="r1", tools={"x": x}, num_workers=4)

    crashed[0] = True
    by_node = {}
    for node_id, step in _node_steps(log):
        by_node.setdefault(node_id, []).append(step)
    ids = {n: ctx.node for n, ctx in seen if ctx.attempt == 1}
    assert len(set(ids.values())) == 2
    for n, node_id in ids.items():
        assert by_node[node_id] == [_failed(1, f"a{n}", 1.0), _started(2)]

    seen.clear()
    result = resume_flow(flow, log=log, tools={"x": x}, num_workers=4)

    assert result.status == "succeeded", result.failure
    assert result.output == ["ok0", "ok1"]
    for n, node_id in ids.items():
        mine = [ctx for m, ctx in seen if m == n]
        assert [ctx.attempt for ctx in mine] == [3, 4]
        assert {ctx.call_id for ctx in mine} == {f"r1/{node_id}"}
        assert {ctx.node for ctx in mine} == {node_id}
    by_node = {}
    for node_id, step in _node_steps(log):
        by_node.setdefault(node_id, []).append(step)
    for n, node_id in ids.items():
        assert by_node[node_id] == [
            _failed(1, f"a{n}", 1.0), _started(2), _failed(3, f"c{n}", 2.0), _started(4),
        ]


_LOOP_FLOW = """\
id: loop_tools
name: loop_tools
defs:
  turn:
    input:
      count: !int
    nodes:
      place:
        kind: tool
        tool_id: x
        retries: 1
        input: {n: "${input.count}"}
    output:
      count: ${input.count + 1}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      count: 0
    while: count < 50
    max: 60
output: ${loop.output}
"""


def test_a_long_loop_keeps_no_steps_for_finished_iterations(tmp_path, monkeypatch, clock):
    log = tmp_path / "run.jsonl"
    flow = load_flow(_LOOP_FLOW)
    calls: list[tuple[int, int]] = []

    def x(ctx: ToolContext, n: int) -> str:
        """Fail each iteration's first attempt."""
        calls.append((n, ctx.attempt))
        if ctx.attempt == 1:
            raise TimeoutError(f"a{n}")
        return "ok"

    live = _capture_live(monkeypatch)
    result = run_flow(flow, {}, log=log, run_id="r1", tools={"x": x}, num_workers=0)

    assert result.status == "succeeded", result.failure
    assert result.output == {"count": 50}
    assert calls == [(n, attempt) for n in range(50) for attempt in (1, 2)]
    assert len(_steps(log)) == 100
    # Live: only in-flight nodes hold a start count or steps, so a finished
    # iteration leaves nothing behind.
    for starts, steps, in_flight in live:
        assert set(starts) <= set(in_flight)
        assert set(steps) <= set(in_flight)
        assert all(len(recorded) <= 2 for recorded in steps.values())
    assert live[-1][:2] == ({}, {})

    replayed = _replay(flow, log)
    assert replayed.node_starts == {}
    assert replayed.node_steps == {}
    # Replay of a prefix ending inside an iteration holds that iteration's steps only.
    records = _records(log)
    step_prefixes = [i + 1 for i, r in enumerate(records) if r["kind"] == "node_step"]
    for length in step_prefixes[::10]:
        starts, steps, _ = live[length - 2]
        state = _replay(flow, log, length)
        assert state.node_starts == starts
        assert state.node_steps == steps
        assert len(state.node_steps) == 1


def test_an_old_tool_crash_log_resumes_at_attempt_two(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    shutil.copyfile(_FIXTURES / "old_tool_crash.jsonl", log)
    seen: list[ToolContext] = []

    def place(ctx: ToolContext, symbol: str) -> str:
        """Place an order."""
        seen.append(ctx)
        return f"placed {symbol}"

    result = resume_flow(load_flow(_OLD_TOOL_FLOW), log=log, tools={"place": place})

    assert result.status == "succeeded", result.failure
    assert result.output == "placed ACME"
    assert [ctx.attempt for ctx in seen] == [2]
    assert [ctx.call_id for ctx in seen] == ["old-tool-crash/place"]
    assert _steps(log) == []
    assert clock.waited == []


def test_an_old_loop_tool_crash_log_resumes_with_its_growth(tmp_path, clock):
    log = tmp_path / "run.jsonl"
    shutil.copyfile(_FIXTURES / "old_loop_tool_crash.jsonl", log)
    seen: list[tuple[int, ToolContext]] = []

    def place(ctx: ToolContext, n: int) -> str:
        """Place an order."""
        seen.append((n, ctx))
        return "ok"

    result = resume_flow(load_flow(_OLD_LOOP_FLOW), log=log, tools={"place": place})

    assert result.status == "succeeded", result.failure
    assert result.output == {"count": 3}
    assert [(n, ctx.attempt) for n, ctx in seen] == [(1, 2), (2, 1)]
    assert seen[0][1].call_id == "old-loop-tool-crash/loop#1/place"
    assert _steps(log) == []
    assert clock.waited == []
