"""Final package-root run lifecycle over one authoritative append-only log.

These tests deliberately exercise the application boundary rather than the
reducer in isolation.  Fresh runs own either a process-local log or one durable
path; recovery always owns a durable writer from lock acquisition through the
structured engine drain.  Historical records reconstruct state silently and
only records committed in the current segment may produce callbacks/events.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from datetime import datetime, timezone
import inspect
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import yian._settings as settings_module
import yian.llm_clients as llm_clients_module
import yian.nodes.wait.node as wait_module
import yian.run as run_module
from yian.compile.llm_cascade import resolve_llm_cascade
from yian.compile.manifest import flow_fingerprint
from yian.compile.model import CompiledFlow, Edge, END_ID, START_ID
from yian.compile.origins import rebase_flow_origins
from yian.compose import ComposedFlow, load_flow
from yian.diagnostics import OriginRef
from yian.events import (
    NodeFailed,
    NodeStarted,
    NodeSucceeded,
    RunFailed,
    RunResumed,
    RunStarted,
    RunSucceeded,
)
from yian.nodes.base import _HostStop
from yian.nodes.code import CodeNode
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.runtime.log import (
    JsonlRunLog,
    MemoryRunLog,
    NodeFailedRecord,
    NodeRoutedRecord,
    NodeStartedRecord,
    NodeSucceededRecord,
    RunResumedRecord,
    RunStartedRecord,
)
from yian.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_run_started,
)
from yian.typesys import VariablePool
from yian.typesys.seeding import seed_system_clock


_RUN_ID = "lifecycle-run"
_ACTION_PROBE: list[str] = []
_ACTIVE_STOP: "_TrackingStopEvent | None" = None
_HOST_INTERRUPTION = _HostStop("host interrupted the run")
_HOST_INTERRUPTION_ARMED = True


def _record_action(inputs: dict[str, Any]) -> str:
    """Record one real CODE invocation for append-boundary assertions."""
    _ACTION_PROBE.append("ran")
    return "done"


def _complete_after_wait_started(inputs: dict[str, Any]) -> str:
    """Complete only after the sibling WAIT is blocked on the run stop event."""
    assert _ACTIVE_STOP is not None
    assert _ACTIVE_STOP.wait_entered.wait(timeout=2)
    return "triggered"


def _interrupt_after_wait_started(inputs: dict[str, Any]) -> str:
    """Interrupt the first attempt after WAIT blocks, then succeed on recovery."""
    global _HOST_INTERRUPTION_ARMED
    if not _HOST_INTERRUPTION_ARMED:
        return "recovered"
    assert _ACTIVE_STOP is not None
    assert _ACTIVE_STOP.wait_entered.wait(timeout=2)
    _HOST_INTERRUPTION_ARMED = False
    raise _HOST_INTERRUPTION


class _TrackingStopEvent(threading.Event):
    """Real Event subtype that proves cooperative WAIT drain completed."""

    def __init__(self) -> None:
        """Create one unset event with entry/release observations."""
        super().__init__()
        self.wait_entered = threading.Event()
        self.wait_released = threading.Event()
        self.set_calls = 0

    def set(self) -> None:
        """Publish lifecycle stop and wake every cooperative waiter."""
        self.set_calls += 1
        super().set()

    def wait(self, timeout: float | None = None) -> bool:
        """Record WAIT entry and successful cooperative release."""
        self.wait_entered.set()
        stopped = super().wait(timeout=timeout)
        if stopped:
            self.wait_released.set()
        return stopped


class _TrackingRunLog(MemoryRunLog):
    """A path-shaped memory log with injectable append failure and close tracking."""

    def __init__(
        self,
        run_id: str,
        path: Path,
        *,
        append_failure: BaseException | None = None,
        fail_when: Callable[[Any], bool] | None = None,
    ) -> None:
        """Create an open fake durable owner for one public-boundary test."""
        super().__init__(run_id)
        self._test_path = path
        self.append_failure = append_failure
        self.fail_when = fail_when or (lambda _record: False)
        self.close_calls = 0
        self.closed = False

    @property
    def path(self) -> Path:
        """Expose the requested durable path to the run boundary."""
        return self._test_path

    def append(self, record):
        """Fail the selected append before the in-memory authority changes."""
        if self.append_failure is not None and self.fail_when(record):
            raise self.append_failure
        return super().append(record)

    def close(self) -> None:
        """Record owner cleanup while retaining the readable prefix."""
        self.close_calls += 1
        self.closed = True
        super().close()


_SUCCESS_FLOW = """
id: lifecycle-success
name: lifecycle_success
input:
  n: int
nodes:
  twice:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.n}
    output: int
output: ${twice.output}
"""

_NULLABLE_LIST_FLOW = """
id: lifecycle_nullable_list
name: lifecycle_nullable_list
input:
  items: list[Optional[int]]
output: list[Optional[int]]
kind: code
code: |
  return inputs["items"]
"""

_FAILURE_FLOW = """
id: lifecycle-failure
name: lifecycle_failure
nodes:
  fail:
    kind: code
    code: tests.engine._compose_codefns:boom
    output: str
output: ${fail.output}
"""

_LOCATED_FAILURE_FLOW = """
id: lifecycle-located-failure
name: lifecycle_located_failure
input:
  window: int
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${input.window}
    output: int
output: ${echo.output}
"""

_TERMINAL_SKIP_FLOW = """
id: lifecycle-terminal-skip
name: lifecycle_terminal_skip
input:
  score: float
nodes:
  gate:
    kind: case
    cases:
      - when: "${input.score} >= 1.0"
        then: selected
  selected:
    kind: code
    code: tests.engine._compose_codefns:positive
    input:
      topic: selected
    output: str
output: ${selected.output}
"""


def _source_free_failure_flow() -> ComposedFlow:
    """Build one programmatic flow whose failure has no authored origins."""
    nodes = {
        START_ID: StartNode(START_ID),
        "fail": CodeNode("fail", code="tests.engine._compose_codefns:boom"),
        END_ID: EndNode.value(END_ID, output_name=None),
    }
    compiled = CompiledFlow.from_parts(
        nodes,
        [
            Edge("start-fail", START_ID, "fail"),
            Edge("fail-end", "fail", END_ID),
        ],
        wiring={},
    )
    return ComposedFlow(compiled=compiled)

_ACTION_FLOW = """
id: lifecycle-action
name: lifecycle_action
nodes:
  work:
    kind: code
    code: tests.engine.test_run_lifecycle:_record_action
    output: str
output: ${work.output}
"""

_PARALLEL_FLOW = """
id: lifecycle-parallel
name: lifecycle_parallel
input:
  n: int
nodes:
  a:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${input.n}
    output: int
  b:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${input.n}
    output: int
  c:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${input.n}
    output: int
output:
  a: ${a.output}
  b: ${b.output}
  c: ${c.output}
"""

_AGENT_FLOW = """
id: lifecycle-agent
name: lifecycle_agent
input:
  topic: str
nodes:
  answer:
    kind: agent
    input:
      topic: ${input.topic}
    prompt: Answer ${topic}
    output: str
output: ${answer.output}
"""

_CALLBACK_DRAIN_FLOW = """
id: lifecycle-callback-drain
name: lifecycle_callback_drain
nodes:
  trigger:
    kind: code
    code: tests.engine.test_run_lifecycle:_complete_after_wait_started
    output: str
  waiter:
    kind: wait
    until: "2099-01-01T00:00:00+00:00"
output:
  trigger: ${trigger.output}
  waiter: ${waiter.output}
"""

_HUMAN_DRAIN_FLOW = """
id: lifecycle-human-drain
name: lifecycle_human_drain
nodes:
  trigger:
    kind: code
    code: tests.engine.test_run_lifecycle:_complete_after_wait_started
    output: str
  human:
    kind: human_input
    prompt: "Approve?"
    output: str
output:
  trigger: ${trigger.output}
  human: ${human.output}
"""

_HOST_DRAIN_FLOW = """
id: lifecycle-host-drain
name: lifecycle_host_drain
nodes:
  interrupter:
    kind: code
    code: tests.engine.test_run_lifecycle:_interrupt_after_wait_started
    output: str
  waiter:
    kind: wait
    until: "2099-01-01T00:00:00+00:00"
output:
  interrupter: ${interrupter.output}
  waiter: ${waiter.output}
"""


def _assert_keyword_only(signature: inspect.Signature, names: tuple[str, ...]) -> None:
    """Require the named parameters to be keyword-only in their declared order."""
    for name in names:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def _durable_records(path: Path) -> tuple[Any, ...]:
    """Read one closed durable prefix while also proving its lock was released."""
    with JsonlRunLog.resume(path) as log:
        return tuple(envelope.record for envelope in log)


def _write_nonterminal_prefix(
    path: Path,
    loaded,
    inputs: dict[str, Any],
    *,
    started: tuple[str, ...],
    num_workers: int,
    effective: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Write RunStarted, complete START, then leave exact ordered work in flight."""
    private_flow = copy.deepcopy(loaded.compiled)
    pool = VariablePool(start_id=private_flow.start_id)
    pool.set(private_flow.start_id, copy.deepcopy(inputs))
    seed_system_clock(pool)
    pool.add_system("run_id", _RUN_ID)
    state = RunState(private_flow)

    with JsonlRunLog.create(path, _RUN_ID) as log:
        commit_prepared(
            log,
            state,
            prepare_run_started(
                state,
                RunStartData(
                    run_id=_RUN_ID,
                    flow_fingerprint=flow_fingerprint(loaded.compiled),
                    initial_pool=pool,
                    start_wiring_override={
                        name: f"${{input.{name}}}" for name in inputs
                    },
                    effective_llm_by_static_path=effective or {},
                    num_workers=num_workers,
                ),
            ),
        )
        commit_prepared(log, state, prepare_node_started(state, START_ID))
        commit_prepared(
            log,
            state,
            prepare_node_succeeded(state, START_ID, copy.deepcopy(inputs)),
        )
        for node_id in started:
            commit_prepared(log, state, prepare_node_started(state, node_id))
        return log.next_sequence


def _install_create_log(
    monkeypatch: pytest.MonkeyPatch,
    fake: _TrackingRunLog,
) -> None:
    """Make the public durable-create path return one inspectable protocol fake."""

    def create(cls, path, run_id):
        assert Path(path) == fake.path
        assert run_id == fake.run_id
        return fake

    monkeypatch.setattr(JsonlRunLog, "create", classmethod(create))


def test_final_run_boundary_signatures_and_result_fields_are_exact() -> None:
    """Remove checkpoint/abort arguments and keep stop control private."""
    fresh = inspect.signature(run_module.run_flow)
    recovered = inspect.signature(run_module.resume_flow)
    controlled_fresh = inspect.signature(run_module._run_flow_controlled)
    controlled_resume = inspect.signature(run_module._resume_flow_controlled)

    assert list(fresh.parameters) == [
        "flow",
        "inputs",
        "log",
        "run_id",
        "num_workers",
        "on_event",
        "on_stream",
        "llm_config",
        "memory",
        "learnable",
        "tools",
        "skills",
        "steering",
    ]
    _assert_keyword_only(
        fresh,
        (
            "log",
            "run_id",
            "num_workers",
            "on_event",
            "on_stream",
            "llm_config",
            "memory",
            "learnable",
            "tools",
            "skills",
            "steering",
        ),
    )
    assert fresh.parameters["log"].default is None
    assert fresh.parameters["run_id"].default is None
    assert fresh.parameters["num_workers"].default == 8
    assert fresh.parameters["memory"].default is None
    assert fresh.parameters["learnable"].default is False
    assert fresh.parameters["tools"].default is None
    assert fresh.parameters["skills"].default is None
    assert fresh.parameters["steering"].default is None
    assert "stop_event" not in fresh.parameters

    assert list(recovered.parameters) == [
        "flow",
        "log",
        "num_workers",
        "on_event",
        "on_stream",
        "memory",
        "tools",
        "skills",
        "steering",
        "api_key",
    ]
    _assert_keyword_only(
        recovered,
        ("log", "num_workers", "on_event", "on_stream", "memory", "tools", "skills", "steering",
         "api_key"),
    )
    assert recovered.parameters["log"].default is inspect.Parameter.empty
    assert recovered.parameters["num_workers"].default is None
    assert recovered.parameters["memory"].default is None
    assert recovered.parameters["tools"].default is None
    assert recovered.parameters["skills"].default is None
    assert recovered.parameters["steering"].default is None
    assert recovered.parameters["api_key"].default is None
    assert "stop_event" not in recovered.parameters

    assert list(controlled_fresh.parameters) == [
        *fresh.parameters,
        "stop_event",
    ]
    assert list(controlled_resume.parameters) == [
        *recovered.parameters,
        "stop_event",
    ]
    for signature in (controlled_fresh, controlled_resume):
        assert signature.parameters["stop_event"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["stop_event"].default is inspect.Parameter.empty

    assert [field.name for field in fields(run_module.RunResult)] == [
        "input",
        "status",
        "run_id",
        "log",
        "output",
        "failure",
        "events",
        "learning",
        "backward_outcome",
        "_backward_handle",
    ]


def test_fresh_memory_success_returns_current_segment_and_canonical_input() -> None:
    """Use the record/reducer path even when no durable path was requested."""
    seen = []

    result = run_module.run_flow(
        load_flow(_SUCCESS_FLOW),
        {"n": "4"},
        log=None,
        run_id=_RUN_ID,
        on_event=seen.append,
    )

    assert result.status == "succeeded"
    assert result.input == {"n": 4}
    assert result.output == 8
    assert result.failure is None
    assert result.run_id == _RUN_ID
    assert result.log is None
    assert result.events == seen
    assert isinstance(result.events[0], RunStarted)
    assert isinstance(result.events[-1], RunSucceeded)


def test_fresh_durable_success_returns_path_and_releases_the_lock(tmp_path: Path) -> None:
    """Close the boundary-owned writer after a successful structured drain."""
    path = tmp_path / "success.jsonl"

    result = run_module.run_flow(
        load_flow(_SUCCESS_FLOW),
        {"n": 5},
        log=path,
        run_id=_RUN_ID,
    )

    assert result.status == "succeeded"
    assert result.output == 10
    assert result.run_id == _RUN_ID
    assert result.log == path
    records = _durable_records(path)
    assert isinstance(records[0], RunStartedRecord)
    assert isinstance(records[-1], NodeSucceededRecord)


def test_nullable_specialized_list_survives_durable_run_and_terminal_resume(
    tmp_path: Path,
) -> None:
    """Retain nullable element typing across fresh execution and silent replay."""
    path = tmp_path / "nullable-list.jsonl"
    inputs = {"items": [1, None, 3]}

    fresh = run_module.run_flow(
        load_flow(_NULLABLE_LIST_FLOW),
        inputs,
        log=path,
        run_id=_RUN_ID,
    )

    assert fresh.status == "succeeded"
    assert fresh.output == [1, None, 3]
    records = _durable_records(path)
    success = next(
        record
        for record in records
        if isinstance(record, NodeSucceededRecord)
        and record.node_id == "lifecycle_nullable_list"
    )
    assert success == NodeSucceededRecord(
        node_id="lifecycle_nullable_list",
        commit_target="lifecycle_nullable_list",
        value={"kind": "list[int]", "value": [1, None, 3]},
    )
    before = path.read_bytes()

    resumed = run_module.resume_flow(
        load_flow(_NULLABLE_LIST_FLOW),
        log=path,
    )

    assert resumed.status == fresh.status
    assert resumed.output == fresh.output
    assert resumed.events == []
    assert path.read_bytes() == before


def test_fresh_memory_flow_failure_returns_durable_diagnostics() -> None:
    """Derive the same failed result shape from a process-local authority."""
    result = run_module.run_flow(
        load_flow(_FAILURE_FLOW),
        {},
        log=None,
        run_id=_RUN_ID,
    )

    assert result.status == "failed"
    assert result.input == {}
    assert result.output is None
    assert result.failure is not None
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "fail"
    assert result.failure.message == "boom"
    assert result.failure.traceback is not None
    assert "ValueError: boom" in result.failure.traceback
    assert result.run_id == _RUN_ID
    assert result.log is None
    node_failure = next(event for event in result.events if isinstance(event, NodeFailed))
    assert node_failure.failure is result.failure
    assert isinstance(result.events[-1], RunFailed)
    assert result.events[-1].failure is result.failure


def test_fresh_flow_failure_is_a_result_and_releases_the_lock(tmp_path: Path) -> None:
    """Keep node failure distinct from a host exception while closing its log."""
    path = tmp_path / "failed.jsonl"

    result = run_module.run_flow(
        load_flow(_FAILURE_FLOW),
        {},
        log=path,
        run_id=_RUN_ID,
    )

    assert result.status == "failed"
    assert result.input == {}
    assert result.output is None
    assert result.failure is not None
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "fail"
    assert result.failure.message == "boom"
    assert result.failure.traceback is not None
    assert "ValueError: boom" in result.failure.traceback
    assert result.run_id == _RUN_ID
    assert result.log == path
    node_failure = next(event for event in result.events if isinstance(event, NodeFailed))
    assert node_failure.failure is result.failure
    assert isinstance(result.events[-1], RunFailed)
    assert result.events[-1].failure is result.failure
    durable_failure = _durable_records(path)[-1]
    assert isinstance(durable_failure, NodeFailedRecord)
    assert durable_failure.failure == result.failure


@pytest.mark.parametrize(
    ("source", "inputs", "expected_status", "expected_output"),
    [
        (_SUCCESS_FLOW, {"n": 6}, "succeeded", 12),
        (_FAILURE_FLOW, {}, "failed", None),
    ],
    ids=["succeeded", "failed"],
)
def test_terminal_resume_is_read_only_silent_and_unlocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    inputs: dict[str, Any],
    expected_status: str,
    expected_output: Any,
) -> None:
    """Reconstruct a terminal result without appending or replaying observers."""
    path = tmp_path / f"terminal-{expected_status}.jsonl"
    fresh = run_module.run_flow(
        load_flow(source),
        inputs,
        log=path,
        run_id=_RUN_ID,
    )
    before = path.read_bytes()
    seen = []

    def forbidden_default() -> str:
        raise AssertionError("terminal recovery consulted an ambient LLM default")

    monkeypatch.setattr(run_module, "connection_snapshot", forbidden_default)
    monkeypatch.setattr(settings_module, "default_llm_provider", forbidden_default)
    monkeypatch.setattr(settings_module, "default_llm_model", forbidden_default)

    resumed = run_module.resume_flow(
        load_flow(source),
        log=path,
        on_event=seen.append,
    )

    assert resumed.status == expected_status == fresh.status
    assert resumed.output == expected_output == fresh.output
    assert resumed.failure == fresh.failure
    assert resumed.input == fresh.input
    assert resumed.run_id == _RUN_ID
    assert resumed.log == path
    assert resumed.events == []
    assert seen == []
    assert path.read_bytes() == before
    _durable_records(path)


def test_terminal_resume_preserves_compose_failure_and_diagnosis(tmp_path: Path) -> None:
    """Replay preserves the canonical Failure that Compose resolves after execution."""
    path = tmp_path / "terminal-located-failure.jsonl"
    fresh_flow = load_flow(_LOCATED_FAILURE_FLOW)
    fresh = run_module.run_flow(
        fresh_flow,
        {"window": "soon"},
        log=path,
        run_id=_RUN_ID,
    )
    before = path.read_bytes()

    assert fresh.status == "failed"
    assert fresh.failure is not None
    fresh_diagnostic = fresh_flow.diagnose(fresh.failure)
    assert fresh_diagnostic.frames

    resumed_flow = load_flow(_LOCATED_FAILURE_FLOW)
    resumed = run_module.resume_flow(
        resumed_flow,
        log=path,
    )

    assert resumed.status == "failed"
    assert resumed.input == fresh.input == {"window": "soon"}
    assert resumed.failure == fresh.failure
    assert resumed_flow.diagnose(resumed.failure) == fresh_diagnostic
    assert resumed.events == []
    assert path.read_bytes() == before
    _durable_records(path)


def test_terminal_resume_preserves_source_free_failure(tmp_path: Path) -> None:
    """A programmatic flow replays the same semantic failure with an empty trace."""
    path = tmp_path / "terminal-source-free-failure.jsonl"
    fresh_flow = _source_free_failure_flow()
    fresh = run_module.run_flow(fresh_flow, {}, log=path, run_id=_RUN_ID)
    before = path.read_bytes()

    assert fresh.status == "failed"
    assert fresh.failure is not None
    assert fresh.failure.trace == ()
    assert fresh_flow.diagnose(fresh.failure).frames == ()

    resumed_flow = _source_free_failure_flow()
    resumed = run_module.resume_flow(resumed_flow, log=path)

    assert resumed.failure == fresh.failure
    assert resumed_flow.diagnose(resumed.failure) == fresh_flow.diagnose(fresh.failure)
    assert resumed.events == []
    assert path.read_bytes() == before


def test_terminal_resume_preserves_recorded_trace_across_origin_rebase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay uses causal-record provenance, not the newly loaded graph's origins."""
    # A loaded flow no longer skips its END; rebuild the old graph shape to keep the
    # latched terminal failure's replay covered.
    monkeypatch.setattr(
        "yian.compose.build.pipeline.null_skipped_outputs", lambda outputs, model: outputs
    )
    path = tmp_path / "terminal-origin-rebase.jsonl"
    loaded = load_flow(_TERMINAL_SKIP_FLOW)
    fresh = run_module.run_flow(
        loaded,
        {"score": 0.0},
        log=path,
        run_id=_RUN_ID,
    )
    before = path.read_bytes()

    assert fresh.status == "failed"
    assert fresh.failure is not None
    assert fresh.failure.code == "graph.terminal_unreachable"
    assert fresh.failure.trace
    routed = [
        record
        for record in _durable_records(path)
        if isinstance(record, NodeRoutedRecord)
    ]
    assert len(routed) == 1
    assert routed[0].terminal_failure == fresh.failure

    rebased_compiled, resolver = rebase_flow_origins(
        loaded.compiled,
        lambda origin: OriginRef(f"rebased:{origin.key}"),
        resolver=loaded.origin_resolver,
    )
    rebased = ComposedFlow(
        compiled=rebased_compiled,
        version=loaded.version,
        name=loaded.name,
        description=loaded.description,
        origin_resolver=resolver,
    )
    assert flow_fingerprint(rebased.compiled) == flow_fingerprint(loaded.compiled)

    resumed = run_module.resume_flow(rebased, log=path)

    assert resumed.status == "failed"
    assert resumed.failure == fresh.failure
    assert resumed.failure.trace == fresh.failure.trace
    assert resumed.events == []
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("override", "expected_workers"),
    [(None, 2), (0, 0), (4, 4)],
    ids=["recorded", "serial-override", "pooled-override"],
)
def test_nonterminal_resume_commits_exact_retries_and_selected_workers(
    tmp_path: Path,
    override: int | None,
    expected_workers: int,
) -> None:
    """Retry ordered in-flight work before existing readiness and hide history."""
    path = tmp_path / f"resume-{expected_workers}.jsonl"
    loaded = load_flow(_PARALLEL_FLOW)
    prefix_length = _write_nonterminal_prefix(
        path,
        loaded,
        {"n": 7},
        started=("a", "b"),
        num_workers=2,
    )
    seen = []

    result = run_module.resume_flow(
        load_flow(_PARALLEL_FLOW),
        log=path,
        num_workers=override,
        on_event=seen.append,
    )

    assert result.status == "succeeded"
    assert result.input == {"n": 7}
    assert result.output == {"a": 7, "b": 7, "c": 7}
    assert result.run_id == _RUN_ID
    assert result.log == path
    assert result.events == seen
    assert isinstance(result.events[0], RunResumed)
    assert not any(isinstance(event, RunStarted) for event in result.events)
    assert not any(
        isinstance(event, NodeStarted) and event.node_id == START_ID
        for event in result.events
    )

    records = _durable_records(path)
    assert records[prefix_length] == RunResumedRecord(
        retry_node_ids=("a", "b"),
        num_workers=expected_workers,
    )


def test_nonterminal_resume_restores_recorded_llm_without_reading_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Apply the complete logged agent map and retain the canonical original input."""
    path = tmp_path / "agent-resume.jsonl"
    loaded = load_flow(_AGENT_FLOW)
    effective = resolve_llm_cascade(
        copy.deepcopy(loaded.compiled),
        {},
        defaults={"provider": "openai", "model": "recorded-model"},
    )
    _write_nonterminal_prefix(
        path,
        loaded,
        {"topic": "Ada"},
        started=("answer",),
        num_workers=0,
        effective=effective,
    )
    captured = []

    class _Model:
        """Return one deterministic reply from the no-network recovery probe."""

        def invoke(self, messages):
            """Return the scripted text after model construction was captured."""
            return SimpleNamespace(content="recorded reply")

    def model_from_config(config):
        captured.append(copy.deepcopy(config))
        return _Model()

    def forbidden_default() -> str:
        raise AssertionError("nonterminal recovery consulted an ambient LLM default")

    monkeypatch.setattr(llm_clients_module, "model_from_config", model_from_config)
    monkeypatch.setattr(run_module, "connection_snapshot", forbidden_default)
    monkeypatch.setattr(settings_module, "default_llm_provider", forbidden_default)
    monkeypatch.setattr(settings_module, "default_llm_model", forbidden_default)

    result = run_module.resume_flow(load_flow(_AGENT_FLOW), log=path)

    assert result.status == "succeeded"
    assert result.input == {"topic": "Ada"}
    assert result.output == "recorded reply"
    from yian.llm_clients.connection import legacy_connection_config
    assert captured == [legacy_connection_config(effective["answer"])]
    assert isinstance(result.events[0], RunResumed)
    assert not any(isinstance(event, RunStarted) for event in result.events)
    assert not any(
        isinstance(event, NodeStarted) and event.node_id == START_ID
        for event in result.events
    )


def test_run_started_append_failure_prevents_execution_observation_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Treat initial durability failure as an exact host error before reduction."""
    _ACTION_PROBE.clear()
    path = tmp_path / "start-append-failure.jsonl"
    expected = OSError("RunStarted fsync failed")
    fake = _TrackingRunLog(
        _RUN_ID,
        path,
        append_failure=expected,
        fail_when=lambda record: isinstance(record, RunStartedRecord),
    )
    _install_create_log(monkeypatch, fake)
    seen = []

    with pytest.raises(OSError) as caught:
        run_module.run_flow(
            load_flow(_ACTION_FLOW),
            {},
            log=path,
            run_id=_RUN_ID,
            on_event=seen.append,
        )

    assert caught.value is expected
    assert tuple(fake) == ()
    assert _ACTION_PROBE == []
    assert seen == []
    assert fake.closed
    assert fake.close_calls >= 1


def test_outcome_append_failure_leaves_started_work_recoverable_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not reduce or observe a successful node outcome whose append failed."""
    _ACTION_PROBE.clear()
    path = tmp_path / "outcome-append-failure.jsonl"
    expected = OSError("NodeSucceeded fsync failed")
    fake = _TrackingRunLog(
        _RUN_ID,
        path,
        append_failure=expected,
        fail_when=lambda record: (
            isinstance(record, NodeSucceededRecord) and record.node_id == "work"
        ),
    )
    _install_create_log(monkeypatch, fake)
    seen = []

    with pytest.raises(OSError) as caught:
        run_module.run_flow(
            load_flow(_ACTION_FLOW),
            {},
            log=path,
            run_id=_RUN_ID,
            on_event=seen.append,
        )

    assert caught.value is expected
    assert _ACTION_PROBE == ["ran"]
    records = tuple(envelope.record for envelope in fake)
    assert records[-1] == NodeStartedRecord(node_id="work")
    assert not any(
        isinstance(record, NodeSucceededRecord) and record.node_id == "work"
        for record in records
    )
    assert any(
        isinstance(event, NodeStarted) and event.node_id == "work" for event in seen
    )
    assert not any(
        isinstance(event, NodeSucceeded) and event.node_id == "work" for event in seen
    )
    assert fake.closed
    assert fake.close_calls >= 1


def test_resume_callback_exception_closes_and_unlocks_nonterminal_log(
    tmp_path: Path,
) -> None:
    """Release recovery's Jsonl owner when a current-segment callback raises."""
    _ACTION_PROBE.clear()
    path = tmp_path / "resume-callback-failure.jsonl"
    _write_nonterminal_prefix(
        path,
        load_flow(_ACTION_FLOW),
        {},
        started=("work",),
        num_workers=0,
    )
    expected = RuntimeError("resume observer failed")
    seen = []

    def reject(event) -> None:
        seen.append(event)
        if isinstance(event, NodeStarted) and event.node_id == "work":
            raise expected

    with pytest.raises(RuntimeError) as caught:
        run_module._resume_flow_controlled(
            load_flow(_ACTION_FLOW),
            log=path,
            on_event=reject,
            stop_event=threading.Event(),
        )

    assert caught.value is expected
    assert _ACTION_PROBE == []
    assert isinstance(seen[0], RunResumed)
    assert not any(isinstance(event, RunStarted) for event in seen)
    assert not any(
        isinstance(event, NodeStarted) and event.node_id == START_ID for event in seen
    )
    records = _durable_records(path)
    assert records[-1] == NodeStartedRecord(node_id="work")
    assert sum(
        isinstance(record, NodeStartedRecord) and record.node_id == "work"
        for record in records
    ) == 2
    assert any(isinstance(record, RunResumedRecord) for record in records)
    assert not any(type(record).__name__ == "RunAbortedRecord" for record in records)


def test_callback_exception_drains_waiter_reraises_exactly_and_unlocks(
    tmp_path: Path,
) -> None:
    """Wake and join a cooperative sibling before exposing the callback failure."""
    global _ACTIVE_STOP
    path = tmp_path / "callback-interruption.jsonl"
    stop = _TrackingStopEvent()
    expected = RuntimeError("observer rejected a durable outcome")
    _ACTIVE_STOP = stop

    def reject(event) -> None:
        if isinstance(event, NodeSucceeded) and event.node_id == "trigger":
            raise expected

    try:
        with pytest.raises(RuntimeError) as caught:
            run_module._run_flow_controlled(
                load_flow(_CALLBACK_DRAIN_FLOW),
                {},
                log=path,
                run_id=_RUN_ID,
                num_workers=2,
                on_event=reject,
                stop_event=stop,
            )
    finally:
        _ACTIVE_STOP = None

    assert caught.value is expected
    assert stop.is_set()
    assert stop.wait_released.is_set()
    records = _durable_records(path)
    assert not any(
        isinstance(record, NodeSucceededRecord) and record.node_id == "waiter"
        for record in records
    )
    assert not any(type(record).__name__ == "RunAbortedRecord" for record in records)


def test_callback_exception_drains_real_blocked_human_interaction_and_unlocks(
    tmp_path: Path,
) -> None:
    """Wait for the installed HUMAN_INPUT handler before closing its run log."""
    global _ACTIVE_STOP
    path = tmp_path / "human-callback-interruption.jsonl"
    stop = _TrackingStopEvent()
    interaction_released = threading.Event()
    expected = RuntimeError("observer rejected while ask_user was blocked")
    _ACTIVE_STOP = stop

    def answer(**request):
        assert request["node_id"] == "human"
        assert stop.wait(timeout=2), "run did not wake the blocked ask_user handler"
        interaction_released.set()
        return "late answer"

    def reject(event) -> None:
        if isinstance(event, NodeSucceeded) and event.node_id == "trigger":
            raise expected

    try:
        with pytest.raises(RuntimeError) as caught:
            run_module._run_flow_controlled(
                load_flow(_HUMAN_DRAIN_FLOW),
                {},
                log=path,
                run_id=_RUN_ID,
                num_workers=2,
                on_event=reject,
                stop_event=stop,
                tools={"ask_user": answer},
            )
    finally:
        _ACTIVE_STOP = None

    assert caught.value is expected
    assert stop.is_set()
    assert interaction_released.is_set()
    records = _durable_records(path)
    assert any(
        isinstance(record, NodeStartedRecord) and record.node_id == "human"
        for record in records
    )
    assert not any(
        isinstance(record, (NodeSucceededRecord, NodeFailedRecord))
        and record.node_id == "human"
        for record in records
    )
    assert not any(type(record).__name__ == "RunAbortedRecord" for record in records)


def test_host_interruption_drains_then_durable_resume_retries_to_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Leave unfinished starts resumable after exact host-control propagation."""
    global _ACTIVE_STOP, _HOST_INTERRUPTION_ARMED
    path = tmp_path / "host-interruption.jsonl"
    stop = _TrackingStopEvent()
    _ACTIVE_STOP = stop
    _HOST_INTERRUPTION_ARMED = True

    try:
        with pytest.raises(_HostStop) as caught:
            run_module._run_flow_controlled(
                load_flow(_HOST_DRAIN_FLOW),
                {},
                log=path,
                run_id=_RUN_ID,
                num_workers=2,
                stop_event=stop,
            )

        assert caught.value is _HOST_INTERRUPTION
        assert stop.is_set()
        assert stop.wait_released.is_set()
        interrupted = _durable_records(path)
        assert not any(
            isinstance(record, (NodeSucceededRecord, NodeFailedRecord))
            and record.node_id in {"interrupter", "waiter"}
            for record in interrupted
        )
        assert not any(
            type(record).__name__ == "RunAbortedRecord" for record in interrupted
        )

        monkeypatch.setattr(
            wait_module,
            "_utc_now",
            lambda: datetime(2100, 1, 1, tzinfo=timezone.utc),
        )
        resumed = run_module.resume_flow(load_flow(_HOST_DRAIN_FLOW), log=path)

        assert resumed.status == "succeeded"
        assert resumed.output == {"interrupter": "recovered", "waiter": None}
        assert isinstance(resumed.events[0], RunResumed)
        assert not any(isinstance(event, RunStarted) for event in resumed.events)
        assert not any(
            isinstance(event, NodeStarted) and event.node_id == START_ID
            for event in resumed.events
        )
        completed = _durable_records(path)
        assert any(
            record
            == RunResumedRecord(
                retry_node_ids=("interrupter", "waiter"),
                num_workers=2,
            )
            for record in completed
        )
        assert not any(
            type(record).__name__ == "RunAbortedRecord" for record in completed
        )
    finally:
        _ACTIVE_STOP = None
        _HOST_INTERRUPTION_ARMED = True
