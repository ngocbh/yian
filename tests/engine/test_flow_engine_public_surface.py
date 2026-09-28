"""Public direct-engine contract for the authoritative run-log runtime.

These tests start below the package-root run boundary on purpose.  A direct caller
must prepare and durably apply ``RunStarted`` first, then lend that exact state/log
pair to ``agent_composer.runtime.FlowEngine``.  The engine drives only new work,
borrows rather than owns the log, and exposes no checkpoint-era lifecycle methods.
"""

from __future__ import annotations

import inspect
import threading
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, get_args, get_origin, get_type_hints

import pytest

import agent_composer
import agent_composer.llm_clients as llm_clients_module
from agent_composer.compile.expand import build_grow
from agent_composer.compile.llm_cascade import resolve_llm_cascade
from agent_composer.compile.manifest import flow_fingerprint
from agent_composer.compile.model import CompiledFlow, Edge, END_ID, START_ID
from agent_composer.compose import LoadError, load_flow
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.events import (
    NodeExpanded,
    NodeStarted,
    NodeSucceeded,
    RunStarted,
    RunSucceeded,
    StreamChunk,
)
from agent_composer.nodes.base import Node, NodeKind, Output
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime import FlowEngine
from agent_composer.runtime.eval_node import MemoryContextOpener
from agent_composer.runtime.log import MemoryRunLog, NodeStartedRecord, RunLog
from agent_composer.runtime.state import (
    GrowBuilder,
    RunStartData,
    RunState,
    commit_prepared,
    prepare_run_started,
)
from agent_composer.typesys import VariablePool


_RUN_ID = "direct-engine-run"
_EXECUTION_PROBE: list[str] = []


def _record_execution(inputs: dict[str, Any]) -> str:
    """Record one real CODE-node invocation for authority-rejection assertions."""
    _EXECUTION_PROBE.append("ran")
    return "done"


class _StreamingNode(Node):
    """Yield two transient chunks before returning one durable output value."""

    kind = NodeKind.AGENT

    def forward(self, inputs: dict[str, Any]):
        """Emit the scripted token fragments and return their combined value."""
        yield StreamChunk(self.id, "output", "hel")
        yield StreamChunk(self.id, "output", "lo", final=True)
        return Output("hello")


class _TrackingMemoryRunLog(MemoryRunLog):
    """Count close requests while preserving the real in-memory log behavior."""

    def __init__(self, run_id: str) -> None:
        """Create an open log with no observed close requests."""
        super().__init__(run_id)
        self.close_calls = 0

    def close(self) -> None:
        """Record and delegate each owner-issued close request."""
        self.close_calls += 1
        super().close()


def _prepared(
    flow: CompiledFlow,
    inputs: dict[str, Any],
    *,
    num_workers: int = 0,
    effective_llm_by_static_path: dict[str, dict[str, Any]] | None = None,
    log: MemoryRunLog | None = None,
    fingerprint: str | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Durably initialize one private graph and return its ready START frontier."""
    state = RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, dict(inputs))
    pool.add_system("run_id", _RUN_ID)
    log = MemoryRunLog(_RUN_ID) if log is None else log
    prepared = prepare_run_started(
        state,
        RunStartData(
            run_id=_RUN_ID,
            flow_fingerprint=(
                flow_fingerprint(flow) if fingerprint is None else fingerprint
            ),
            initial_pool=pool,
            start_wiring_override={
                name: f"${{input.{name}}}" for name in inputs
            },
            effective_llm_by_static_path=(
                effective_llm_by_static_path
                if effective_llm_by_static_path is not None
                else {}
            ),
            num_workers=num_workers,
        ),
    )
    commit_prepared(log, state, prepared)
    return state, log


def _streaming_flow() -> CompiledFlow:
    """Return a small synthetic graph whose middle node exercises streaming."""
    stream = _StreamingNode("stream")
    end = EndNode.record(END_ID, output_names=["output"])
    return CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            stream.id: stream,
            END_ID: end,
        },
        edges=[
            Edge("start-stream", START_ID, stream.id, optional=True, ordering=True),
            Edge("stream-end", stream.id, END_ID, input_group="output"),
        ],
        wiring={START_ID: {}, stream.id: {}, END_ID: {"output": "${stream.output}"}},
    )


def _optional_inner(annotation: Any) -> Any:
    """Return the non-None member of one required optional annotation."""
    members = get_args(annotation)
    assert types.NoneType in members
    concrete = tuple(member for member in members if member is not types.NoneType)
    assert len(concrete) == 1
    return concrete[0]


def _assert_callable_annotation(
    annotation: Any,
    parameters: tuple[Any, ...],
    result: Any,
) -> None:
    """Compare a callable semantically across typing and collections aliases."""
    assert get_origin(annotation) is Callable
    callable_parts = get_args(annotation)
    assert len(callable_parts) == 2
    annotated_parameters, annotated_result = callable_parts
    assert tuple(annotated_parameters) == parameters
    if result is types.NoneType:
        assert annotated_result in (None, types.NoneType)
    else:
        assert annotated_result is result


_STATIC_FLOW = """
id: direct-static
name: direct_static
input:
  x: int
nodes:
  twice:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.x}
    output: int
output: ${twice.output}
"""

_ACTION_FLOW = """
id: direct-action
name: direct_action
nodes:
  work:
    kind: code
    code: tests.engine.test_flow_engine_public_surface:_record_execution
    output: str
output: ${work.output}
"""

_CHILD_FLOW = """
id: direct-child
name: direct_child
input:
  topic: str
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${input.topic}
    output: str
output: ${echo.output}
"""

_CALL_FLOW = """
id: direct-call
name: direct_call
input:
  topic: str
uses:
  direct-child: direct-child
nodes:
  invoke:
    kind: call
    call: direct-child
    input:
      topic: ${input.topic}
output: ${invoke.output}
"""

_MAP_FLOW = """
id: direct-map
name: direct_map
input:
  topics: list[str]
uses:
  direct-child: direct-child
nodes:
  invoke:
    kind: map
    call: direct-child
    over: ${input.topics}
    input:
      topic: ${item}
output: ${invoke.output}
"""

_LOOP_FLOW = """
id: direct-loop
name: direct_loop
defs:
  bump:
    input:
      n: int
      exited: bool
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
    output: ${step.output}
nodes:
  repeat:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not ${exited}
    max: 5
output: ${repeat.output}
"""

_AGENT_FLOW = """
id: direct-agent
name: direct_agent
nodes:
  answer:
    kind: agent
    mode: plain
    prompt: answer directly
output: ${answer.output}
"""


def _child_resolver(flow_id: str, version: str | None = None):
    """Resolve the one test child used by direct CALL and MAP runs."""
    if flow_id != "direct-child":
        raise LoadDiagnostics().error(f"unknown child {flow_id!r}")
    return load_flow(_CHILD_FLOW)


def _load_dynamic(kind: str):
    """Load one real authorable spawner graph for the named combinator."""
    if kind == "call":
        return load_flow(_CALL_FLOW, child_resolver=_child_resolver), {"topic": "Ada"}
    if kind == "map":
        return load_flow(_MAP_FLOW, child_resolver=_child_resolver), {
            "topics": ["Ada", "Grace"]
        }
    if kind == "loop":
        return load_flow(_LOOP_FLOW), {}
    raise AssertionError(f"unknown dynamic test kind {kind!r}")


def test_constructor_is_exactly_the_recorded_state_log_surface():
    """Require the public runtime class to expose only the new constructor inputs."""
    signature = inspect.signature(FlowEngine.__init__)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "self",
        "state",
        "log",
        "grow_builder",
        "num_workers",
        "on_event",
        "on_stream",
        "stop_event",
        "memory_context_opener",
    ]
    assert [parameter.kind for parameter in parameters] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
    ]
    assert [parameter.default for parameter in parameters] == [
        inspect.Parameter.empty,
        inspect.Parameter.empty,
        inspect.Parameter.empty,
        inspect.Parameter.empty,
        inspect.Parameter.empty,
        None,
        None,
        None,
        None,
    ]
    hints = get_type_hints(FlowEngine.__init__)
    assert set(hints) == {
        "state",
        "log",
        "grow_builder",
        "num_workers",
        "on_event",
        "on_stream",
        "stop_event",
        "memory_context_opener",
        "return",
    }
    assert hints["state"] is RunState
    assert hints["log"] is RunLog
    grow_parameters, grow_result = get_args(GrowBuilder)
    _assert_callable_annotation(
        hints["grow_builder"],
        tuple(grow_parameters),
        grow_result,
    )
    assert hints["num_workers"] is int
    _assert_callable_annotation(
        _optional_inner(hints["on_event"]),
        (Any,),
        types.NoneType,
    )
    _assert_callable_annotation(
        _optional_inner(hints["on_stream"]),
        (StreamChunk,),
        types.NoneType,
    )
    assert _optional_inner(hints["stop_event"]) is threading.Event
    assert _optional_inner(hints["memory_context_opener"]) is MemoryContextOpener
    assert hints["return"] is types.NoneType


def test_static_run_yields_only_new_post_durable_observers():
    """Exclude the already-applied RunStarted observer and mirror callback order."""
    state, log = _prepared(load_flow(_STATIC_FLOW).compiled, {"x": 4})
    callbacks = []

    iterator = FlowEngine(
        state,
        log,
        grow_builder=build_grow,
        num_workers=0,
        on_event=callbacks.append,
    ).run()
    first = next(iterator)

    # The callback is invoked synchronously before the same observer is yielded.
    assert callbacks == [first]
    yielded = [first, *iterator]

    assert callbacks == yielded
    assert not any(isinstance(event, RunStarted) for event in yielded)
    assert [type(event) for event in yielded] == [
        NodeStarted,
        NodeSucceeded,
        NodeStarted,
        NodeSucceeded,
        NodeStarted,
        NodeSucceeded,
        RunSucceeded,
    ]
    assert yielded[-1].output == 8
    assert state.result == yielded[-1]


def test_stream_chunks_are_callback_only_and_never_durable():
    """Send chunks only to on_stream while records and yielded observers stay closed."""
    state, log = _prepared(
        _streaming_flow(),
        {},
        fingerprint="f" * 64,
    )
    chunks = []

    yielded = list(
        FlowEngine(
            state,
            log,
            grow_builder=build_grow,
            num_workers=0,
            on_stream=chunks.append,
        ).run()
    )

    assert chunks == [
        StreamChunk("stream", "output", "hel"),
        StreamChunk("stream", "output", "lo", final=True),
    ]
    assert not any(isinstance(event, StreamChunk) for event in yielded)
    assert not any(isinstance(envelope.record, StreamChunk) for envelope in log)
    assert log.next_sequence == 7
    assert state.result == yielded[-1]
    assert yielded[-1] == RunSucceeded(output="hello")


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("call", "Ada"),
        ("map", ["Ada", "Grace"]),
        ("loop", {"n": 3, "exited": True}),
    ],
)
def test_direct_engine_executes_real_dynamic_combinators(kind: str, expected: Any):
    """Drive real CALL, MAP, and LOOP nodes through the injected growth builder."""
    loaded, inputs = _load_dynamic(kind)
    state, log = _prepared(loaded.compiled, inputs)

    events = list(
        FlowEngine(
            state,
            log,
            grow_builder=build_grow,
            num_workers=0,
        ).run()
    )

    assert any(isinstance(event, NodeExpanded) for event in events)
    assert isinstance(events[-1], RunSucceeded)
    assert events[-1].output == expected


def test_direct_agent_uses_its_node_owned_model_factory(monkeypatch):
    """Keep direct AGENT execution on ``model_from_config`` with no engine LLM arg."""
    captured = []

    class _Model:
        """Return one text-like reply from the no-network fake model."""

        def invoke(self, messages):
            """Capture that invocation occurred and return the scripted answer."""
            captured.append(tuple(messages))
            return SimpleNamespace(content="direct answer")

    monkeypatch.setattr(
        llm_clients_module,
        "model_from_config",
        lambda config: captured.append(dict(config)) or _Model(),
    )
    flow = load_flow(_AGENT_FLOW).compiled
    effective = resolve_llm_cascade(
        flow,
        {},
        defaults={"provider": "openai", "model": "fake-direct"},
    )
    state, log = _prepared(
        flow,
        {},
        effective_llm_by_static_path=effective,
    )

    events = list(
        FlowEngine(
            state,
            log,
            grow_builder=build_grow,
            num_workers=0,
        ).run()
    )

    assert captured[0] == {"provider": "openai", "model": "fake-direct"}
    assert len(captured[1]) == 2
    assert isinstance(events[-1], RunSucceeded)
    assert events[-1].output == "direct answer"


def test_growth_builder_is_a_required_constructor_dependency():
    """Reject omission of the compile-owned growth seam even for a static graph."""
    state, log = _prepared(load_flow(_STATIC_FLOW).compiled, {"x": 2})

    with pytest.raises(TypeError, match="grow_builder"):
        FlowEngine(state, log, num_workers=0)


def test_engine_borrows_and_never_closes_the_run_log():
    """Leave log lifetime under the direct caller that supplied the authority."""
    log = _TrackingMemoryRunLog(_RUN_ID)
    state, log = _prepared(
        load_flow(_STATIC_FLOW).compiled,
        {"x": 3},
        log=log,
    )

    list(
        FlowEngine(
            state,
            log,
            grow_builder=build_grow,
            num_workers=0,
        ).run()
    )

    assert log.close_calls == 0
    log.close()
    assert log.close_calls == 1


def test_unprepared_state_is_rejected_before_any_node_executes():
    """Require an applied RunStarted transition before the engine can run code."""
    _EXECUTION_PROBE.clear()
    state = RunState(load_flow(_ACTION_FLOW).compiled)
    log = MemoryRunLog(_RUN_ID)

    with pytest.raises(ValueError, match="initialized"):
        list(
            FlowEngine(
                state,
                log,
                grow_builder=build_grow,
                num_workers=0,
            ).run()
        )

    assert _EXECUTION_PROBE == []
    assert log.next_sequence == 0


def test_mismatched_state_and_log_are_rejected_before_execution():
    """Reject a log from another run before consuming the prepared ready frontier."""
    _EXECUTION_PROBE.clear()
    state, source_log = _prepared(
        load_flow(_ACTION_FLOW).compiled,
        {},
    )
    other_log = MemoryRunLog("another-run")

    with pytest.raises(ValueError, match="run_id mismatch"):
        list(
            FlowEngine(
                state,
                other_log,
                grow_builder=build_grow,
                num_workers=0,
            ).run()
        )

    assert _EXECUTION_PROBE == []
    assert source_log.next_sequence == 1
    assert other_log.next_sequence == 0


def test_mismatched_durable_sequence_is_rejected_before_execution():
    """Reject a log whose durable tail is ahead of the state's applied sequence."""
    _EXECUTION_PROBE.clear()
    state, log = _prepared(load_flow(_ACTION_FLOW).compiled, {})
    log.append(NodeStartedRecord(node_id="unapplied"))

    with pytest.raises(ValueError, match="sequence mismatch"):
        list(
            FlowEngine(
                state,
                log,
                grow_builder=build_grow,
                num_workers=0,
            ).run()
        )

    assert _EXECUTION_PROBE == []
    assert state.next_sequence == 1
    assert log.next_sequence == 2


def test_mismatched_durable_worker_count_is_rejected_before_execution():
    """Honor the RunStarted worker count instead of silently changing drive mode."""
    _EXECUTION_PROBE.clear()
    state, log = _prepared(
        load_flow(_ACTION_FLOW).compiled,
        {},
        num_workers=1,
    )

    with pytest.raises(ValueError, match="worker mismatch"):
        list(
            FlowEngine(
                state,
                log,
                grow_builder=build_grow,
                num_workers=0,
            ).run()
        )

    assert _EXECUTION_PROBE == []
    assert state.num_workers == 1
    assert state.next_sequence == log.next_sequence == 1


def test_checkpoint_era_engine_methods_are_absent():
    """Delete the legacy live-resume and artificial-abort lifecycle from the class."""
    for method in ("snapshot", "restore", "resume", "request_abort"):
        assert not hasattr(FlowEngine, method), method


def test_flow_engine_is_not_exported_from_the_package_root():
    """Keep the advanced prepared-state engine scoped to agent_composer.runtime."""
    assert not hasattr(agent_composer, "FlowEngine")
    assert "FlowEngine" not in getattr(agent_composer, "__all__", ())
