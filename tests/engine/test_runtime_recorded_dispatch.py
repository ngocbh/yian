"""Bounded dispatch over the append-only run-state reducer.

These tests exercise the private recorded dispatcher before the atomic public
``FlowEngine`` cutover.  They deliberately use real reducer/log transitions so
durability order, unfinished starts, observer order, and drain behavior cannot be
faked by an event-only adapter.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

import agent_composer.runtime.engine as engine_module
import agent_composer.runtime.state as state_module
from agent_composer.compile.model import CompiledFlow, Edge, END_ID, Flow, START_ID
from agent_composer.diagnostics import Failure, OriginRef
from agent_composer.events import (
    NodeExpanded,
    NodeFailed,
    NodeRouted,
    NodeStarted,
    NodeSucceeded,
    RunFailed,
    RunSucceeded,
    StreamChunk,
)
from agent_composer.nodes.base import Grow, Node, NodeKind, Output, Route, _HostStop
from agent_composer.nodes.contracts import ContractOrigins, ValueContract
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.log import (
    MemoryRunLog,
    NodeExpandedRecord,
    NodeFailedRecord,
    NodeStartedRecord,
    NodeSucceededRecord,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_run_started,
)
from agent_composer.typesys import Type, ValueKind, VariablePool


_RUN_ID = "recorded-dispatch"
_FINGERPRINT = "d" * 64


class _ActionNode(Node):
    """Run one injected side effect and return its value."""

    kind = NodeKind.CODE

    def __init__(
        self,
        node_id: str,
        action: Callable[[], Any],
        *,
        output_contract: ValueContract | None = None,
        origin: OriginRef | None = None,
    ) -> None:
        super().__init__(node_id, output_contract=output_contract, origin=origin)
        self._action = action

    def _run_impl(self, inputs: dict[str, Any]) -> Output:
        return Output(self._action())


class _RouteNode(Node):
    """Return one fixed route handle through the raw-outcome arm."""

    kind = NodeKind.CASE

    def _run_impl(self, inputs: dict[str, Any]) -> Route:
        return Route("chosen")


class _InvalidRouteNode(Node):
    """Return a handle absent from this node's outgoing topology."""

    kind = NodeKind.CASE

    def _run_impl(self, inputs: dict[str, Any]) -> Route:
        return Route("absent")


class _InvalidCommitTargetNode(Node):
    """Return malformed node-produced success metadata."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs: dict[str, Any]) -> Output:
        return Output("done", commit_as=42)  # type: ignore[arg-type]


class _ExplicitCommitNode(_ActionNode):
    """Return one value with a result-owned success target."""

    def __init__(self, node_id: str, action: Callable[[], Any], target: str) -> None:
        super().__init__(node_id, action)
        self._target = target

    def _run_impl(self, inputs: dict[str, Any]) -> Output:
        return Output(self._action(), commit_as=self._target)


class _GrowNode(Node):
    """Return one seed-first expansion through the raw-outcome arm."""

    kind = NodeKind.CALL
    is_spawner = True
    grow_depth_delta = 1

    def _run_impl(self, inputs: dict[str, Any]) -> Grow:
        return Grow(seed={"canonical": (1, 2)})


class _MalformedSeedGrowNode(_GrowNode):
    """Return a malformed seed while retaining one legal spawner kind."""

    def _run_impl(self, inputs: dict[str, Any]) -> Grow:
        return Grow(seed=[])


class _StopAwareNode(Node):
    """Block until cooperative drain wakes this worker."""

    kind = NodeKind.CODE
    needs_stop_event = True

    def __init__(
        self,
        node_id: str,
        entered: threading.Event,
        released: threading.Event,
    ) -> None:
        super().__init__(node_id)
        self._entered = entered
        self._released = released

    def _run_impl(self, inputs: dict[str, Any], *, stop_event: threading.Event) -> Output:
        self._entered.set()
        stop_event.wait(2)
        self._released.set()
        raise _HostStop(primary=False)


class _PrimaryHostStopNode(Node):
    """Raise one caller-owned host exception object unchanged."""

    kind = NodeKind.CODE

    def __init__(
        self,
        node_id: str,
        exception: _HostStop,
        *,
        wait_for: threading.Event | None = None,
    ) -> None:
        super().__init__(node_id)
        self._exception = exception
        self._wait_for = wait_for

    def _run_impl(self, inputs: dict[str, Any]) -> Output:
        if self._wait_for is not None:
            assert self._wait_for.wait(2)
        raise self._exception


class _LateStreamNode(Node):
    """Emit one live chunk and one chunk after failure-driven stop begins."""

    kind = NodeKind.CODE
    needs_stop_event = True

    def __init__(self, node_id: str, failure_may_run: threading.Event) -> None:
        super().__init__(node_id)
        self._failure_may_run = failure_may_run

    def _run_impl(self, inputs: dict[str, Any], *, stop_event: threading.Event):
        yield StreamChunk(self.id, "output", "early")
        self._failure_may_run.set()
        stop_event.wait(2)
        yield StreamChunk(self.id, "output", "late")
        return Output("discarded")


def _flow(nodes: list[Node]) -> CompiledFlow:
    """Return START fan-out into ``nodes`` followed by one ordering-joined END."""
    all_nodes = {
        START_ID: StartNode(START_ID),
        **{node.id: node for node in nodes},
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    edges = [
        Edge(
            id=f"start-{node.id}",
            from_=START_ID,
            to=node.id,
            optional=True,
            ordering=True,
        )
        for node in nodes
    ]
    edges.extend(
        Edge(
            id=f"{node.id}-end",
            from_=node.id,
            to=END_ID,
            optional=True,
            ordering=True,
        )
        for node in nodes
    )
    return CompiledFlow.from_parts(nodes=all_nodes, edges=edges)


def _growth_fragment(spawner_id: str) -> Flow:
    """Return one valid dynamic boundary pair committing under ``spawner_id``."""
    start_id = f"{spawner_id}/child/start"
    end_id = f"{spawner_id}/child/end"
    start = StartNode(start_id, params=[])
    end = EndNode.record(end_id, output_names=[])
    end.commit_as = spawner_id
    return Flow(
        nodes={start_id: start, end_id: end},
        edges=[
            Edge(
                id=f"{start_id}-end",
                from_=start_id,
                to=end_id,
                ordering=True,
            )
        ],
        wiring={start_id: {}, end_id: {}},
        start_id=start_id,
        end_id=end_id,
    )


def _initialized(
    flow: CompiledFlow,
    *,
    num_workers: int,
    log: MemoryRunLog | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Apply one authoritative RunStarted and return its ready START frontier."""
    state = RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {})
    pool.add_system("run_id", _RUN_ID)
    log = MemoryRunLog(_RUN_ID) if log is None else log
    prepared = prepare_run_started(
        state,
        RunStartData(
            run_id=_RUN_ID,
            flow_fingerprint=_FINGERPRINT,
            initial_pool=pool,
            start_wiring_override={},
            effective_llm_by_static_path={},
            num_workers=num_workers,
        ),
    )
    commit_prepared(log, state, prepared)
    return state, log


def _dispatcher(
    state: RunState,
    log: MemoryRunLog,
    *,
    num_workers: int,
    **kwargs: Any,
) -> FlowEngine:
    """Construct the private runtime with a growth seam unused by static tests."""
    return FlowEngine(
        state,
        log,
        grow_builder=lambda _spawner, _seed: None,
        num_workers=num_workers,
        **kwargs,
    )


def _capture_semantic_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> list[Failure]:
    """Spy on the canonical failure-record preparation without replacing it."""
    captured: list[Failure] = []
    project = FlowEngine._prepare_failure

    def capture(self, node_id, failure, *, locator=None, range=None):
        captured.append(failure)
        return project(
            self,
            node_id,
            failure,
            locator=locator,
            range=range,
        )

    monkeypatch.setattr(FlowEngine, "_prepare_failure", capture)
    return captured


class _GuardedLog(MemoryRunLog):
    """Inject append failures or inspect execution at exact durable records."""

    def __init__(self, run_id: str) -> None:
        super().__init__(run_id)
        self.guard: Callable[[Any], None] | None = None
        self.close_calls = 0

    def append(self, record):
        if self.guard is not None:
            self.guard(record)
        return super().append(record)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def test_started_record_is_durable_before_node_code_runs():
    executed = []
    log = _GuardedLog(_RUN_ID)
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work") or "done")]),
        num_workers=0,
        log=log,
    )

    def guard(record):
        if isinstance(record, NodeStartedRecord) and record.node_id == "work":
            assert executed == []

    log.guard = guard
    list(_dispatcher(state, log, num_workers=0).run())

    records = [envelope.record for envelope in log]
    start_index = records.index(NodeStartedRecord(node_id="work"))
    success_index = next(
        index
        for index, record in enumerate(records)
        if isinstance(record, NodeSucceededRecord) and record.node_id == "work"
    )
    assert executed == ["work"]
    assert start_index < success_index


def test_start_append_failure_prevents_node_execution_and_propagates_exactly():
    executed = []
    expected = OSError("start fsync failed")
    log = _GuardedLog(_RUN_ID)
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work") or "done")]),
        num_workers=0,
        log=log,
    )

    def reject(record):
        if isinstance(record, NodeStartedRecord) and record.node_id == "work":
            raise expected

    log.guard = reject
    with pytest.raises(OSError) as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert caught.value is expected
    assert executed == []
    assert "work" not in state.in_flight


def test_outcome_append_failure_leaves_one_unfinished_durable_start():
    executed = []
    expected = OSError("outcome fsync failed")
    log = _GuardedLog(_RUN_ID)
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work") or "done")]),
        num_workers=0,
        log=log,
    )

    def reject(record):
        if isinstance(record, NodeSucceededRecord) and record.node_id == "work":
            raise expected

    log.guard = reject
    with pytest.raises(OSError) as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert caught.value is expected
    assert executed == ["work"]
    assert tuple(state.in_flight) == ("work",)
    assert any(
        isinstance(envelope.record, NodeStartedRecord)
        and envelope.record.node_id == "work"
        for envelope in log
    )
    assert not any(
        isinstance(envelope.record, NodeSucceededRecord)
        and envelope.record.node_id == "work"
        for envelope in log
    )


def test_reducer_programming_failure_is_not_rewritten_as_node_failure(monkeypatch):
    expected = RuntimeError("reducer invariant broke")
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
    )
    real_commit = engine_module.commit_prepared

    def broken_commit(target_log, target_state, prepared):
        if (
            isinstance(prepared.record, NodeSucceededRecord)
            and prepared.record.node_id == "work"
        ):
            raise expected
        return real_commit(target_log, target_state, prepared)

    monkeypatch.setattr(engine_module, "commit_prepared", broken_commit)

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == ("work",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


def test_success_preparation_runtime_error_remains_exact_host_failure(monkeypatch):
    expected = RuntimeError("success reducer programming fault")
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
    )
    real_prepare = engine_module.prepare_node_succeeded

    def broken_prepare(target_state, node_id, output, *, commit_as=None):
        if node_id == "work":
            raise expected
        return real_prepare(
            target_state,
            node_id,
            output,
            commit_as=commit_as,
        )

    monkeypatch.setattr(engine_module, "prepare_node_succeeded", broken_prepare)

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == ("work",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


@pytest.mark.parametrize(
    ("prepare_name", "node_factory"),
    [
        (
            "prepare_node_succeeded",
            lambda: _ActionNode("work", lambda: "done"),
        ),
        ("prepare_node_routed", lambda: _RouteNode("work")),
        ("prepare_expanded", lambda: _GrowNode("work")),
    ],
    ids=("success", "route", "growth"),
)
@pytest.mark.parametrize("error_type", [TypeError, ValueError])
@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
def test_transition_preparation_data_shaped_programming_error_remains_exact_host_failure(
    monkeypatch,
    prepare_name,
    node_factory,
    error_type,
    num_workers,
):
    expected = error_type("transition reducer programming fault")
    state, log = _initialized(
        _flow([node_factory()]),
        num_workers=num_workers,
    )
    real_prepare = getattr(engine_module, prepare_name)

    def broken_prepare(*args, **kwargs):
        if args[1] == "work":
            raise expected
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(engine_module, prepare_name, broken_prepare)

    with pytest.raises(error_type) as caught:
        list(_dispatcher(state, log, num_workers=num_workers).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == ("work",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


def test_serial_yields_and_callbacks_follow_post_durable_observer_order():
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
    )
    callbacks = []

    yielded = list(
        _dispatcher(state, log, num_workers=0, on_event=callbacks.append).run()
    )

    assert callbacks == yielded
    assert [type(event) for event in yielded] == [
        NodeStarted,
        NodeSucceeded,
        NodeStarted,
        NodeSucceeded,
        NodeStarted,
        NodeSucceeded,
        RunSucceeded,
    ]
    assert [event.node_id for event in yielded if hasattr(event, "node_id")] == [
        START_ID,
        START_ID,
        "work",
        "work",
        END_ID,
        END_ID,
    ]


def test_invalid_raw_output_constructs_contract_failure_with_operation_origin(
    monkeypatch: pytest.MonkeyPatch,
):
    outer_origin = OriginRef("nodes/outer")
    node_origin = OriginRef("nodes/work")
    contract_origin = OriginRef("nodes/work/output")
    node = _ActionNode(
        "work",
        lambda: "not-an-int",
        output_contract=ValueContract(
            Type.scalar(ValueKind.INTEGER),
            ContractOrigins(default=contract_origin),
        ),
        origin=node_origin,
    )
    node.origin_context = (outer_origin,)
    state, log = _initialized(_flow([node]), num_workers=0)
    failures = _capture_semantic_failures(monkeypatch)

    events = list(_dispatcher(state, log, num_workers=0).run())

    assert len(failures) == 1
    assert failures[0].code == "contract.output"
    assert failures[0].node_id == "work"
    assert failures[0].trace == (outer_origin, contract_origin)
    assert failures[0].cause_type == "TypeCheckError"
    assert failures[0].traceback is not None
    assert isinstance(events[-2], NodeFailed)
    assert isinstance(events[-1], RunFailed)
    assert events[-1].failure.cause_type == "TypeCheckError"
    assert isinstance(state.result, RunFailed)
    assert isinstance(tuple(log)[-1].record, NodeFailedRecord)


def test_nested_output_failure_uses_the_exact_contract_origin(
    monkeypatch: pytest.MonkeyPatch,
):
    root_origin = OriginRef("nodes/work/output")
    score_origin = OriginRef("nodes/work/output/items/0/score")
    item_type = Type(
        kind=ValueKind.OBJECT,
        fields={"score": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"score"}),
    )
    output_type = Type(kind=ValueKind.LIST_OBJECT, element=item_type)
    node = _ActionNode(
        "work",
        lambda: [{"score": "bad"}],
        output_contract=ValueContract(
            output_type,
            ContractOrigins(
                default=root_origin,
                entries=(((0, "score"), score_origin),),
            ),
        ),
    )
    state, log = _initialized(_flow([node]), num_workers=0)
    failures = _capture_semantic_failures(monkeypatch)

    list(_dispatcher(state, log, num_workers=0).run())

    assert len(failures) == 1
    assert failures[0].code == "contract.output"
    assert failures[0].trace == (score_origin,)


def test_non_type_output_failure_uses_the_contract_default_origin(
    monkeypatch: pytest.MonkeyPatch,
):
    """Canonical JSON rejection has no value path and retains the root fallback."""
    root_origin = OriginRef("nodes/work/output")
    unrelated_origin = OriginRef("nodes/work/output/unrelated")
    node = _ActionNode(
        "work",
        lambda: float("nan"),
        output_contract=ValueContract(
            Type.scalar(ValueKind.NUMBER),
            ContractOrigins(
                default=root_origin,
                entries=((("unrelated",), unrelated_origin),),
            ),
        ),
    )
    state, log = _initialized(_flow([node]), num_workers=0)
    failures = _capture_semantic_failures(monkeypatch)

    list(_dispatcher(state, log, num_workers=0).run())

    assert len(failures) == 1
    assert failures[0].code == "contract.output"
    assert failures[0].cause_type == "ValueError"
    assert failures[0].trace == (root_origin,)


@pytest.mark.parametrize("explicit", [False, True], ids=("baked", "result"))
def test_redirected_output_uses_effective_target_contract_origin(
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
):
    producer_origin = OriginRef("nodes/producer")
    contract_origin = OriginRef("nodes/target/output")
    items_origin = OriginRef("nodes/target/output/items")

    def action() -> dict[str, list[dict[str, int | str]]]:
        """Return one invalid value for the selected target contract."""
        return {"items": [{"score": 1}, {"score": "bad"}]}

    producer = (
        _ExplicitCommitNode("producer", action, "target")
        if explicit
        else _ActionNode("producer", action, origin=producer_origin)
    )
    producer.origin = producer_origin
    producer.commit_as = "absent" if explicit else "target"
    flow = _flow([producer])
    item_type = Type(
        kind=ValueKind.OBJECT,
        fields={"score": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"score"}),
    )
    output_type = Type(
        kind=ValueKind.OBJECT,
        fields={"items": Type(kind=ValueKind.LIST_OBJECT, element=item_type)},
        required=frozenset({"items"}),
    )
    flow.nodes["target"] = _ActionNode(
        "target",
        lambda: {"items": [{"score": 1}]},
        output_contract=ValueContract(
            output_type,
            ContractOrigins(
                default=contract_origin,
                entries=((("items",), items_origin),),
            ),
        ),
    )
    state, log = _initialized(flow, num_workers=0)
    failures = _capture_semantic_failures(monkeypatch)

    list(_dispatcher(state, log, num_workers=0).run())

    assert len(failures) == 1
    assert failures[0].code == "contract.output"
    assert failures[0].node_id == "producer"
    assert failures[0].trace == (items_origin,)


def test_malformed_compiled_output_contract_remains_a_host_failure():
    """The dispatcher must not blame a node for corrupt compiled IR."""
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
    )
    state.flow.nodes["work"].output_contract = Type.scalar(  # type: ignore[assignment]
        ValueKind.STRING
    )

    with pytest.raises(TypeError, match="exact ValueContract") as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert not isinstance(caught.value, state_module._NodeDataError)
    assert tuple(state.in_flight) == ("work",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
def test_malformed_nested_output_type_remains_a_host_failure(num_workers: int):
    """Corrupt contract metadata must not masquerade as rejected node output."""
    nested_type = Type(
        kind=ValueKind.OBJECT,
        fields={"value": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"value"}),
    )
    output_type = Type(
        kind=ValueKind.OBJECT,
        fields={"nested": nested_type},
        required=frozenset({"nested"}),
    )
    object.__setattr__(nested_type, "required", [1, "value"])
    state, log = _initialized(
        _flow(
            [
                _ActionNode(
                    "work",
                    lambda: {"nested": {}},
                    output_contract=ValueContract(output_type),
                )
            ]
        ),
        num_workers=num_workers,
    )

    with pytest.raises(TypeError, match="required fields") as caught:
        list(_dispatcher(state, log, num_workers=num_workers).run())

    assert not isinstance(caught.value, state_module._NodeDataError)
    assert tuple(state.in_flight) == ("work",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
def test_noncanonical_output_type_metadata_remains_a_host_failure(num_workers: int):
    """A contract-owned surrogate must fail before output data is classified."""
    field = "bad\ud800"
    output_type = Type(
        kind=ValueKind.OBJECT,
        fields={field: Type.scalar(ValueKind.STRING)},
        required=frozenset({field}),
    )
    state, log = _initialized(
        _flow(
            [
                _ActionNode(
                    "work",
                    lambda: {field: "value"},
                    output_contract=ValueContract(output_type),
                )
            ]
        ),
        num_workers=num_workers,
    )

    with pytest.raises(ValueError, match="Unicode surrogate") as caught:
        list(_dispatcher(state, log, num_workers=num_workers).run())

    assert not isinstance(caught.value, state_module._NodeDataError)
    assert tuple(state.in_flight) == ("work",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
def test_excessively_nested_output_constructs_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
):
    value: Any = "leaf"
    for _ in range(1_200):
        value = [value]
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: value)]),
        num_workers=num_workers,
    )
    failures = _capture_semantic_failures(monkeypatch)

    events = list(_dispatcher(state, log, num_workers=num_workers).run())

    assert len(failures) == 1
    assert failures[0].code == "contract.output"
    assert failures[0].cause_type == "ValueError"
    assert "nesting exceeds" in failures[0].message
    assert isinstance(events[-2], NodeFailed)
    assert isinstance(events[-1], RunFailed)
    assert isinstance(tuple(log)[-1].record, NodeFailedRecord)


@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
def test_invalid_output_metadata_constructs_contract_failure_with_node_origin(
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
):
    node_origin = OriginRef("nodes/work")
    state, log = _initialized(
        _flow([_InvalidCommitTargetNode("work", origin=node_origin)]),
        num_workers=num_workers,
    )
    failures = _capture_semantic_failures(monkeypatch)

    events = list(_dispatcher(state, log, num_workers=num_workers).run())

    assert len(failures) == 1
    assert failures[0].code == "contract.output"
    assert failures[0].node_id == "work"
    assert failures[0].trace == (node_origin,)
    assert failures[0].cause_type == "TypeError"
    assert isinstance(events[-2], NodeFailed)
    assert events[-2].failure.node_id == "work"
    assert events[-2].failure.message == "commit_as must be a string or None"
    assert events[-2].failure.cause_type == "TypeError"
    assert events[-2].locator == engine_module.SourceSpan(
        node="work",
        kind="field",
        key="output",
    )
    assert isinstance(events[-1], RunFailed)
    assert isinstance(tuple(log)[-1].record, NodeFailedRecord)


def test_route_raw_outcome_is_prepared_committed_and_reduced():
    ran = []
    router = _RouteNode("route")
    chosen = _ActionNode("chosen", lambda: ran.append("chosen") or "yes")
    skipped = _ActionNode("skipped", lambda: ran.append("skipped") or "no")
    nodes = {
        START_ID: StartNode(START_ID),
        router.id: router,
        chosen.id: chosen,
        skipped.id: skipped,
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    flow = CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge("start-route", START_ID, router.id, optional=True, ordering=True),
            Edge("route-chosen", router.id, chosen.id, source_handle="chosen"),
            Edge("route-skipped", router.id, skipped.id, source_handle="other"),
            Edge("chosen-end", chosen.id, END_ID, optional=True, ordering=True),
            Edge("skipped-end", skipped.id, END_ID, optional=True, ordering=True),
        ],
    )
    state, log = _initialized(flow, num_workers=0)

    events = list(_dispatcher(state, log, num_workers=0).run())

    assert ran == ["chosen"]
    assert any(isinstance(event, NodeRouted) for event in events)
    assert isinstance(state.result, RunSucceeded)


def test_invalid_route_constructs_contract_failure_with_node_origin(
    monkeypatch: pytest.MonkeyPatch,
):
    outer_origin = OriginRef("nodes/outer")
    node_origin = OriginRef("nodes/route")
    node = _InvalidRouteNode("route", origin=node_origin)
    node.origin_context = (outer_origin,)
    state, log = _initialized(
        _flow([node]),
        num_workers=0,
    )
    failures = _capture_semantic_failures(monkeypatch)

    events = list(_dispatcher(state, log, num_workers=0).run())

    assert len(failures) == 1
    assert failures[0].code == "contract.route"
    assert failures[0].node_id == "route"
    assert failures[0].trace == (outer_origin, node_origin)
    assert failures[0].cause_type == "ValueError"
    assert isinstance(events[-2], NodeFailed)
    assert events[-2].failure.node_id == "route"
    assert events[-2].failure.cause_type == "ValueError"
    assert "matches no outgoing edge" in events[-2].failure.message
    assert isinstance(events[-1], RunFailed)
    assert isinstance(tuple(log)[-1].record, NodeFailedRecord)


def test_source_free_route_rejection_has_an_empty_trace(
    monkeypatch: pytest.MonkeyPatch,
):
    """Python-authored nodes do not invent source provenance."""
    state, log = _initialized(_flow([_InvalidRouteNode("route")]), num_workers=0)
    failures = _capture_semantic_failures(monkeypatch)

    list(_dispatcher(state, log, num_workers=0).run())

    assert len(failures) == 1
    assert failures[0].code == "contract.route"
    assert failures[0].trace == ()


def test_grow_raw_outcome_uses_injected_builder_and_flat_record():
    state, log = _initialized(_flow([_GrowNode("spawn")]), num_workers=0)
    dispatcher = FlowEngine(
        state,
        log,
        grow_builder=lambda spawner, _seed: _growth_fragment(spawner.id),
        num_workers=0,
    )

    events = list(dispatcher.run())

    assert any(isinstance(event, NodeExpanded) for event in events)
    expanded = [
        envelope.record
        for envelope in log
        if isinstance(envelope.record, NodeExpandedRecord)
    ]
    assert len(expanded) == 1
    assert expanded[0].spawner_id == "spawn"
    assert expanded[0].seed == {"canonical": [1, 2]}
    assert isinstance(state.result, RunSucceeded)


def test_invalid_live_grow_depth_trait_remains_a_host_failure():
    spawner = _GrowNode("spawn")
    spawner.grow_depth_delta = "invalid"
    state, log = _initialized(_flow([spawner]), num_workers=0)

    with pytest.raises(ValueError, match="grow_depth_delta") as caught:
        list(
            FlowEngine(
                state,
                log,
                grow_builder=lambda node, _seed: _growth_fragment(node.id),
                num_workers=0,
            ).run()
        )

    assert type(caught.value) is ValueError
    assert tuple(state.in_flight) == ("spawn",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_growth_digest_fault_remains_exact_host_failure(
    monkeypatch,
    num_workers: int,
    error_type: type[Exception],
):
    expected = error_type("digest programming fault")
    state, log = _initialized(_flow([_GrowNode("spawn")]), num_workers=num_workers)

    def broken_digest(_fragment):
        raise expected

    monkeypatch.setattr(state_module, "expansion_digest", broken_digest)

    with pytest.raises(error_type) as caught:
        list(
            FlowEngine(
                state,
                log,
                grow_builder=lambda node, _seed: _growth_fragment(node.id),
                num_workers=num_workers,
            ).run()
        )

    assert caught.value is expected
    assert tuple(state.in_flight) == ("spawn",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_injected_grow_builder_fault_remains_exact_host_failure(
    num_workers: int,
    error_type: type[Exception],
):
    expected = error_type("builder programming fault")
    state, log = _initialized(
        _flow([_GrowNode("spawn")]),
        num_workers=num_workers,
    )

    def reject_builder(_spawner, _seed):
        raise expected

    with pytest.raises(error_type) as caught:
        list(
            FlowEngine(
                state,
                log,
                grow_builder=reject_builder,
                num_workers=num_workers,
            ).run()
        )

    assert caught.value is expected
    assert tuple(state.in_flight) == ("spawn",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
def test_invalid_grow_builder_return_remains_a_host_failure(num_workers: int):
    state, log = _initialized(
        _flow([_GrowNode("spawn")]),
        num_workers=num_workers,
    )

    with pytest.raises(TypeError, match="grow_builder must return a Flow") as caught:
        list(
            FlowEngine(
                state,
                log,
                grow_builder=lambda _spawner, _seed: object(),
                num_workers=num_workers,
            ).run()
        )

    assert type(caught.value) is TypeError
    assert tuple(state.in_flight) == ("spawn",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


@pytest.mark.parametrize(
    ("origin", "origin_context", "expected_trace"),
    [
        (
            OriginRef("nodes/spawn"),
            (OriginRef("nodes/outer"),),
            (OriginRef("nodes/outer"), OriginRef("nodes/spawn")),
        ),
        (None, (), ()),
    ],
    ids=("authored", "source-free"),
)
def test_legal_kind_malformed_seed_constructs_contract_growth_failure(
    monkeypatch: pytest.MonkeyPatch,
    origin: OriginRef | None,
    origin_context: tuple[OriginRef, ...],
    expected_trace: tuple[OriginRef, ...],
):
    from agent_composer.compile.expand import build_grow

    node = _MalformedSeedGrowNode("spawn", origin=origin)
    node.origin_context = origin_context
    state, log = _initialized(
        _flow([node]),
        num_workers=0,
    )
    failures = _capture_semantic_failures(monkeypatch)

    events = list(
        FlowEngine(
            state,
            log,
            grow_builder=build_grow,
            num_workers=0,
        ).run()
    )

    assert len(failures) == 1
    assert failures[0].code == "contract.growth"
    assert failures[0].node_id == "spawn"
    assert failures[0].trace == expected_trace
    assert failures[0].cause_type == "TypeError"
    assert isinstance(events[-2], NodeFailed)
    assert isinstance(events[-1], RunFailed)


@pytest.mark.parametrize("num_workers", [0, 1], ids=("serial", "pooled"))
def test_growth_materialization_fault_remains_exact_host_failure(
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
):
    expected = RuntimeError("growth materialization fault")
    state, log = _initialized(
        _flow([_GrowNode("spawn")]),
        num_workers=num_workers,
    )

    def broken_materialization(_plan):
        raise expected

    monkeypatch.setattr(state_module, "_materialize_growth", broken_materialization)

    with pytest.raises(RuntimeError) as caught:
        list(
            FlowEngine(
                state,
                log,
                grow_builder=lambda node, _seed: _growth_fragment(node.id),
                num_workers=num_workers,
            ).run()
        )

    assert caught.value is expected
    assert tuple(state.in_flight) == ("spawn",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


def test_grow_builder_fault_drains_pooled_sibling_and_preserves_identity():
    entered = threading.Event()
    released = threading.Event()
    expected = RuntimeError("builder programming fault")
    state, log = _initialized(
        _flow(
            [
                _GrowNode("spawn"),
                _StopAwareNode("blocked", entered, released),
            ]
        ),
        num_workers=2,
    )

    def reject_builder(_spawner, _seed):
        assert entered.wait(2)
        raise expected

    with pytest.raises(RuntimeError) as caught:
        list(
            FlowEngine(
                state,
                log,
                grow_builder=reject_builder,
                num_workers=2,
            ).run()
        )

    assert caught.value is expected
    assert released.is_set()
    assert tuple(state.in_flight) == ("spawn", "blocked")
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


def test_positive_worker_count_never_exceeds_the_submission_bound():
    lock = threading.Lock()
    active = 0
    maximum = 0

    def work():
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return "done"

    state, log = _initialized(
        _flow([_ActionNode(f"work-{index}", work) for index in range(5)]),
        num_workers=2,
    )

    list(_dispatcher(state, log, num_workers=2).run())

    assert maximum == 2


def test_parallel_outcomes_commit_in_completion_message_order():
    fast_committed = threading.Event()

    def slow():
        assert fast_committed.wait(2)
        return "slow"

    def fast():
        return "fast"

    def observe(event):
        if isinstance(event, NodeSucceeded) and event.node_id == "fast":
            fast_committed.set()

    state, log = _initialized(
        _flow([_ActionNode("slow", slow), _ActionNode("fast", fast)]),
        num_workers=2,
    )

    list(_dispatcher(state, log, num_workers=2, on_event=observe).run())

    completed = [
        envelope.record.node_id
        for envelope in log
        if isinstance(envelope.record, NodeSucceededRecord)
        and envelope.record.node_id in {"slow", "fast"}
    ]
    assert completed == ["fast", "slow"]


def test_submit_failure_is_exact_and_leaves_the_started_node_recoverable(monkeypatch):
    expected = RuntimeError("submit rejected")
    shutdown = []

    class RejectingExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, *_args, **_kwargs):
            raise expected

        def shutdown(self, *, wait, cancel_futures):
            shutdown.append((wait, cancel_futures))

    monkeypatch.setattr(engine_module, "ThreadPoolExecutor", RejectingExecutor)
    state, log = _initialized(
        _flow([_ActionNode("never", lambda: "unexpected")]),
        num_workers=1,
    )

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=1).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == (START_ID,)
    assert shutdown == [(True, False)]


def test_event_callback_failure_signals_drains_and_reraises_same_exception():
    entered = threading.Event()
    released = threading.Event()
    expected = RuntimeError("observer failed")

    def trigger():
        assert entered.wait(2)
        return "triggered"

    state, log = _initialized(
        _flow(
            [
                _ActionNode("trigger", trigger),
                _StopAwareNode("blocked", entered, released),
            ]
        ),
        num_workers=2,
    )

    def reject(event):
        if isinstance(event, NodeSucceeded) and event.node_id == "trigger":
            raise expected

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=2, on_event=reject).run())

    assert caught.value is expected
    assert released.is_set()
    assert "blocked" in state.in_flight
    assert state.result is None


def test_cleanup_failure_never_replaces_callback_error_and_drain_is_attempted(
    monkeypatch,
):
    entered = threading.Event()
    released = threading.Event()
    expected = RuntimeError("observer failed first")
    cleanup = RuntimeError("executor shutdown failed afterward")
    real_executor = engine_module.ThreadPoolExecutor

    class HostileShutdownExecutor:
        """Join real workers, then raise to probe first-cause preservation."""

        def __init__(self, **kwargs):
            self._delegate = real_executor(**kwargs)

        def submit(self, *args, **kwargs):
            return self._delegate.submit(*args, **kwargs)

        def shutdown(self, *, wait, cancel_futures):
            self._delegate.shutdown(wait=wait, cancel_futures=cancel_futures)
            raise cleanup

    monkeypatch.setattr(
        engine_module,
        "ThreadPoolExecutor",
        HostileShutdownExecutor,
    )

    def trigger():
        assert entered.wait(2)
        return "triggered"

    state, log = _initialized(
        _flow(
            [
                _ActionNode("trigger", trigger),
                _StopAwareNode("blocked", entered, released),
            ]
        ),
        num_workers=2,
    )

    def reject(event):
        if isinstance(event, NodeSucceeded) and event.node_id == "trigger":
            raise expected

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=2, on_event=reject).run())

    assert caught.value is expected
    assert released.is_set()
    assert any(
        "executor shutdown failed afterward" in note
        for note in getattr(expected, "__notes__", ())
    )


def test_started_observer_failure_prevents_submission_of_that_node():
    executed = []
    expected = RuntimeError("started observer failed")
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work") or "done")]),
        num_workers=1,
    )

    def reject(event):
        if isinstance(event, NodeStarted) and event.node_id == "work":
            raise expected

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=1, on_event=reject).run())

    assert caught.value is expected
    assert executed == []
    assert tuple(state.in_flight) == ("work",)


def test_primary_host_interruption_is_preserved_after_full_sibling_drain():
    entered = threading.Event()
    released = threading.Event()
    primary = _HostStop("terminal EOF", primary=True)
    state, log = _initialized(
        _flow(
            [
                _StopAwareNode("blocked", entered, released),
                _PrimaryHostStopNode("interrupt", primary, wait_for=entered),
            ]
        ),
        num_workers=2,
    )

    with pytest.raises(_HostStop) as caught:
        list(_dispatcher(state, log, num_workers=2).run())

    assert caught.value is primary
    assert entered.is_set()
    assert released.is_set()
    assert state.result is None


def test_node_failure_drains_sibling_and_discards_late_stream_chunks():
    failure_may_run = threading.Event()
    chunks = []

    def fail():
        assert failure_may_run.wait(2)
        raise RuntimeError("boom")

    state, log = _initialized(
        _flow(
            [
                _LateStreamNode("stream", failure_may_run),
                _ActionNode("fail", fail),
            ]
        ),
        num_workers=2,
    )

    events = list(
        _dispatcher(state, log, num_workers=2, on_stream=chunks.append).run()
    )

    assert [chunk.chunk for chunk in chunks] == ["early"]
    assert isinstance(events[-2], NodeFailed)
    assert events[-2].failure.node_id == "fail"
    assert isinstance(events[-1], RunFailed)
    assert tuple(state.in_flight) == ("stream",)
    assert not any(
        isinstance(envelope.record, NodeSucceededRecord)
        and envelope.record.node_id == "stream"
        for envelope in log
    )


def test_stream_callback_failure_is_host_error_not_durable_node_failure():
    expected = RuntimeError("stream callback failed")
    failure_may_run = threading.Event()
    state, log = _initialized(
        _flow([_LateStreamNode("stream", failure_may_run)]),
        num_workers=1,
    )

    def reject(_chunk):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=1, on_stream=reject).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == ("stream",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


def test_authority_mismatch_is_rejected_before_any_node_executes():
    executed = []
    state, _ = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work"))]),
        num_workers=0,
    )
    other_log = MemoryRunLog("other-run")

    with pytest.raises(ValueError, match="run_id mismatch"):
        list(_dispatcher(state, other_log, num_workers=0).run())

    assert executed == []
    assert other_log.next_sequence == 0


def test_dispatcher_borrows_and_never_closes_its_log():
    log = _GuardedLog(_RUN_ID)
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
        log=log,
    )

    list(_dispatcher(state, log, num_workers=0).run())

    assert log.close_calls == 0
    assert isinstance(state.result, RunSucceeded)
    log.close()
    assert log.close_calls == 1
