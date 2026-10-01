"""Transactional initialization tests for the authoritative run-state reducer."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, Callable

import pytest

from agent_composer.compile.model import CompiledFlow, Edge, END_ID, START_ID
from agent_composer.runtime.common import NodeState
from agent_composer.events import RunStarted
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.log import (
    MemoryRunLog,
    NodeStartedRecord,
    RecordEnvelope,
    RunStartedRecord,
    SCHEMA_VERSION,
    encode_pool,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_run_started,
)
from agent_composer.typesys import VariablePool
from tests.engine._fakes import FuncNode


_FINGERPRINT = "a" * 64
_RUN_ID = "run-123"


class _DictSubclass(dict):
    """Value-equal mapping with the wrong reducer container type."""


def _flow() -> CompiledFlow:
    """Return a small complete graph with a real synthesized START boundary."""
    nodes = {
        START_ID: StartNode(START_ID),
        "work": FuncNode("work", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-work", from_=START_ID, to="work"),
            Edge(id="work-end", from_="work", to=END_ID),
        ],
    )


def _pool(flow: CompiledFlow, *, run_id: str = _RUN_ID) -> VariablePool:
    """Build the complete pool captured by a fresh-run record."""
    pool = VariablePool(start_id=flow.start_id)
    pool.set(
        flow.start_id,
        {"name": "Ada", "count": 2, "aliases": ("A", "B")},
    )
    pool.set("preseeded", {"retained": True})
    pool.add_system("run_id", run_id)
    pool.add_system("today", "2026-08-13")
    pool.add_system("now", "2026-08-13T12:34:56+00:00")
    pool.add_system("tenant", "tests")
    return pool


def _start_data(
    flow: CompiledFlow,
    *,
    run_id: str = _RUN_ID,
    pool: VariablePool | None = None,
    num_workers: int = 3,
) -> RunStartData:
    """Return representative fresh-run data for ``flow``."""
    return RunStartData(
        run_id=run_id,
        flow_fingerprint=_FINGERPRINT,
        initial_pool=pool if pool is not None else _pool(flow, run_id=run_id),
        start_wiring_override={
            "name": "${input.name}",
            "metadata": {"labels": ["one", "two"]},
        },
        effective_llm_by_static_path={
            "agent": {
                "provider": "openai",
                "model": "gpt-test",
                "options": {"temperature": 0.2},
            }
        },
        num_workers=num_workers,
    )


def _snapshot(state: RunState) -> dict[str, Any]:
    """Return all reducer-owned values in equality-friendly form."""
    return {
        "flow": state.flow,
        "pool": state.pool.model_dump(mode="json"),
        "node_state": dict(state.node_state),
        "edge_state": dict(state.edge_state),
        "ready": tuple(state.ready),
        "in_flight": tuple(state.in_flight),
        "accepted_starts": dict(state.accepted_starts),
        "result": state.result,
        "depth": dict(state.depth),
        "wiring_overrides": deepcopy(state.wiring_overrides),
        "run_id": state.run_id,
        "flow_fingerprint": state.flow_fingerprint,
        "next_sequence": state.next_sequence,
        "num_workers": state.num_workers,
        "input": deepcopy(state.input),
        "effective_llm_by_static_path": deepcopy(
            state.effective_llm_by_static_path
        ),
    }


def test_run_state_starts_pristine_around_the_owned_graph():
    flow = _flow()

    state = RunState(flow)

    assert state.flow is flow
    assert state.pool == VariablePool(start_id=flow.start_id)
    assert state.node_state == {node_id: NodeState.UNKNOWN for node_id in flow.nodes}
    assert state.edge_state == {edge.id: NodeState.UNKNOWN for edge in flow.edges}
    assert tuple(state.ready) == ()
    assert state.in_flight == {}
    assert state.accepted_starts == {}
    assert not hasattr(state, "conversation_state")
    assert state.result is None
    assert state.depth == {}
    assert state.wiring_overrides == {}
    assert state.run_id is None
    assert state.flow_fingerprint is None
    assert state.next_sequence == 0
    assert state.num_workers == 0
    assert state.input == {}
    assert state.effective_llm_by_static_path == {}


def test_historical_start_prepares_recorded_schema_bytes():
    from agent_composer.runtime.log import decode_record, encode_record

    state = RunState(_flow())
    before = _snapshot(state)
    prepared = prepare_run_started(state, _start_data(state.flow), schema=SCHEMA_VERSION)
    expected = RecordEnvelope(SCHEMA_VERSION, _RUN_ID, 0, prepared.record)
    assert prepared._plan.envelope_bytes == encode_record(expected)
    assert decode_record(prepared._plan.envelope_bytes).schema == SCHEMA_VERSION
    assert _snapshot(state) == before


@pytest.mark.parametrize("schema", [True, 9.0, "9", 6, 7, 8, 99])
def test_historical_start_requires_an_exact_supported_schema(schema):
    state = RunState(_flow())
    before = _snapshot(state)
    with pytest.raises((TypeError, ValueError)):
        prepare_run_started(state, _start_data(state.flow), schema=schema)
    assert _snapshot(state) == before


@pytest.mark.parametrize("corruption", ["schema", "numeric_payload"])
def test_append_result_must_preserve_the_complete_validated_envelope(corruption):
    class CorruptingLog(MemoryRunLog):
        """Return a value-equal payload or invalid schema after a valid append."""

        def append(self, record):
            envelope = super().append(record)
            if corruption == "schema":
                object.__setattr__(envelope, "schema", True)
            else:
                envelope.record.start_wiring_override["numeric"] = 1.0
                assert envelope.record == record
            return envelope

    state = RunState(_flow())
    data = _start_data(state.flow)
    data.start_wiring_override["numeric"] = 1
    prepared = prepare_run_started(state, data)
    before = _snapshot(state)
    log = CorruptingLog(_RUN_ID)
    with pytest.raises((TypeError, ValueError)):
        commit_prepared(log, state, prepared)
    assert _snapshot(state) == before
    assert log.next_sequence == 1


def test_run_started_preparation_is_immutable_and_commit_restores_complete_state():
    flow = _flow()
    state = RunState(flow)
    pool = _pool(flow)
    data = _start_data(flow, pool=pool)
    before = _snapshot(state)

    prepared = prepare_run_started(state, data)

    assert _snapshot(state) == before
    assert isinstance(prepared.record, RunStartedRecord)
    assert prepared.expected_run_id == _RUN_ID
    assert prepared.expected_sequence == 0
    assert prepared.observers == (RunStarted(),)
    with pytest.raises(FrozenInstanceError):
        data.run_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        prepared.expected_sequence = 1  # type: ignore[misc]

    # Preparation owns canonical copies rather than retaining caller-owned containers.
    pool.set(flow.start_id, {"name": "mutated"})
    pool.add_system("today", "never")
    data.start_wiring_override["metadata"]["labels"].append("mutated")
    data.effective_llm_by_static_path["agent"]["options"]["temperature"] = 1.0

    log = MemoryRunLog(_RUN_ID)
    observers = commit_prepared(log, state, prepared)

    assert observers == (RunStarted(),)
    assert tuple(log) == (
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=_RUN_ID,
            seq=0,
            record=prepared.record,
        ),
    )
    assert state.run_id == _RUN_ID
    assert state.flow_fingerprint == _FINGERPRINT
    assert state.next_sequence == 1
    assert state.num_workers == 3
    assert state.pool is not pool
    assert state.pool.start_id == flow.start_id
    assert state.pool.get(flow.start_id) == {
        "name": "Ada",
        "count": 2,
        "aliases": ["A", "B"],
    }
    assert state.pool.get("preseeded") == {"retained": True}
    assert state.pool.resolve("system", ["run_id"]) == _RUN_ID
    assert state.pool.resolve("system", ["today"]) == "2026-08-13"
    assert state.pool.resolve("system", ["now"]) == "2026-08-13T12:34:56+00:00"
    assert state.pool.resolve("system", ["tenant"]) == "tests"
    assert encode_pool(state.pool) == prepared.record.initial_pool
    assert state.input == {"name": "Ada", "count": 2, "aliases": ["A", "B"]}
    assert state.wiring_overrides == {
        flow.start_id: {
            "name": "${input.name}",
            "metadata": {"labels": ["one", "two"]},
        }
    }
    assert state.effective_llm_by_static_path == {
        "agent": {
            "provider": "openai",
            "model": "gpt-test",
            "options": {"temperature": 0.2},
        }
    }
    assert tuple(state.ready) == (flow.start_id,)
    assert state.node_state[flow.start_id] is NodeState.UNKNOWN
    assert state.in_flight == {}
    state.pool.set(flow.start_id, {"name": "coerced later"})
    assert state.input == {"name": "Ada", "count": 2, "aliases": ["A", "B"]}


@pytest.mark.parametrize("num_workers", [-1, True, 1.0, "2"])
def test_prepare_run_started_requires_an_exact_nonnegative_worker_count(num_workers):
    flow = _flow()
    state = RunState(flow)
    before = _snapshot(state)

    with pytest.raises((TypeError, ValueError), match="num_workers"):
        prepare_run_started(state, _start_data(flow, num_workers=num_workers))

    assert _snapshot(state) == before


def test_prepare_run_started_requires_a_pristine_state_with_a_start_node():
    flow = _flow()
    state = RunState(flow)
    state.ready.append("work")

    with pytest.raises(ValueError, match="pristine"):
        prepare_run_started(state, _start_data(flow))

    missing_start = _flow()
    missing_start.nodes.pop(missing_start.start_id)
    missing_state = RunState(missing_start)
    with pytest.raises(ValueError, match="START"):
        prepare_run_started(missing_state, _start_data(flow))


def test_prepare_run_started_rejects_pool_identity_and_start_priming_mismatches():
    flow = _flow()

    wrong_start = _pool(flow)
    wrong_start.start_id = "other"
    with pytest.raises(ValueError, match="start_id"):
        prepare_run_started(RunState(flow), _start_data(flow, pool=wrong_start))

    wrong_run = _pool(flow, run_id="different")
    with pytest.raises(ValueError, match="system.run_id"):
        prepare_run_started(RunState(flow), _start_data(flow, pool=wrong_run))

    scalar_start = _pool(flow)
    scalar_start.set(flow.start_id, "not a record")
    with pytest.raises(ValueError, match="START.*dict"):
        prepare_run_started(RunState(flow), _start_data(flow, pool=scalar_start))


@pytest.mark.parametrize(
    "data_factory",
    [
        lambda flow: replace(_start_data(flow), flow_fingerprint="not-a-digest"),
        lambda flow: _start_data(flow, run_id=""),
        lambda flow: replace(
            _start_data(flow), effective_llm_by_static_path={"agent": "bad"}
        ),
    ],
)
def test_prepare_run_started_validates_the_public_envelope_contract(
    data_factory: Callable[[CompiledFlow], RunStartData],
):
    flow = _flow()
    data = data_factory(flow)

    with pytest.raises((TypeError, ValueError)):
        prepare_run_started(RunState(flow), data)


class _InspectingLog(MemoryRunLog):
    """Memory log that proves the reducer has not applied before append."""

    def __init__(self, state: RunState) -> None:
        super().__init__(_RUN_ID)
        self._state = state
        self.saw_append = False

    def append(self, record):
        assert self._state.run_id is None
        assert self._state.next_sequence == 0
        assert tuple(self._state.ready) == ()
        assert self._state.input == {}
        self.saw_append = True
        return super().append(record)


def test_commit_prepared_appends_before_applying_the_total_plan():
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    log = _InspectingLog(state)

    observers = commit_prepared(log, state, prepared)

    assert log.saw_append
    assert state.run_id == _RUN_ID
    assert observers == (RunStarted(),)


class _FailingLog(MemoryRunLog):
    """Memory log whose append boundary fails before retaining a record."""

    def append(self, record):
        raise OSError("disk full")


def test_commit_prepared_append_failure_leaves_state_and_observers_untouched():
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    before = _snapshot(state)

    with pytest.raises(OSError, match="disk full"):
        commit_prepared(_FailingLog(_RUN_ID), state, prepared)

    assert _snapshot(state) == before
    assert prepared.observers == (RunStarted(),)


def test_commit_prepared_rejects_mutation_of_the_public_prepared_record():
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    assert isinstance(prepared.record, RunStartedRecord)
    prepared.record.start_wiring_override["name"] = "changed"
    before = _snapshot(state)
    log = MemoryRunLog(_RUN_ID)

    with pytest.raises(ValueError, match="changed after validation"):
        commit_prepared(log, state, prepared)

    assert _snapshot(state) == before
    assert log.next_sequence == 0


def test_private_plan_contains_no_mutable_transition_payload():
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    plan = prepared._plan

    assert plan.originating_state is state
    assert plan.originating_flow is flow
    assert isinstance(plan.envelope_bytes, bytes)

    def assert_deeply_immutable(value: Any) -> None:
        if isinstance(value, tuple):
            for item in value:
                assert_deeply_immutable(item)
            return
        assert isinstance(value, (bytes, str, int, bool, type(None)))

    for name in plan.__slots__:
        if name not in {"originating_state", "originating_flow"}:
            assert_deeply_immutable(getattr(plan, name))
    with pytest.raises(FrozenInstanceError):
        plan.envelope_bytes = b"changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.envelope_bytes[0] = 0  # type: ignore[index]


def test_commit_prepared_rejects_a_different_pristine_state_before_append():
    flow = _flow()
    originating_state = RunState(flow)
    other_state = RunState(flow)
    prepared = prepare_run_started(originating_state, _start_data(flow))
    log = MemoryRunLog(_RUN_ID)
    before = _snapshot(other_state)

    with pytest.raises(ValueError, match="originating RunState"):
        commit_prepared(log, other_state, prepared)

    assert _snapshot(other_state) == before
    assert log.next_sequence == 0


@pytest.mark.parametrize(
    "replace_boundary",
    [
        lambda state: setattr(state, "flow", _flow()),
        lambda state: setattr(state.flow, "start_id", "work"),
        lambda state: state.flow.nodes.__setitem__(START_ID, StartNode(START_ID)),
        lambda state: state.flow.nodes.__setitem__(
            END_ID, EndNode.record(END_ID, output_names=[])
        ),
    ],
    ids=["flow-object", "start-id", "start-node", "end-node"],
)
def test_commit_prepared_rejects_flow_or_boundary_replacement_before_append(
    replace_boundary: Callable[[RunState], None],
):
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    replace_boundary(state)
    log = MemoryRunLog(_RUN_ID)

    with pytest.raises(ValueError, match="originating flow"):
        commit_prepared(log, state, prepared)

    assert log.next_sequence == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.ready.append("work"),
        lambda state: setattr(state, "ready", ()),
        lambda state: setattr(state, "in_flight", set()),
        lambda state: setattr(state, "node_state", _DictSubclass(state.node_state)),
        lambda state: state.node_state.__setitem__("work", "unknown"),
        lambda state: state.node_state.__setitem__("work", NodeState.TAKEN),
        lambda state: state.edge_state.__setitem__("start-work", NodeState.TAKEN),
        lambda state: state.wiring_overrides.__setitem__("work", {}),
        lambda state: state.pool.add_system("stale", True),
    ],
    ids=[
        "ready-frontier",
        "ready-container",
        "in-flight-container",
        "node-state-container",
        "node-disposition-type",
        "node-disposition",
        "edge-disposition",
        "wiring",
        "pool",
    ],
)
def test_commit_prepared_rejects_every_stale_state_mutation_before_append(
    mutate: Callable[[RunState], None],
):
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    mutate(state)
    before = _snapshot(state)
    log = MemoryRunLog(_RUN_ID)

    with pytest.raises(ValueError, match="pristine"):
        commit_prepared(log, state, prepared)

    assert _snapshot(state) == before
    assert log.next_sequence == 0


def test_committed_state_does_not_alias_old_containers_or_prepared_payloads():
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    assert isinstance(prepared.record, RunStartedRecord)
    old_ready = state.ready
    old_pool = state.pool
    old_in_flight = state.in_flight
    old_node_state = state.node_state
    old_edge_state = state.edge_state
    old_depth = state.depth
    old_wiring = state.wiring_overrides
    old_input = state.input
    old_effective = state.effective_llm_by_static_path

    commit_prepared(MemoryRunLog(_RUN_ID), state, prepared)
    committed = _snapshot(state)

    old_ready.append("old-ready")
    old_pool.add_system("old", True)
    old_in_flight["old-flight"] = None
    old_node_state["work"] = NodeState.TAKEN
    old_edge_state["start-work"] = NodeState.TAKEN
    old_depth["work"] = 9
    old_wiring["work"] = {"x": 1}
    old_input["old"] = True
    old_effective["old"] = {"model": "wrong"}
    prepared.record.initial_pool["system"].clear()
    prepared.record.start_wiring_override["metadata"]["labels"].append("wrong")
    prepared.record.effective_llm_by_static_path["agent"]["model"] = "wrong"

    assert _snapshot(state) == committed
    assert state.ready is not old_ready
    assert state.pool is not old_pool
    assert state.in_flight is not old_in_flight
    assert state.node_state is not old_node_state
    assert state.edge_state is not old_edge_state
    assert state.depth is not old_depth
    assert state.wiring_overrides is not old_wiring
    assert state.input is not old_input
    assert state.effective_llm_by_static_path is not old_effective


@pytest.mark.parametrize(
    "mismatch",
    ["log_run_id", "log_sequence", "state_sequence", "state_run_id"],
)
def test_commit_prepared_rejects_state_and_log_identity_mismatches_before_append(
    mismatch: str,
):
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    log = MemoryRunLog("other" if mismatch == "log_run_id" else _RUN_ID)
    if mismatch == "log_sequence":
        log.append(NodeStartedRecord(node_id="already-there"))
    elif mismatch == "state_sequence":
        state.next_sequence = 1
    elif mismatch == "state_run_id":
        state.run_id = "already-started"
    before = _snapshot(state)
    before_sequence = log.next_sequence

    with pytest.raises(ValueError, match="mismatch|RunStarted"):
        commit_prepared(log, state, prepared)

    assert _snapshot(state) == before
    assert log.next_sequence == before_sequence


class _ReturningLog:
    """Minimal custom log that returns a caller-selected append result."""

    path: Path | None = None

    def __init__(self, result: Callable[[RunStartedRecord], RecordEnvelope]) -> None:
        self.run_id = _RUN_ID
        self.next_sequence = 0
        self._result = result

    def append(self, record: RunStartedRecord) -> RecordEnvelope:
        return self._result(record)


class _RetainingMismatchedLog(MemoryRunLog):
    """Retain the right record but violate the append-return protocol."""

    def append(self, record):
        retained = super().append(record)
        return RecordEnvelope(
            schema=retained.schema,
            run_id=retained.run_id,
            seq=retained.seq + 1,
            record=retained.record,
        )


@pytest.mark.parametrize(
    "result",
    [
        lambda record: RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id="other",
            seq=0,
            record=record,
        ),
        lambda record: RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=_RUN_ID,
            seq=1,
            record=record,
        ),
        lambda record: RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=_RUN_ID,
            seq=0,
            record=NodeStartedRecord(node_id="different"),
        ),
        lambda record: RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=_RUN_ID,
            seq=0,
            record=record,
        ),
    ],
    ids=["run-id", "sequence", "record", "log-frontier"],
)
def test_commit_prepared_rejects_a_custom_log_return_mismatch_without_applying(
    result: Callable[[RunStartedRecord], RecordEnvelope],
):
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    before = _snapshot(state)

    with pytest.raises(ValueError, match="append result"):
        commit_prepared(_ReturningLog(result), state, prepared)  # type: ignore[arg-type]

    assert _snapshot(state) == before


def test_mismatched_append_return_leaves_the_advanced_log_as_recovery_authority():
    flow = _flow()
    state = RunState(flow)
    prepared = prepare_run_started(state, _start_data(flow))
    before = _snapshot(state)
    log = _RetainingMismatchedLog(_RUN_ID)

    with pytest.raises(ValueError, match="append result"):
        commit_prepared(log, state, prepared)

    assert _snapshot(state) == before
    assert log.next_sequence == 1
    retained = tuple(log)
    assert retained == (
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=_RUN_ID,
            seq=0,
            record=prepared.record,
        ),
    )
    assert isinstance(retained[0].record, RunStartedRecord)
