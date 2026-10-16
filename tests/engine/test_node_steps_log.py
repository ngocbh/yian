"""Node steps: the run record a running node appends, and how the reducer folds it."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from yian.compile.expand import build_grow
from yian.compile.model import START_ID
from yian.defaults import current
from yian.nodes.base import Grow
from yian.runtime.log import (
    SCHEMA_VERSION,
    NodeStartedRecord,
    NodeStepRecord,
    RecordEnvelope,
    decode_record,
    encode_record,
)
from yian.runtime.state import (
    RunState,
    _replay_records,
    commit_prepared,
    prepare_expanded,
    prepare_node_failed,
    prepare_node_routed,
    prepare_node_started,
    prepare_node_step,
    prepare_node_succeeded,
    prepare_run_resumed,
)
from tests.engine._graph_builder import _forward_capture
from tests.engine.test_run_state_growth import _fragment, _initialized_spawner
from tests.engine.test_run_state_outcomes import (
    _commit_start_success,
    _fanout_flow,
    _initialized_and_started,
    _node_failure,
    _route_in_flight,
)


_FAILED = {
    "event": "failed",
    "attempt": 1,
    "error": {"type": "TimeoutError", "message": "slow"},
    "retry_in": 1.5,
}


def _first_in_flight(*, learnable: bool = False):
    """Return a state/log with START done and node ``first`` in flight."""
    state, log = _initialized_and_started(_fanout_flow(), learnable=learnable)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "first"))
    return state, log


def _nested(levels: int) -> dict:
    """Return a JSON object nested ``levels`` objects deep, itself included."""
    value: dict = {}
    for _ in range(levels - 1):
        value = {"inner": value}
    return value


def test_node_step_round_trips_under_its_wire_kind():
    record = NodeStepRecord("a", _FAILED)
    envelope = RecordEnvelope(schema=SCHEMA_VERSION, run_id="run-1", seq=3, record=record)

    encoded = encode_record(envelope)
    decoded = decode_record(encoded)

    assert decoded == envelope
    assert encode_record(decoded) == encoded
    wire = json.loads(encoded)
    assert wire["kind"] == "node_step"
    assert wire["schema"] == 9
    assert wire["payload"] == {"node_id": "a", "step": _FAILED}


def test_node_step_record_detaches_the_callers_step():
    step = {"event": "started", "attempt": 2, "tags": ("x", "y")}
    record = NodeStepRecord("a", step)
    step["attempt"] = 99

    assert record.step == {"attempt": 2, "event": "started", "tags": ["x", "y"]}


@pytest.mark.parametrize(
    ("node_id", "step"),
    [
        ("a", ["not", "an", "object"]),
        ("a", "text"),
        ("a", None),
        ("a", {"retry_in": float("nan")}),
        ("a", {"retry_in": float("inf")}),
        ("a", {"nested": {1: "non-text key"}}),
        ("a", {"when": object()}),
        ("a", {"items": {1, 2}}),
        ("", {"event": "started"}),
        (1, {"event": "started"}),
    ],
    ids=[
        "list", "text", "none", "nan", "infinity", "non-text-key", "object",
        "set", "empty-node-id", "non-text-node-id",
    ],
)
def test_node_step_record_rejects_invalid_values(node_id, step):
    with pytest.raises((TypeError, ValueError)):
        NodeStepRecord(node_id, step)


@pytest.mark.parametrize(
    "payload",
    [
        {"node_id": "a", "step": [1]},
        {"node_id": "a", "step": "text"},
        {"node_id": "", "step": {}},
        {"node_id": "a"},
        {"node_id": "a", "step": {}, "extra": True},
    ],
    ids=["list", "text", "empty-node-id", "missing-step", "extra-field"],
)
def test_node_step_wire_payload_rejects_invalid_shapes(payload):
    data = json.dumps({
        "schema": SCHEMA_VERSION, "run_id": "run-1", "seq": 0,
        "kind": "node_step", "payload": payload,
    })
    with pytest.raises((TypeError, ValueError)):
        decode_record(data)


def test_node_step_wire_payload_rejects_non_finite_numbers():
    data = (
        '{"schema":9,"run_id":"run-1","seq":0,"kind":"node_step",'
        '"payload":{"node_id":"a","step":{"retry_in":NaN}}}'
    )
    with pytest.raises(ValueError):
        decode_record(data)


def test_node_step_too_deep_for_python_is_a_value_error():
    with pytest.raises(ValueError):
        NodeStepRecord("a", _nested(5000))


def test_prepare_node_step_appends_in_order_without_observers():
    state, log = _first_in_flight()
    started = {"event": "started", "attempt": 2}

    first = prepare_node_step(state, "first", _FAILED)
    assert first.observers == ()
    assert first.record == NodeStepRecord("first", _FAILED)
    assert "first" not in state.node_steps
    assert commit_prepared(log, state, first) == ()
    assert commit_prepared(log, state, prepare_node_step(state, "first", started)) == ()

    assert state.node_steps == {"first": (_FAILED, started)}
    assert type(state.node_steps["first"]) is tuple
    assert state.node_starts["first"] == 1
    assert [type(envelope.record) for envelope in log][-2:] == [NodeStepRecord, NodeStepRecord]
    assert state.next_sequence == log.next_sequence


def test_committed_step_is_detached_from_the_caller_and_the_record():
    state, log = _first_in_flight()
    step = {"event": "failed", "error": {"type": "E", "message": "m"}}

    prepared = prepare_node_step(state, "first", step)
    commit_prepared(log, state, prepared)
    step["error"]["type"] = "changed"
    prepared.record.step["error"]["type"] = "changed"

    assert state.node_steps["first"] == ({"error": {"message": "m", "type": "E"}, "event": "failed"},)


@pytest.mark.parametrize("node_id", ["second", "missing", START_ID])
def test_prepare_node_step_requires_an_in_flight_node(node_id):
    state, log = _first_in_flight()
    before = log.next_sequence

    with pytest.raises(ValueError):
        prepare_node_step(state, node_id, {"event": "started"})

    assert log.next_sequence == before
    assert state.node_steps == {}


def test_prepare_node_step_rejects_a_terminal_state():
    state, log = _first_in_flight()
    commit_prepared(log, state, prepare_node_failed(state, _node_failure("first")))
    assert state.result is not None

    with pytest.raises(ValueError):
        prepare_node_step(state, "second", {"event": "started"})


def test_prepare_node_step_rejects_an_uninitialized_state():
    with pytest.raises(ValueError):
        prepare_node_step(RunState(_fanout_flow()), START_ID, {"event": "started"})


def test_prepare_node_step_rejects_a_step_past_the_nesting_limit():
    state, _ = _first_in_flight()
    limit = current().max_value_nesting

    prepare_node_step(state, "first", _nested(limit))
    with pytest.raises(ValueError, match="nests deeper"):
        prepare_node_step(state, "first", _nested(limit + 1))


@pytest.mark.parametrize("step", [["x"], {"n": float("nan")}, {1: "x"}])
def test_prepare_node_step_rejects_invalid_steps(step):
    state, _ = _first_in_flight()
    with pytest.raises((TypeError, ValueError)):
        prepare_node_step(state, "first", step)


def test_node_starts_count_started_records():
    state, log = _initialized_and_started(_fanout_flow())
    assert state.node_starts == {START_ID: 1}
    _commit_start_success(state, log)
    assert state.node_starts == {}
    commit_prepared(log, state, prepare_node_started(state, "first"))
    commit_prepared(log, state, prepare_node_started(state, "second"))
    assert state.node_starts == {"first": 1, "second": 1}


def test_resume_keeps_steps_and_a_resumed_start_counts_two():
    state, log = _first_in_flight()
    commit_prepared(log, state, prepare_node_step(state, "first", _FAILED))

    commit_prepared(log, state, prepare_run_resumed(state, num_workers=0))
    assert state.node_steps == {"first": (_FAILED,)}
    assert state.node_starts == {"first": 1}
    assert "first" not in state.in_flight

    # A resumed node is ready again, not in flight, so it cannot record a step yet.
    with pytest.raises(ValueError):
        prepare_node_step(state, "first", {"event": "started"})
    commit_prepared(log, state, prepare_node_started(state, "first"))
    assert state.node_starts == {"first": 2}
    assert state.node_steps == {"first": (_FAILED,)}


def _settle_succeeded(state, log, node_id):
    commit_prepared(log, state, prepare_node_succeeded(state, node_id, {"done": True}))


def _settle_failed(state, log, node_id):
    commit_prepared(log, state, prepare_node_failed(state, _node_failure(node_id)))


def test_success_deletes_the_nodes_steps_and_starts():
    state, log = _first_in_flight()
    commit_prepared(log, state, prepare_node_step(state, "first", _FAILED))
    commit_prepared(log, state, prepare_node_started(state, "second"))
    commit_prepared(log, state, prepare_node_step(state, "second", _FAILED))

    _settle_succeeded(state, log, "first")

    assert state.node_steps == {"second": (_FAILED,)}
    assert state.node_starts == {"second": 1}


def test_failure_deletes_the_nodes_steps_and_starts():
    state, log = _first_in_flight()
    commit_prepared(log, state, prepare_node_step(state, "first", _FAILED))

    _settle_failed(state, log, "first")

    assert "first" not in state.node_steps
    assert "first" not in state.node_starts


def test_route_deletes_the_nodes_steps_and_starts():
    state, log = _route_in_flight()
    commit_prepared(log, state, prepare_node_step(state, "router", _FAILED))
    assert state.node_starts == {"router": 1}

    commit_prepared(log, state, prepare_node_routed(state, "router", "yes"))

    assert state.node_steps == {}
    assert state.node_starts == {}


def test_expansion_deletes_the_spawners_steps_and_starts():
    state, log = _initialized_spawner()
    commit_prepared(log, state, prepare_node_step(state, "spawn", _FAILED))
    assert state.node_starts == {"spawn": 1}

    commit_prepared(
        log, state,
        prepare_expanded(state, "spawn", Grow(seed={}), lambda spawner, seed: _fragment(spawner.id)),
    )

    assert state.node_steps == {}
    assert state.node_starts == {}


def test_replay_reproduces_steps_and_starts():
    state, log = _first_in_flight()
    commit_prepared(log, state, prepare_node_step(state, "first", _FAILED))
    commit_prepared(log, state, prepare_run_resumed(state, num_workers=0))
    commit_prepared(log, state, prepare_node_started(state, "first"))
    commit_prepared(log, state, prepare_node_step(state, "first", {"event": "started", "attempt": 3}))

    replayed = _replay_records(RunState(deepcopy(state.flow)), tuple(log), grow_builder=build_grow)

    assert replayed.node_steps == state.node_steps
    assert replayed.node_starts == state.node_starts == {"first": 2}
    assert replayed.next_sequence == state.next_sequence


def test_replay_without_steps_has_no_steps():
    state, log = _first_in_flight()

    replayed = _replay_records(RunState(deepcopy(state.flow)), tuple(log), grow_builder=build_grow)

    assert replayed.node_steps == {}
    assert replayed.node_starts == {"first": 1}


def test_learnable_run_accepts_a_step_with_an_empty_trace_delta():
    state, log = _first_in_flight(learnable=True)
    assert state.trace is not None

    prepared = prepare_node_step(state, "first", _FAILED)
    delta = prepared._plan.trace_delta
    assert delta is not None
    assert delta.started is None and delta.resumed is False and delta.occurrence is None
    commit_prepared(log, state, prepared)
    assert state.node_steps == {"first": (_FAILED,)}
    assert state.trace.next_sequence == state.next_sequence

    commit_prepared(
        log, state,
        prepare_node_succeeded(
            state, "first", {"done": True},
            forward_capture=_forward_capture(state.flow.nodes["first"], {}),
        ),
    )
    assert state.node_steps == {}


def test_frontier_commit_rejects_a_step_record_with_a_start_delta():
    from dataclasses import replace
    from yian.runtime.state import _NodeStartedDelta

    state, log = _first_in_flight()
    prepared = prepare_node_step(state, "first", _FAILED)
    forged = replace(prepared, _plan=replace(prepared._plan, delta=_NodeStartedDelta("first")))

    with pytest.raises(TypeError):
        commit_prepared(log, state, forged)
    assert state.node_steps == {}


def test_a_step_record_is_never_prepared_as_a_resume():
    state, log = _first_in_flight()
    prepared = prepare_node_step(state, "first", _FAILED)
    assert type(prepared._plan.delta).__name__ == "_NodeStepDelta"
    assert prepared._plan.delta.node_id == "first"
    assert prepared._plan.delta.step == _FAILED
    assert isinstance(prepared.record, NodeStepRecord)
    assert not isinstance(prepared.record, NodeStartedRecord)
