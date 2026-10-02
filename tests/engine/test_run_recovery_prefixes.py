"""Silent authoritative-log replay over exact reducer prefixes."""

from __future__ import annotations

import copy
from typing import Any, Iterator

import pytest

import agent_composer.run as run_module
import agent_composer.runtime.state as state_module
from agent_composer.compile.expand import build_grow
from agent_composer.compile.llm_cascade import resolve_llm_cascade
from agent_composer.compile.manifest import flow_fingerprint, flow_manifest
from agent_composer.compile.model import END_ID, START_ID
from agent_composer.compose import load_flow
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Grow
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
    prepare_expanded,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_run_started,
)
from agent_composer.typesys import VariablePool


# Fixed writer output; keep these bytes independent of the current codec.
_RECORDED_LINEAR = (
    (
        b'{"kind":"run_started","payload":{"effective_llm_by_static_path":{},'
        b'"flow_fingerprint":"f9dc393826a54789e40a098e4605a49c79aa199a801ad1b4b7a7ed1295949e74",'
        b'"initial_pool":{"start_id":"__start__","store":{"__start__":{"kind":"object",'
        b'"value":{"x":"Ada"}}},"system":{"now":{"kind":"datetime",'
        b'"value":"2026-09-20T01:23:31.293007+00:00"},"run_id":{"kind":"str",'
        b'"value":"historical-linear"},"today":{"kind":"date","value":"2026-09-20"}}},"learnable":false,'
        b'"memories":null,"memory_snapshots":null,"memory_stacks":null,"num_workers":0,'
        b'"start_wiring_override":{"x":"${input.x}"}},"run_id":"historical-linear","schema":9,"seq":0}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"__start__"},"run_id":"historical-linear",'
        b'"schema":9,"seq":1}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"__start__","evidence":null,'
        b'"forward_capture":null,"node_id":"__start__","value":{"kind":"object","value":{"x":"Ada"}}},'
        b'"run_id":"historical-linear","schema":9,"seq":2}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"n"},"run_id":"historical-linear","schema":9,'
        b'"seq":3}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"n","evidence":null,"forward_capture":null,'
        b'"node_id":"n","value":{"kind":"str","value":"Ada"}},"run_id":"historical-linear","schema":9,'
        b'"seq":4}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"__end__"},"run_id":"historical-linear","schema":9,'
        b'"seq":5}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"__end__","evidence":null,'
        b'"forward_capture":null,"node_id":"__end__","value":{"kind":"str","value":"Ada"}},'
        b'"run_id":"historical-linear","schema":9,"seq":6}'
    ),
)


_RUN_ID = "recovery-run"

_LINEAR = """
id: recovery-linear
name: recovery_linear
input:
  x: str
nodes:
  n:
    kind: code
    input:
      x: ${input.x}
    output: str
    code: tests.engine._compose_codefns:echo_x
output: ${n.output}
"""

_FANOUT = """
id: recovery-fanout
name: recovery_fanout
nodes:
  first:
    kind: code
    input: {topic: first}
    output: str
    code: tests.engine._compose_codefns:cautious
  second:
    kind: code
    input: {topic: second}
    output: str
    code: tests.engine._compose_codefns:cautious
output:
  first: ${first.output}
  second: ${second.output}
"""

_CALL_AGENT = """
id: recovery-call-agent
name: recovery_call_agent
defs:
  child:
    nodes:
      leaf:
        kind: agent
        mode: plain
        prompt: replay me
    output: ${leaf.output}
nodes:
  gate:
    kind: call
    call: child
output: ${gate.output}
"""


class _ReadOnlyLog:
    """Stable borrowed prefix whose write and close boundaries must stay unused."""

    def __init__(
        self,
        envelopes: tuple[RecordEnvelope, ...],
        *,
        run_id: str | None = None,
        next_sequence: int | None = None,
    ) -> None:
        self._envelopes = envelopes
        self._run_id = (
            run_id
            if run_id is not None
            else (envelopes[0].run_id if envelopes else _RUN_ID)
        )
        self._next_sequence = (
            next_sequence if next_sequence is not None else len(envelopes)
        )
        self.append_calls = 0
        self.close_calls = 0

    @property
    def run_id(self) -> str:
        """Return the identity claimed by this source prefix."""
        return self._run_id

    @property
    def path(self):
        """Return no path because this test prefix is process-local."""
        return None

    @property
    def next_sequence(self) -> int:
        """Return the source prefix length claimed by the log."""
        return self._next_sequence

    def __iter__(self) -> Iterator[RecordEnvelope]:
        """Iterate the retained historical envelopes without copying them."""
        return iter(self._envelopes)

    def append(self, record):
        """Fail if recovery tries to append to historical authority."""
        self.append_calls += 1
        raise AssertionError("historical source log must remain read-only")

    def close(self) -> None:
        """Record an ownership violation if recovery closes the borrowed log."""
        self.close_calls += 1


def _started_state(
    loaded,
    *,
    start_input: dict[str, Any],
    start_wiring: dict[str, Any],
    num_workers: int = 0,
    effective: dict[str, dict[str, Any]] | None = None,
    private_flow=None,
) -> tuple[RunState, MemoryRunLog]:
    """Commit a canonical RunStarted record for one private graph."""
    flow = copy.deepcopy(loaded.compiled) if private_flow is None else private_flow
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, start_input)
    pool.add_system("run_id", _RUN_ID)
    pool.add_system("today", "2026-08-14")
    pool.add_system("now", "2026-08-14T12:00:00+00:00")
    state = RunState(flow)
    log = MemoryRunLog(_RUN_ID)
    commit_prepared(
        log,
        state,
        prepare_run_started(
            state,
            RunStartData(
                run_id=_RUN_ID,
                flow_fingerprint=flow_fingerprint(loaded.compiled),
                initial_pool=pool,
                start_wiring_override=start_wiring,
                effective_llm_by_static_path=effective or {},
                num_workers=num_workers,
            ),
        ),
    )
    return state, log


def _state_snapshot(state: RunState) -> dict[str, Any]:
    """Return every replay-relevant coordinate in value-comparable form."""
    return {
        "flow": flow_manifest(state.flow),
        "pool": encode_pool(state.pool),
        "node_state": dict(state.node_state),
        "edge_state": dict(state.edge_state),
        "ready": tuple(state.ready),
        "in_flight": tuple(state.in_flight),
        "result": state.result,
        "terminal_unreachable_failure": state.terminal_unreachable_failure,
        "depth": dict(state.depth),
        "wiring_overrides": copy.deepcopy(state.wiring_overrides),
        "run_id": state.run_id,
        "flow_fingerprint": state.flow_fingerprint,
        "next_sequence": state.next_sequence,
        "num_workers": state.num_workers,
        "input": copy.deepcopy(state.input),
        "effective": copy.deepcopy(state.effective_llm_by_static_path),
        "initial_static_ids": state.initial_static_ids,
    }


def _linear_history():
    """Return a terminal log and the live state after every authoritative prefix."""
    loaded = load_flow(_LINEAR)
    state, log = _started_state(
        loaded,
        start_input={"x": "Ada"},
        start_wiring={"x": "${input.x}"},
        num_workers=2,
    )
    snapshots = [_state_snapshot(state)]

    transitions = (
        lambda: prepare_node_started(state, START_ID),
        lambda: prepare_node_succeeded(state, START_ID, {"x": "Ada"}),
        lambda: prepare_node_started(state, "n"),
        lambda: prepare_node_succeeded(state, "n", "Ada"),
        lambda: prepare_node_started(state, END_ID),
        lambda: prepare_node_succeeded(state, END_ID, "Ada"),
    )
    for transition in transitions:
        commit_prepared(log, state, transition())
        snapshots.append(_state_snapshot(state))
    assert state.result is not None
    return loaded, state, log, snapshots


def _agent_growth_history():
    """Return a log whose CALL expansion contains a configured AGENT clone."""
    loaded = load_flow(_CALL_AGENT)
    private_flow = copy.deepcopy(loaded.compiled)
    effective = resolve_llm_cascade(
        private_flow,
        {"temperature": 0.25},
        defaults={"provider": "openai", "model": "recorded-model"},
    )
    state, log = _started_state(
        loaded,
        start_input={},
        start_wiring={},
        effective=effective,
        private_flow=private_flow,
    )
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(state, START_ID, {}))
    commit_prepared(log, state, prepare_node_started(state, "gate"))
    grow = state.flow.nodes["gate"].run({})
    assert type(grow) is Grow
    commit_prepared(log, state, prepare_expanded(state, "gate", grow, build_grow))
    return loaded, state, log, effective


def test_replay_every_prefix_matches_live_state_and_uses_commit_path(monkeypatch):
    loaded, _, log, snapshots = _linear_history()
    envelopes = tuple(log)
    original_manifest = flow_manifest(loaded.compiled)
    original_commit = state_module.commit_prepared
    committed_records: list[Any] = []
    historical_observers: list[tuple[Any, ...]] = []

    def observed_commit(cursor, state, prepared):
        committed_records.append(prepared.record)
        observers = original_commit(cursor, state, prepared)
        historical_observers.append(observers)
        return observers

    monkeypatch.setattr(state_module, "commit_prepared", observed_commit)

    for length, expected in enumerate(snapshots, start=1):
        source = _ReadOnlyLog(envelopes[:length])
        recovered = run_module._replay_run_log(loaded, source)

        assert _state_snapshot(recovered) == expected
        assert source.append_calls == 0
        assert source.close_calls == 0
        assert source.next_sequence == length

    assert len(committed_records) == sum(range(1, len(envelopes) + 1))
    assert historical_observers
    assert flow_manifest(loaded.compiled) == original_manifest


def test_replay_preserves_ordered_unfinished_starts_for_later_resume():
    loaded = load_flow(_FANOUT)
    state, log = _started_state(
        loaded,
        start_input={},
        start_wiring={},
        num_workers=4,
    )
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(state, START_ID, {}))
    assert tuple(state.ready) == ("first", "second")
    commit_prepared(log, state, prepare_node_started(state, "first"))
    commit_prepared(log, state, prepare_node_started(state, "second"))
    assert tuple(state.in_flight) == ("first", "second")

    recovered = run_module._replay_run_log(
        loaded,
        _ReadOnlyLog(tuple(log)),
    )

    assert tuple(recovered.ready) == ()
    assert tuple(recovered.in_flight) == ("first", "second")
    assert recovered.num_workers == 4


def test_replay_applies_logged_llm_map_before_rebuilding_growth(monkeypatch):
    loaded, live, log, effective = _agent_growth_history()
    original_manifest = flow_manifest(loaded.compiled)
    source = _ReadOnlyLog(tuple(log))

    def unexpected_default():
        pytest.fail("recovery read an ambient LLM default")

    monkeypatch.setattr(run_module, "connection_snapshot", unexpected_default)

    recovered = run_module._replay_run_log(loaded, source)

    assert _state_snapshot(recovered) == _state_snapshot(live)
    assert recovered.effective_llm_by_static_path == effective
    assert isinstance(recovered.flow.nodes["gate/leaf"], AgentNode)
    from agent_composer.llm_clients.connection import legacy_connection_config
    assert recovered.flow.nodes["gate/leaf"].llm_config == legacy_connection_config(effective["gate/leaf"])
    assert source.append_calls == 0
    assert source.close_calls == 0
    assert flow_manifest(loaded.compiled) == original_manifest


def test_historical_cursor_matches_each_envelopes_schema():
    from agent_composer.runtime.log import decode_record, encode_record

    recorded_prefix = _RECORDED_LINEAR[:4]
    with MemoryRunLog._from_records(tuple(decode_record(wire) for wire in recorded_prefix)) as log:
        appended_suffix = tuple(
            encode_record(log.append(decode_record(wire).record))
            for wire in _RECORDED_LINEAR[4:]
        )
    assert all(decode_record(wire).schema == SCHEMA_VERSION for wire in appended_suffix)
    wires = recorded_prefix + appended_suffix
    cursor = state_module._HistoricalLogCursor(wires)
    for wire in wires:
        expected = decode_record(wire)
        before = cursor.next_sequence
        with pytest.raises(ValueError, match="historical"):
            cursor.append(NodeStartedRecord("wrong"))
        assert cursor.next_sequence == before
        accepted = cursor.append(expected.record)
        assert accepted.schema == expected.schema
        assert encode_record(accepted) == wire
    assert cursor.next_sequence == len(wires)


@pytest.mark.parametrize("prefix_length", [5, 6])
def test_recorded_public_resume_keeps_completed_code_and_wire(tmp_path, monkeypatch, prefix_length):
    from agent_composer.nodes.code import CodeNode
    from agent_composer.runtime.log import JsonlRunLog, decode_record, encode_record

    flow = load_flow(_LINEAR)
    prefix = b"\n".join(_RECORDED_LINEAR[:prefix_length]) + b"\n"
    path = tmp_path / "historical.jsonl"
    path.write_bytes(prefix)
    monkeypatch.setattr(CodeNode, "forward", lambda *args, **kwargs: pytest.fail("accepted CODE reran"))
    monkeypatch.setattr(run_module, "connection_snapshot", lambda: pytest.fail("read current defaults"))

    result = run_module.resume_flow(flow, log=path)

    assert result.status == "succeeded" and result.output == "Ada"
    assert result.run_id == "historical-linear" and result.input == {"x": "Ada"}
    assert path.read_bytes().startswith(prefix)
    with JsonlRunLog.resume(path) as log:
        envelopes = tuple(log)
        replayed = run_module._replay_run_log(flow, log)
    assert tuple(encode_record(item) for item in envelopes[:prefix_length]) == _RECORDED_LINEAR[:prefix_length]
    assert all(item.schema == SCHEMA_VERSION for item in envelopes[prefix_length:])
    start = decode_record(_RECORDED_LINEAR[0]).record
    assert replayed.flow_fingerprint == start.flow_fingerprint == flow_fingerprint(flow.compiled)
    assert replayed.effective_llm_by_static_path == start.effective_llm_by_static_path
    assert replayed.num_workers == start.num_workers == 0
    assert replayed.learnable is False and replayed.trace is None
    complete = path.read_bytes()
    inspected = run_module.resume_flow(flow, log=path)
    assert inspected.output == result.output and inspected.events == []
    assert path.read_bytes() == complete


def test_replay_rejects_changed_but_valid_growth_digest(monkeypatch):
    loaded, _, log, _ = _agent_growth_history()
    original_builder = build_grow

    def changed_builder(spawner, seed):
        fragment = original_builder(spawner, seed)
        fragment.nodes[fragment.start_id].env["replay-drift"] = True
        return fragment

    monkeypatch.setattr(run_module, "build_grow", changed_builder)

    with pytest.raises(ValueError, match="topology digest mismatch"):
        run_module._replay_run_log(loaded, _ReadOnlyLog(tuple(log)))


@pytest.mark.parametrize(
    "source",
    [
        _ReadOnlyLog(()),
        _ReadOnlyLog(
            (
                RecordEnvelope(
                    schema=SCHEMA_VERSION,
                    run_id=_RUN_ID,
                    seq=0,
                    record=NodeStartedRecord(node_id=START_ID),
                ),
            )
        ),
        _ReadOnlyLog(
            (
                RecordEnvelope(
                    schema=SCHEMA_VERSION,
                    run_id=_RUN_ID,
                    seq=1,
                    record=NodeStartedRecord(node_id=START_ID),
                ),
            )
        ),
    ],
    ids=["empty", "first-not-run-started", "sequence-not-zero"],
)
def test_replay_rejects_invalid_prefix_roots(source):
    with pytest.raises(ValueError):
        run_module._replay_run_log(load_flow(_LINEAR), source)


def test_replay_rejects_log_identity_and_length_disagreement():
    loaded, _, log, _ = _linear_history()
    envelopes = tuple(log)

    with pytest.raises(ValueError, match="run_id"):
        run_module._replay_run_log(
            loaded,
            _ReadOnlyLog(envelopes, run_id="different-run"),
        )
    with pytest.raises(ValueError, match="sequence"):
        run_module._replay_run_log(
            loaded,
            _ReadOnlyLog(envelopes, next_sequence=len(envelopes) + 1),
        )


def test_fingerprint_mismatch_fails_before_copy_or_llm_apply(monkeypatch):
    _, _, log, _ = _linear_history()
    source = _ReadOnlyLog(tuple(log))
    changed = load_flow(_LINEAR)
    changed.compiled.nodes["n"].env["semantic-change"] = True

    def unexpected(*args, **kwargs):
        pytest.fail("fingerprint mismatch performed recovery mutation")

    class _CopyBomb:
        """Fail if recovery copies the flow before validating its fingerprint."""

        deepcopy = staticmethod(unexpected)

    monkeypatch.setattr(run_module, "copy", _CopyBomb)
    monkeypatch.setattr(
        run_module,
        "apply_effective_llm_map",
        unexpected,
        raising=False,
    )

    with pytest.raises(ValueError, match="flow fingerprint mismatch"):
        run_module._replay_run_log(changed, source)


def test_replay_requires_pristine_state():
    loaded, _, log, _ = _linear_history()
    state = RunState(copy.deepcopy(loaded.compiled))
    state.ready.append(START_ID)

    with pytest.raises(ValueError, match="pristine"):
        state_module._replay_records(
            state,
            tuple(log),
            grow_builder=build_grow,
        )


def test_replay_rejects_records_after_terminal_without_touching_source():
    loaded, _, log, _ = _linear_history()
    trailing = MemoryRunLog(_RUN_ID)
    for envelope in log:
        trailing.append(envelope.record)
    trailing.append(NodeStartedRecord(node_id=START_ID))
    source = _ReadOnlyLog(tuple(trailing))

    with pytest.raises(ValueError, match="terminal"):
        run_module._replay_run_log(loaded, source)

    assert source.append_calls == 0
    assert source.close_calls == 0
    assert source.next_sequence == len(tuple(trailing))


def test_run_started_replay_rejects_pool_and_envelope_identity_drift():
    loaded, _, log, _ = _linear_history()
    first = tuple(log)[0]
    assert isinstance(first.record, RunStartedRecord)
    first.record.initial_pool["system"]["run_id"]["value"] = "other-run"
    source = _ReadOnlyLog((first,))

    with pytest.raises(ValueError, match="system.run_id"):
        run_module._replay_run_log(loaded, source)
