"""Recovery equivalence across nested growth and unfinished live effects.

The append-only log is the sole run authority.  These tests capture the live
``RunState`` after every committed record, replay every resulting prefix through
the public-boundary recovery seam, and compare the complete reducer state.  The
effect cases stop the dispatcher immediately after durable ``NodeStarted`` so no
model, human interaction, or wall-clock wait is invoked.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import pytest

import agent_composer.nodes.wait.node as wait_module
import agent_composer.run as run_module
import agent_composer.runtime.engine as engine_module
from agent_composer.compile.expand import build_grow
from agent_composer.compile.llm_cascade import resolve_llm_cascade
from agent_composer.compile.manifest import flow_fingerprint, flow_manifest
from agent_composer.compile.model import START_ID
from agent_composer.compose import load_flow
from agent_composer.events import NodeStarted, NodeSucceeded, RunSucceeded
from agent_composer.nodes.base import Output
from agent_composer.runtime.engine import _RecordedDispatcher
from agent_composer.runtime.eval_node import _eval_node_raw
from agent_composer.runtime.log import (
    MemoryRunLog,
    NodeStartedRecord,
    RunResumedRecord,
    encode_pool,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_run_resumed,
    prepare_run_started,
)
from agent_composer.typesys import VariablePool


_RUN_ID = "recovery-matrix"
_SYSTEM_TODAY = "2026-08-14"
_SYSTEM_NOW = "2026-08-14T12:00:00+00:00"
_WAIT_DEADLINE = "2026-08-14T12:00:10+00:00"

_NESTED_GROWTH = """
id: recovery-matrix-growth
name: recovery_matrix_growth
defs:
  echo_one:
    input:
      value: int
    nodes:
      echo:
        kind: code
        input:
          x: ${input.value}
        output: int
        code: tests.engine._compose_codefns:echo_x
    output: ${echo.output}
  bump_once:
    input:
      n: int
      exited: bool
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
        code: tests.engine._compose_codefns:loop_bump
    output: ${bump.output}
  composite:
    nodes:
      mapped:
        kind: map
        call: echo_one
        over: ${[1, 2]}
        input:
          value: ${item}
      repeated:
        kind: loop
        call: bump_once
        input:
          n: 0
          exited: false
        times: 2
    output:
      mapped: ${mapped.output}
      repeated: ${repeated.output}
nodes:
  outer:
    kind: call
    call: composite
output: ${outer.output}
"""

_EFFECT_FLOWS = {
    "agent": """
id: recovery-agent
name: recovery_agent
nodes:
  effect:
    kind: agent
    mode: plain
    prompt: this model must never be invoked
    llm_config:
      provider: openai
      model: offline-test
output: ${effect.output}
""",
    "human_input": """
id: recovery-human
name: recovery_human
nodes:
  effect:
    kind: human_input
    prompt: this host interaction must never be invoked
    output: str
output: ${effect.output}
""",
    "wait": f"""
id: recovery-wait
name: recovery_wait
nodes:
  effect:
    kind: wait
    until: {_WAIT_DEADLINE!r}
output: ${{effect.output}}
""",
}

_ORDERED_EFFECTS = f"""
id: recovery-ordered-effects
name: recovery_ordered_effects
nodes:
  release:
    kind: code
    input: {{x: ready}}
    output: str
    code: tests.engine._compose_codefns:echo_x
  wait:
    kind: wait
    depends_on: [release]
    until: {_WAIT_DEADLINE!r}
  agent:
    kind: agent
    depends_on: [release]
    mode: plain
    prompt: never invoked
    llm_config: {{provider: openai, model: offline-test}}
  human:
    kind: human_input
    depends_on: [release]
    prompt: never invoked
    output: str
  later:
    kind: code
    depends_on: [release]
    input: {{x: still-ready}}
    output: str
    code: tests.engine._compose_codefns:echo_x
output:
  agent: ${{agent.output}}
  human: ${{human.output}}
  wait: ${{wait.output}}
  later: ${{later.output}}
"""


def _state_snapshot(state: RunState) -> dict[str, Any]:
    """Return every topology and reducer coordinate as detached values."""
    edges = tuple(
        (
            edge.id,
            edge.from_,
            edge.to,
            edge.source_handle,
            edge.input_group,
            edge.optional,
            edge.ordering,
        )
        for edge in state.flow.edges
    )
    return {
        "topology": flow_manifest(state.flow),
        "node_order": tuple(state.flow.nodes),
        "edges": edges,
        "wiring": copy.deepcopy(state.flow.wiring),
        "incoming": {
            node_id: tuple(edge.id for edge in state.flow.incoming(node_id))
            for node_id in state.flow.nodes
        },
        "outgoing": {
            node_id: tuple(edge.id for edge in state.flow.outgoing(node_id))
            for node_id in state.flow.nodes
        },
        "pool": encode_pool(state.pool),
        "node_state": dict(state.node_state),
        "edge_state": dict(state.edge_state),
        "ready": tuple(state.ready),
        "in_flight": tuple(state.in_flight),
        "depth": dict(state.depth),
        "wiring_overrides": copy.deepcopy(state.wiring_overrides),
        "effective_llm_by_static_path": copy.deepcopy(
            state.effective_llm_by_static_path
        ),
        "result": copy.deepcopy(state.result),
        "next_sequence": state.next_sequence,
        "run_id": state.run_id,
        "flow_fingerprint": state.flow_fingerprint,
        "num_workers": state.num_workers,
        "input": copy.deepcopy(state.input),
        "initial_static_ids": state.initial_static_ids,
    }


def _new_started_run(loaded, *, num_workers: int) -> tuple[RunState, MemoryRunLog]:
    """Create one private explicitly configured run through its first record."""
    private_flow = copy.deepcopy(loaded.compiled)
    effective = resolve_llm_cascade(
        private_flow,
        {},
        defaults={"provider": "openai", "model": "offline-test"},
    )
    pool = VariablePool(start_id=private_flow.start_id)
    pool.set(private_flow.start_id, {})
    pool.add_system("run_id", _RUN_ID)
    pool.add_system("today", _SYSTEM_TODAY)
    pool.add_system("now", _SYSTEM_NOW)

    state = RunState(private_flow)
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
                start_wiring_override={},
                effective_llm_by_static_path=effective,
                num_workers=num_workers,
            ),
        ),
    )
    return state, log


def _prefix_log(log: MemoryRunLog, length: int) -> MemoryRunLog:
    """Copy the first ``length`` authoritative records into an open memory log."""
    prefix = MemoryRunLog(log.run_id)
    for envelope in tuple(log)[:length]:
        prefix.append(envelope.record)
    return prefix


def _pause_after_effect_started(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> tuple[Any, RunState, MemoryRunLog]:
    """Stop a serial dispatcher at the target's post-durable start yield."""
    loaded = load_flow(source)
    state, log = _new_started_run(loaded, num_workers=0)
    dispatcher = _RecordedDispatcher(
        state,
        log,
        grow_builder=build_grow,
        num_workers=0,
    )
    drive = dispatcher.run()

    assert next(drive) == NodeStarted(START_ID)
    assert next(drive) == NodeSucceeded(START_ID, output={})

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("effect work ran after only its durable NodeStarted was requested")

    monkeypatch.setattr(dispatcher, "_evaluate", unexpected_evaluation)
    assert next(drive) == NodeStarted("effect")
    assert tuple(log)[-1].record == NodeStartedRecord(node_id="effect")
    drive.close()
    return loaded, state, log


@pytest.mark.parametrize("num_workers", [0, 2], ids=["serial", "pooled"])
def test_every_nested_growth_prefix_replays_to_the_live_state(
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
) -> None:
    """CALL containing MAP and LOOP replays identically after every commit."""
    loaded = load_flow(_NESTED_GROWTH)
    state, log = _new_started_run(loaded, num_workers=num_workers)
    snapshots = [_state_snapshot(state)]
    original_commit = engine_module.commit_prepared

    def capture_after_commit(cursor, live_state, prepared):
        observers = original_commit(cursor, live_state, prepared)
        assert live_state is state
        snapshots.append(_state_snapshot(live_state))
        return observers

    monkeypatch.setattr(engine_module, "commit_prepared", capture_after_commit)
    observers = list(
        _RecordedDispatcher(
            state,
            log,
            grow_builder=build_grow,
            num_workers=num_workers,
        ).run()
    )

    assert isinstance(state.result, RunSucceeded)
    assert state.result.output == {
        "mapped": [1, 2],
        "repeated": {"n": 2, "exited": False},
    }
    assert observers[-1] == state.result
    records = tuple(log)
    assert len(snapshots) == len(records)

    for length, expected in enumerate(snapshots, start=1):
        recovered = run_module._replay_run_log(loaded, _prefix_log(log, length))
        assert _state_snapshot(recovered) == expected, (
            f"replayed state diverged after authoritative prefix {length}/{len(records)}"
        )


@pytest.mark.parametrize(
    "source",
    tuple(_EFFECT_FLOWS.values()),
    ids=tuple(_EFFECT_FLOWS),
)
def test_effect_start_prefix_replays_without_running_effect_work(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """AGENT, HUMAN_INPUT, and WAIT starts remain recoverable before execution."""
    loaded, live, log = _pause_after_effect_started(monkeypatch, source)
    recovered = run_module._replay_run_log(loaded, log)

    assert _state_snapshot(recovered) == _state_snapshot(live)
    assert tuple(recovered.in_flight) == ("effect",)
    assert tuple(recovered.ready) == ()

    prepared = prepare_run_resumed(recovered, num_workers=0)
    assert prepared.record == RunResumedRecord(
        retry_node_ids=("effect",),
        num_workers=0,
    )
    commit_prepared(log, recovered, prepared)
    assert tuple(recovered.in_flight) == ()
    assert tuple(recovered.ready) == ("effect",)


def test_resume_prepends_ordered_effect_retries_before_existing_ready_work() -> None:
    """Recovery preserves nonlexicographic starts ahead of untouched readiness."""
    loaded = load_flow(_ORDERED_EFFECTS)
    state, log = _new_started_run(loaded, num_workers=2)
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(state, START_ID, {}))
    commit_prepared(log, state, prepare_node_started(state, "release"))
    commit_prepared(log, state, prepare_node_succeeded(state, "release", "ready"))
    assert tuple(state.ready) == ("wait", "agent", "human", "later")

    retry_order = ("wait", "agent", "human")
    assert retry_order != tuple(sorted(retry_order))
    for node_id in retry_order:
        commit_prepared(log, state, prepare_node_started(state, node_id))
    started_order = tuple(
        envelope.record.node_id
        for envelope in log
        if isinstance(envelope.record, NodeStartedRecord)
    )
    assert started_order == (START_ID, "release", *retry_order)
    assert tuple(state.in_flight) == retry_order
    assert tuple(state.ready) == ("later",)

    recovered = run_module._replay_run_log(loaded, log)
    assert tuple(recovered.in_flight) == retry_order
    assert tuple(recovered.ready) == ("later",)

    prepared = prepare_run_resumed(recovered, num_workers=2)
    assert prepared.record == RunResumedRecord(
        retry_node_ids=retry_order,
        num_workers=2,
    )
    commit_prepared(log, recovered, prepared)
    assert tuple(log)[-1].record == prepared.record
    assert tuple(recovered.in_flight) == ()
    assert tuple(recovered.ready) == (*retry_order, "later")


class _ImmediateTimeoutEvent:
    """Record requested timeout durations while returning without sleeping."""

    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def is_set(self) -> bool:
        """Report an active run before evaluation begins."""
        return False

    def wait(self, timeout: float | None = None) -> bool:
        """Capture one absolute-deadline remainder and simulate its timeout."""
        self.timeouts.append(timeout)
        return False


def test_recovered_wait_recomputes_from_the_authored_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAIT recovery retains the deadline and derives a new remaining duration."""
    loaded, _, log = _pause_after_effect_started(
        monkeypatch,
        _EFFECT_FLOWS["wait"],
    )
    recovered = run_module._replay_run_log(loaded, log)
    assert recovered.flow.wiring["effect"]["until"] == _WAIT_DEADLINE

    prepared = prepare_run_resumed(recovered, num_workers=0)
    commit_prepared(log, recovered, prepared)
    node = recovered.flow.nodes["effect"]

    observed_timeouts = []
    for second, expected in ((4, 6.0), (7, 3.0)):
        event = _ImmediateTimeoutEvent()
        monkeypatch.setattr(
            wait_module,
            "_utc_now",
            lambda second=second: datetime(
                2026,
                8,
                14,
                12,
                0,
                second,
                tzinfo=timezone.utc,
            ),
        )
        outcome = _eval_node_raw(
            node,
            recovered.flow,
            recovered.pool,
            stop_event=event,
        )
        assert outcome == Output(None)
        assert event.timeouts == [expected]
        observed_timeouts.append(event.timeouts[0])

    assert observed_timeouts == [6.0, 3.0]
