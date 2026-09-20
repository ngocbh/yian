"""Pure accepted conversation history and its append-before-apply reducer."""

from copy import deepcopy
from dataclasses import replace

import pytest

import agent_composer.memory.contracts as c
from agent_composer.compile.model import CompiledFlow, Edge, END_ID, START_ID
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.memory.conversation import advance_conversation
from agent_composer.memory.session import SessionMemory
from agent_composer.diagnostics import Failure
from agent_composer.expr.lineage import ValueRoute
from agent_composer.runtime.contracts import ForwardCapture
from agent_composer.runtime.learning import project_outcome_episodes
from agent_composer.runtime import state as reducer
from agent_composer.runtime.conversation import (
    RuntimeConversationState,
    prepare_conversation_checkpoint,
)
from agent_composer.runtime.log import (
    ConversationAdmittedRecord,
    ConversationEffectRequestedRecord,
    ConversationEffectObservedRecord,
    ConversationClosedRecord,
    MemoryRunLog,
    RunResumedRecord,
    RecordEnvelope,
    BackwardStartedRecord,
    BackwardFinishedRecord,
)
from agent_composer.typesys import VariablePool
from tests.engine._fakes import FuncNode
from tests.engine._session_fakes import REF, admission, item, snapshot


def _flow(*, sibling=False):
    node = FuncNode("agent", lambda _: pytest.fail("replay invoked a node"))
    node.memory_scope = c.MemoryScope("root", ("root", "agent"), (REF,), (), None)
    flow = CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            "agent": node,
            END_ID: EndNode.record(END_ID, output_names=[]),
        },
        edges=[
            Edge(id="start-agent", from_=START_ID, to="agent"),
            Edge(id="agent-end", from_="agent", to=END_ID),
        ],
    )
    if sibling:
        flow.add_subgraph(
            {"sibling": FuncNode("sibling", lambda _: pytest.fail("node called"))},
            [Edge(id="start-sibling", from_=START_ID, to="sibling")],
            {"sibling": {}},
        )
    return flow


def _capture(state):
    return ForwardCapture({}, ValueRoute.record(())) if state.learnable else None


def _started(*, learnable=False, sibling=False, enabled=True):
    flow = _flow(sibling=sibling)
    state = reducer.RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {})
    pool.add_system("run_id", "run")
    log = MemoryRunLog("run")
    reducer.commit_prepared(
        log,
        state,
        reducer.prepare_run_started(
            state,
            reducer.RunStartData(
                "run",
                "a" * 64,
                pool,
                {},
                {},
                0,
                (c.MemoryFrameSpec(REF, "session", {}),) if enabled else None,
                (c.MemorySnapshotEnvelope("session", {}),) if enabled else None,
                learnable,
            ),
        ),
    )
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, START_ID))
    reducer.commit_prepared(
        log,
        state,
        reducer.prepare_node_succeeded(
            state, START_ID, {}, forward_capture=_capture(state)
        ),
    )
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, "agent"))
    return state, log


def _snapshot(state):
    return (
        deepcopy(state.pool),
        dict(state.node_state),
        dict(state.edge_state),
        tuple(state.ready),
        dict(state.in_flight),
        dict(state.accepted_starts),
        state.result,
        state.next_sequence,
        state.conversation_state,
        state.learning_state,
        state.backward_state,
        dict(state.depth),
        deepcopy(state.wiring_overrides),
        deepcopy(state.input),
        deepcopy(state.effective_llm_by_static_path),
        state.memory_layout,
        state.memory_snapshots,
        state.learnable,
        state.terminal_unreachable_failure,
        state.num_workers,
        state.run_id,
        state.flow_fingerprint,
        tuple((name, id(node)) for name, node in state.flow.nodes.items()),
    )


def _trace_snapshot(trace):
    if trace is None:
        return None
    return {
        name: dict(value) if name == "_backward_handlers" else deepcopy(value)
        for name, value in trace.__dict__.items()
    }


def _admit(state, *, baseline=None):
    return admission(
        c.ConversationInvocation("run", state.accepted_starts["agent"]),
        baseline=baseline,
    )


_RECORDS = {
    c.ConversationAdmitted: ConversationAdmittedRecord,
    c.ConversationEffectRequested: ConversationEffectRequestedRecord,
    c.ConversationEffectObserved: ConversationEffectObservedRecord,
    c.ConversationClosed: ConversationClosedRecord,
}


def _accept(state, log, checkpoint):
    prepared = reducer.prepare_record(state, _RECORDS[type(checkpoint)](checkpoint))
    assert reducer.commit_prepared(log, state, prepared) == ()
    return state.conversation_state.views[-1]


def _checkpoints(first):
    requested = c.ConversationEffectRequested(
        replace(first.stamp, expected_revision=first.baseline.revision + 1, ordinal=1),
        "model",
        {"input": "hello"},
        (item("user", "hello", "user"),),
    )
    observed = c.ConversationEffectObserved(
        replace(
            requested.stamp,
            expected_revision=requested.stamp.expected_revision + 1,
            ordinal=2,
        ),
        1,
        "completed",
        (item("reply", "world"),),
        None,
        None,
    )
    closed = c.ConversationClosed(
        replace(
            observed.stamp,
            expected_revision=observed.stamp.expected_revision + 1,
            ordinal=3,
        ),
        "completed",
    )
    return first, requested, observed, closed


def _replay(state, log, *, sibling=False):
    replayed = reducer.RunState(_flow(sibling=sibling))
    reducer._replay_records(
        replayed, tuple(log), grow_builder=lambda *_: pytest.fail("builder called")
    )
    # The final snapshot entry holds live node identities, not replay semantics.
    assert _snapshot(replayed)[:-1] == _snapshot(state)[:-1]
    assert replayed.initial_static_ids == state.initial_static_ids
    assert replayed.flow.start_id == state.flow.start_id
    assert replayed.flow.end_id == state.flow.end_id
    assert replayed.flow.edges == state.flow.edges
    assert replayed.flow.wiring == state.flow.wiring
    assert replayed.flow._incoming == state.flow._incoming
    assert replayed.flow._outgoing == state.flow._outgoing
    assert {
        name: (type(node), node.memory_scope, node.execution_scope, node.commit_as)
        for name, node in replayed.flow.nodes.items()
    } == {
        name: (type(node), node.memory_scope, node.execution_scope, node.commit_as)
        for name, node in state.flow.nodes.items()
    }
    traces = (_trace_snapshot(replayed.trace), _trace_snapshot(state.trace))
    for trace in traces:
        if trace is not None:
            trace["_backward_handlers"] = {
                source: (type(handler.__self__), handler.__self__.id, handler.__func__)
                for source, handler in trace["_backward_handlers"].items()
            }
    assert traces[0] == traces[1]
    return replayed


@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("index", range(4))
def test_prepare_conversation_record_is_pure(learnable, index):
    state, log = _started(learnable=learnable)
    checkpoints = _checkpoints(_admit(state))
    for checkpoint in checkpoints[:index]:
        _accept(state, log, checkpoint)
    checkpoint = checkpoints[index]
    backend = SessionMemory()
    backend_before = snapshot(backend)
    before = _snapshot(state)
    trace_before = _trace_snapshot(state.trace)
    prepared = reducer.prepare_conversation_record(
        state, _RECORDS[type(checkpoint)](checkpoint)
    )
    assert _snapshot(state) == before
    assert _trace_snapshot(state.trace) == trace_before
    assert snapshot(backend) == backend_before
    assert prepared.observers == ()
    assert log.next_sequence == state.next_sequence
    assert prepared._plan.next_conversation_state != state.conversation_state


def test_pure_projection_uses_the_lower_conversation_fold():
    checkpoint = admission(c.ConversationInvocation("run", 3))
    before = RuntimeConversationState()
    after = prepare_conversation_checkpoint(before, checkpoint)
    assert before == RuntimeConversationState()
    assert after.segments[0].checkpoints == (checkpoint,)
    assert after.views[0].revision == 1
    with pytest.raises(ValueError, match="admitted"):
        prepare_conversation_checkpoint(after, checkpoint)


def test_projection_rejects_a_view_from_different_segment_authority():
    first = admission(c.ConversationInvocation("run", 3))
    other = replace(first, stamp=replace(first.stamp, policy_name="other"))
    wrong_view = advance_conversation(other.baseline, other)
    with pytest.raises(ValueError, match="view"):
        RuntimeConversationState(
            (c.ConversationAuthority(first.baseline, (first,)),), (wrong_view,)
        )


@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("index", range(4))
def test_append_failure_does_not_apply_conversation_delta(learnable, index):
    state, log = _started(learnable=learnable)
    checkpoints = _checkpoints(_admit(state))
    for checkpoint in checkpoints[:index]:
        _accept(state, log, checkpoint)
    checkpoint = checkpoints[index]
    backend = SessionMemory()
    backend_before = snapshot(backend)
    failure = OSError("disk full")

    class FailingLog:
        run_id = log.run_id
        next_sequence = log.next_sequence

        def append(self, record):
            raise failure

    prepared = reducer.prepare_conversation_record(
        state, _RECORDS[type(checkpoint)](checkpoint)
    )
    before = _snapshot(state)
    trace_before = _trace_snapshot(state.trace)
    with pytest.raises(OSError) as caught:
        reducer.commit_prepared(FailingLog(), state, prepared)
    assert caught.value is failure
    assert _snapshot(state) == before
    assert _trace_snapshot(state.trace) == trace_before
    assert snapshot(backend) == backend_before
    assert log.next_sequence == state.next_sequence


@pytest.mark.parametrize("learnable", [False, True])
def test_live_and_replay_conversation_state_match(learnable):
    state, log = _started(learnable=learnable)
    for checkpoint in _checkpoints(_admit(state)):
        _accept(state, log, checkpoint)
    _replay(state, log)


@pytest.mark.parametrize("learnable", [False, True])
def test_session_records_do_not_require_learning_or_bind_outputs(learnable):
    state, log = _started(learnable=learnable)
    before = _snapshot(state)
    trace_before = _trace_snapshot(state.trace)
    for checkpoint in _checkpoints(_admit(state)):
        record = _RECORDS[type(checkpoint)](checkpoint)
        assert project_outcome_episodes(state.run_id, record) == ()
        _accept(state, log, checkpoint)
    after = _snapshot(state)
    assert after[:7] == before[:7]
    assert after[9:] == before[9:]
    assert state.next_sequence == before[7] + 4
    if learnable:
        trace_after = _trace_snapshot(state.trace)
        assert trace_after.pop("next_sequence") == trace_before.pop("next_sequence") + 4
        assert trace_after == trace_before
    else:
        assert state.trace is None


@pytest.mark.parametrize(
    "corruption", ["run", "start", "policy", "frame", "consumer", "stack", "disabled"]
)
def test_checkpoint_rejects_foreign_or_out_of_order_invocation(corruption):
    state, log = _started(enabled=corruption != "disabled")
    first = _admit(state)
    if corruption == "run":
        first = replace(
            first,
            stamp=replace(first.stamp, invocation=c.ConversationInvocation("other", 3)),
        )
    elif corruption == "start":
        first = replace(
            first,
            stamp=replace(first.stamp, invocation=c.ConversationInvocation("run", 1)),
        )
    elif corruption == "policy":
        first = replace(first, stamp=replace(first.stamp, policy_name="not-enabled"))
    elif corruption == "frame":
        first = admission(
            first.stamp.invocation,
            ref=c.MemoryFrameRef("root", ("root", "agent"), c.MemoryOwner.AGENT),
        )
    elif corruption == "consumer":
        first = admission(first.stamp.invocation, consumer=("root", "other"))
    elif corruption == "stack":
        state.flow.nodes["agent"].memory_scope = c.MemoryScope(
            "root", ("root", "agent"), (), (), None
        )
    before = _snapshot(state)
    with pytest.raises((TypeError, ValueError)):
        reducer.prepare_record(state, ConversationAdmittedRecord(first))
    assert _snapshot(state) == before
    assert log.next_sequence == state.next_sequence


@pytest.mark.parametrize(
    "corruption",
    [
        "ordinal",
        "revision",
        "foreign-key",
        "policy",
        "duplicate",
        "second-admission",
        "second-key",
        "post-close",
    ],
)
def test_admitted_invocation_rejects_invalid_checkpoint_order(corruption):
    state, log = _started()
    first, request, observation, close = _checkpoints(_admit(state))
    _accept(state, log, first)
    bad = request
    if corruption == "ordinal":
        bad = c.ConversationEffectRequested(
            replace(request.stamp, ordinal=2), "model", {}, ()
        )
    elif corruption == "revision":
        bad = c.ConversationEffectRequested(
            replace(request.stamp, expected_revision=5), "model", {}, ()
        )
    elif corruption == "foreign-key":
        bad = c.ConversationEffectRequested(
            replace(request.stamp, key=c.ConversationKey(REF, ("root", "other"))),
            "model",
            {},
            (),
        )
    elif corruption == "policy":
        bad = c.ConversationEffectRequested(
            replace(request.stamp, policy_name="other"), "model", {}, ()
        )
    elif corruption == "duplicate":
        _accept(state, log, request)
        bad = c.ConversationEffectRequested(
            request.stamp, "model", {"different": True}, ()
        )
    elif corruption == "second-admission":
        bad = first
    elif corruption == "second-key":
        bad = admission(first.stamp.invocation, consumer=("root", "other"))
    elif corruption == "post-close":
        for checkpoint in (request, observation, close):
            _accept(state, log, checkpoint)
        bad = c.ConversationEffectRequested(
            replace(close.stamp, expected_revision=4, ordinal=4), "model", {}, ()
        )
    before = _snapshot(state)
    with pytest.raises((TypeError, ValueError, c.MemoryOperationError)):
        reducer.prepare_record(state, _RECORDS[type(bad)](bad))
    assert _snapshot(state) == before


def test_success_requires_closed_handle_and_durable_start_inventory():
    state, log = _started()
    first = _admit(state)
    _accept(state, log, first)
    with pytest.raises(ValueError, match="closed"):
        reducer.prepare_node_succeeded(state, "agent", "value")
    close = c.ConversationClosed(
        replace(first.stamp, expected_revision=1, ordinal=1), "completed"
    )
    _accept(state, log, close)
    reducer.commit_prepared(
        log, state, reducer.prepare_node_succeeded(state, "agent", "value")
    )
    assert "agent" not in state.accepted_starts
    assert len(state.conversation_state.segments) == 1
    _replay(state, log)


@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("tamper", ["trace", "successor", "record", "stale"])
def test_forged_conversation_plan_is_rejected_before_append(learnable, tamper):
    state, log = _started(learnable=learnable)
    prepared = reducer.prepare_record(state, ConversationAdmittedRecord(_admit(state)))
    if tamper == "trace":
        from agent_composer.runtime.trace import TraceDelta

        delta = (
            replace(prepared._plan.trace_delta, resumed=True)
            if learnable
            else TraceDelta(None, state.next_sequence)
        )
        prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=delta))
    elif tamper == "successor":
        prepared = replace(
            prepared,
            _plan=replace(
                prepared._plan, next_conversation_state=RuntimeConversationState()
            ),
        )
    elif tamper == "record":
        prepared = replace(
            prepared,
            record=ConversationAdmittedRecord(
                admission(c.ConversationInvocation("run", 3), policy="other")
            ),
        )
    else:
        state.conversation_state = RuntimeConversationState()
    before = _snapshot(state)
    with pytest.raises((TypeError, ValueError)):
        reducer.commit_prepared(log, state, prepared)
    assert _snapshot(state) == before
    assert log.next_sequence == state.next_sequence


@pytest.mark.parametrize("prefix", ["admitted", "requested", "observed", "closed"])
def test_failure_drain_accepts_only_existing_observations_and_close(prefix):
    state, log = _started(sibling=True)
    checkpoints = _checkpoints(_admit(state))
    count = ["admitted", "requested", "observed", "closed"].index(prefix) + 1
    for checkpoint in checkpoints[:count]:
        _accept(state, log, checkpoint)
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, "sibling"))
    reducer.commit_prepared(
        log,
        state,
        reducer.prepare_node_failed(
            state, Failure("node.execute", "failed sibling", "sibling")
        ),
    )
    terminal = state.result
    if prefix == "admitted":
        with pytest.raises(ValueError, match="failed run"):
            reducer.prepare_record(
                state, ConversationEffectRequestedRecord(checkpoints[1])
            )
        _accept(
            state,
            log,
            c.ConversationClosed(
                replace(checkpoints[0].stamp, expected_revision=1, ordinal=1),
                "interrupted",
            ),
        )
    elif prefix in ("requested", "observed"):
        for checkpoint in checkpoints[count:]:
            _accept(state, log, checkpoint)
    with pytest.raises(ValueError):
        reducer.prepare_record(state, ConversationAdmittedRecord(checkpoints[0]))
    assert state.result is terminal
    _replay(state, log, sibling=True)


def test_failed_invocation_with_an_outcome_cannot_publish_late_checkpoints():
    state, log = _started()
    first, request, observation, _ = _checkpoints(_admit(state))
    _accept(state, log, first)
    _accept(state, log, request)
    reducer.commit_prepared(
        log,
        state,
        reducer.prepare_node_failed(
            state, Failure("node.execute", "failed agent", "agent")
        ),
    )
    with pytest.raises(ValueError, match="unfinished accepted start"):
        reducer.prepare_record(state, ConversationEffectObservedRecord(observation))
    assert "agent" not in state.accepted_starts


@pytest.mark.parametrize("backward", ["started", "finished"])
def test_any_backward_attempt_closes_the_conversation_checkpoint_window(backward):
    state, log = _started(learnable=True, sibling=True)
    first, request, observation, _ = _checkpoints(_admit(state))
    _accept(state, log, first)
    _accept(state, log, request)
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, "sibling"))
    reducer.commit_prepared(
        log,
        state,
        reducer.prepare_node_failed(
            state,
            Failure("node.execute", "failed sibling", "sibling"),
            forward_capture=_capture(state),
        ),
    )
    reducer.commit_prepared(
        log, state, reducer.prepare_record(state, BackwardStartedRecord(1, None))
    )
    if backward == "finished":
        reducer.commit_prepared(
            log,
            state,
            reducer.prepare_record(
                state, BackwardFinishedRecord(1, state.backward_state.outcome)
            ),
        )
    with pytest.raises(ValueError, match="backward start"):
        reducer.prepare_record(state, ConversationEffectObservedRecord(observation))
    _replay(state, log, sibling=True)


def test_success_closes_the_checkpoint_window():
    state, log = _started()
    first = _admit(state)
    _accept(state, log, first)
    close = c.ConversationClosed(
        replace(first.stamp, expected_revision=1, ordinal=1), "completed"
    )
    _accept(state, log, close)
    reducer.commit_prepared(
        log, state, reducer.prepare_node_succeeded(state, "agent", "done")
    )
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, END_ID))
    reducer.commit_prepared(
        log, state, reducer.prepare_node_succeeded(state, END_ID, {})
    )
    with pytest.raises(ValueError, match="after success"):
        reducer.prepare_record(state, ConversationAdmittedRecord(first))


@pytest.mark.parametrize("count", [2, 3, 4])
def test_resume_rejects_external_effects_without_node_outcome(count):
    state, log = _started()
    for checkpoint in _checkpoints(_admit(state))[:count]:
        _accept(state, log, checkpoint)
    before = _snapshot(state)
    with pytest.raises(ValueError, match="accepted conversation effects"):
        reducer.prepare_run_resumed(state, num_workers=1)
    assert _snapshot(state) == before
    forged = RecordEnvelope(
        8, "run", state.next_sequence, RunResumedRecord(("agent",), 1)
    )
    with pytest.raises(ValueError, match="accepted conversation effects"):
        reducer._replay_records(
            reducer.RunState(_flow()),
            tuple(log) + (forged,),
            grow_builder=lambda *_: pytest.fail("builder called"),
        )


def test_resume_allows_memory_free_unfinished_invocations():
    state, log = _started(enabled=False)
    first_sequence = state.accepted_starts["agent"]
    reducer.commit_prepared(
        log, state, reducer.prepare_run_resumed(state, num_workers=2)
    )
    assert state.accepted_starts == {}
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, "agent"))
    assert state.accepted_starts["agent"] > first_sequence
    assert state.conversation_state == RuntimeConversationState()
    _replay(state, log)


def test_resume_ignores_prior_completed_baseline_effects():
    old = admission(c.ConversationInvocation("previous-run", 1))
    baseline = old.baseline
    for checkpoint in _checkpoints(old):
        baseline = advance_conversation(baseline, checkpoint)
    state, log = _started()
    first = _admit(state, baseline=baseline)
    _accept(state, log, first)
    original = state.conversation_state
    reducer.commit_prepared(
        log, state, reducer.prepare_run_resumed(state, num_workers=0)
    )
    assert first.stamp.invocation in state.conversation_state.retired_invocations
    assert state.conversation_state.views == original.views
    assert state.conversation_state.segments == original.segments
    _replay(state, log)


def test_two_effect_free_resume_cycles_retire_without_inventing_history():
    state, log = _started()
    originals = []
    for _ in range(2):
        baseline = state.conversation_state.views[-1] if originals else None
        first = _admit(state, baseline=baseline)
        _accept(state, log, first)
        originals.append((first, state.conversation_state.views[-1]))
        reducer.commit_prepared(
            log, state, reducer.prepare_run_resumed(state, num_workers=0)
        )
        reducer.commit_prepared(
            log, state, reducer.prepare_node_started(state, "agent")
        )
    final = _admit(state, baseline=state.conversation_state.views[-1])
    effects = []
    for checkpoint in _checkpoints(final):
        _accept(state, log, checkpoint)
        if type(checkpoint) is c.ConversationEffectRequested:
            effects.append("actual fake effect")
    assert effects == ["actual fake effect"]
    ids = [first.stamp.invocation for first, _ in originals] + [final.stamp.invocation]
    assert len({value.start_sequence for value in ids}) == 3
    assert state.conversation_state.retired_invocations == frozenset(ids[:2])
    for index, (first, view) in enumerate(originals):
        assert state.conversation_state.segments[index].checkpoints == (first,)
        assert state.conversation_state.views[index] == view
        assert not view.closed_invocations and not view.items and not view.effects
        late = c.ConversationClosed(
            replace(first.stamp, expected_revision=view.revision, ordinal=1),
            "interrupted",
        )
        with pytest.raises(ValueError, match="retired"):
            reducer.prepare_record(state, ConversationClosedRecord(late))
    reducer.commit_prepared(
        log, state, reducer.prepare_node_succeeded(state, "agent", "done")
    )
    _replay(state, log)


def test_old_run_started_has_no_synthesized_conversations():
    import agent_composer.run as run_module
    from agent_composer.compose import load_flow
    from agent_composer.runtime.log import decode_record, encode_record
    from tests.engine.test_run_recovery_prefixes import (
        _LINEAR,
        _ReadOnlyLog,
        _SCHEMA_SEVEN_LINEAR,
    )

    wires = _SCHEMA_SEVEN_LINEAR[:4]
    records = tuple(decode_record(wire) for wire in wires)
    replayed = run_module._replay_run_log(load_flow(_LINEAR), _ReadOnlyLog(records))
    assert replayed.conversation_state == RuntimeConversationState()
    reopened = MemoryRunLog._from_records(records)
    reducer.commit_prepared(
        reopened, replayed, reducer.prepare_run_resumed(replayed, num_workers=0)
    )
    assert [envelope.schema for envelope in reopened] == [7, 7, 7, 7, 8]
    assert tuple(encode_record(envelope) for envelope in tuple(reopened)[:4]) == wires
    again = run_module._replay_run_log(
        load_flow(_LINEAR), _ReadOnlyLog(tuple(reopened))
    )
    assert again.conversation_state == RuntimeConversationState()


def test_admitted_checkpoint_drain_does_not_lookup_a_pruned_node():
    state, log = _started()
    first, request, observation, close = _checkpoints(_admit(state))
    _accept(state, log, first)
    _accept(state, log, request)
    # Admission's scope is already durable; later truth is invocation-owned.
    state.flow.nodes.pop("agent")
    _accept(state, log, observation)
    _accept(state, log, close)
    assert (
        state.conversation_state.views[-1].closed_invocations[0].status == "completed"
    )


def test_forged_resume_projection_cannot_omit_retirement():
    state, log = _started()
    first = _admit(state)
    _accept(state, log, first)
    prepared = reducer.prepare_run_resumed(state, num_workers=0)
    forged = replace(prepared._plan.delta.next_conversation_state)
    object.__setattr__(forged, "retired_invocations", frozenset())
    prepared = replace(
        prepared,
        _plan=replace(
            prepared._plan,
            delta=replace(prepared._plan.delta, next_conversation_state=forged),
        ),
    )
    before = _snapshot(state)
    with pytest.raises(ValueError):
        reducer.commit_prepared(log, state, prepared)
    assert _snapshot(state) == before


def test_forged_checkpoint_plan_rejects_equality_collapsing_payload_change():
    state, log = _started()
    first = _admit(state)
    _accept(state, log, first)
    requested = c.ConversationEffectRequested(
        replace(first.stamp, expected_revision=1, ordinal=1), "model", {"count": 1}, ()
    )
    prepared = reducer.prepare_record(
        state, ConversationEffectRequestedRecord(requested)
    )
    next_state = prepared._plan.next_conversation_state
    changed = c.ConversationEffectRequested(
        requested.stamp, "model", {"count": 1.0}, ()
    )
    segment = c.ConversationAuthority(first.baseline, (first, changed))
    forged = object.__new__(RuntimeConversationState)
    object.__setattr__(forged, "segments", (segment,))
    object.__setattr__(forged, "views", next_state.views)
    object.__setattr__(forged, "retired_invocations", next_state.retired_invocations)
    prepared = replace(
        prepared, _plan=replace(prepared._plan, next_conversation_state=forged)
    )
    before = _snapshot(state)
    with pytest.raises(ValueError, match="canonical checkpoint"):
        reducer.commit_prepared(log, state, prepared)
    assert _snapshot(state) == before


@pytest.mark.parametrize("field", ["occurrence", "terminal", "writer"])
def test_checkpoint_trace_cannot_inject_an_outcome_or_terminal(field):
    state, log = _started(learnable=True)
    prepared = reducer.prepare_record(state, ConversationAdmittedRecord(_admit(state)))
    trace_before = _trace_snapshot(state.trace)
    if field == "occurrence":
        value = state.trace.occurrences[0]
    elif field == "terminal":
        from agent_composer.runtime.trace import TraceTerminal

        value = TraceTerminal(
            False, None, None, Failure("node.execute", "invented", "agent")
        )
    else:
        value = ("invented", next(iter(state.trace._current_writers.values())))
    delta = replace(prepared._plan.trace_delta, **{field: value})
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=delta))
    with pytest.raises(ValueError, match="only advance"):
        reducer.commit_prepared(log, state, prepared)
    assert _trace_snapshot(state.trace) == trace_before
    assert log.next_sequence == state.next_sequence


def test_conversation_state_constructor_detaches_all_supplied_history():
    first = admission(c.ConversationInvocation("run", 3))
    view = advance_conversation(first.baseline, first)
    segment = c.ConversationAuthority(first.baseline, (first,))
    state = RuntimeConversationState((segment,), (view,))
    assert state.segments[0] is not segment
    assert state.views[0] is not view
    object.__setattr__(view, "revision", True)
    object.__setattr__(segment.checkpoints[0].stamp, "ordinal", True)
    assert state.views[0].revision == 1 and type(state.views[0].revision) is int
    assert state.segments[0].checkpoints[0].stamp.ordinal == 0


def test_conversation_state_constructor_detaches_retired_invocations():
    source = c.ConversationInvocation("run", 3)
    first = admission(source)
    view = advance_conversation(first.baseline, first)
    segment = c.ConversationAuthority(first.baseline, (first,))
    supplied = frozenset((source,))

    state = RuntimeConversationState((segment,), (view,), supplied)

    retained = next(iter(state.retired_invocations))
    assert state.retired_invocations is not supplied
    assert retained is not source
    object.__setattr__(source, "start_sequence", 99)
    expected = c.ConversationInvocation("run", 3)
    assert state.retired_invocations == frozenset((expected,))
    assert retained == state.segments[0].checkpoints[0].stamp.invocation
    assert state.index(expected) == 0
    with pytest.raises(ValueError, match="retired"):
        prepare_conversation_checkpoint(state, first)


def test_checkpoint_advances_only_its_invocation_without_refolding_older_segments(
    monkeypatch,
):
    import agent_composer.runtime.conversation as projection

    first = admission(c.ConversationInvocation("run", 3))
    current = prepare_conversation_checkpoint(RuntimeConversationState(), first)
    second = admission(c.ConversationInvocation("run", 5), consumer=("root", "other"))
    current = prepare_conversation_checkpoint(current, second)
    calls = []
    advance = projection.advance_conversation

    def tracked(before, checkpoint):
        calls.append(checkpoint.stamp.invocation)
        return advance(before, checkpoint)

    monkeypatch.setattr(projection, "advance_conversation", tracked)
    close = c.ConversationClosed(
        replace(second.stamp, expected_revision=1, ordinal=1), "completed"
    )
    after = prepare_conversation_checkpoint(current, close)
    assert calls == [second.stamp.invocation]
    assert after.segments[0] is current.segments[0]
    assert after.views[0] is current.views[0]
