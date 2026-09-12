"""Authoritative runtime projection and reducer transitions for memory learning."""

from __future__ import annotations

import copy

import pytest

import agent_composer.runtime.learning as learning_module
from agent_composer.compile.expand import build_grow
from agent_composer.compile.model import END_ID, START_ID
from agent_composer.diagnostics import Failure, OriginRef
from agent_composer.events import RunFailed, RunSucceeded
from agent_composer.memory import (
    EvaluationEvidence,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceRef,
    MemoryOccurrenceSegment,
    MemoryOperationFailure,
    MemoryOwner,
    MemoryScope,
    MemorySnapshotEnvelope,
    ModelTextInteraction,
)
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Grow, Node, NodeKind
from agent_composer.nodes.map import MapNode
from agent_composer.runtime.learning import (
    FrameLearningState,
    FrameLearningStatus,
    PolicyCommitStatus,
    RuntimeLearningState,
    aggregate_policy_mutations,
    memory_commit_id,
    project_outcome_episodes,
)
from agent_composer.runtime.log import (
    MemoryCommitFailedRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryFrameLearnedRecord,
    MemoryFrameLearningFailedRecord,
    MemoryFrameLearningStartedRecord,
    MemoryRunLog,
    NodeExpandedRecord,
    NodeRoutedRecord,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    _replay_records,
    commit_prepared,
    prepare_expanded,
    prepare_node_failed,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_record,
    prepare_run_started,
)
from agent_composer.typesys import VariablePool
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _forward_capture, _graph


_RUN_ID = "learning-run"
_FINGERPRINT = "a" * 64
_POLICY = "reflective"


class _TraceNode(Node):
    """Trait-capable node used without invoking an external AGENT mode."""

    kind = NodeKind.AGENT
    needs_evaluation_trace = True

    def forward(self, inputs, **_caps):
        return inputs


def _refs(node_id: str = "agent") -> tuple[MemoryFrameRef, MemoryFrameRef]:
    """Return one outer FLOW frame and its nested AGENT frame."""
    return (
        MemoryFrameRef("root", ("root", "flow"), MemoryOwner.FLOW),
        MemoryFrameRef(
            "root",
            ("root", "flow", node_id),
            MemoryOwner.AGENT,
        ),
    )


def _occurrence(index: int = 2) -> tuple[MemoryOccurrenceSegment, ...]:
    """Return one canonical dynamic MAP occurrence."""
    return (
        MemoryOccurrenceSegment(
            MemoryOccurrenceKind.MAP_ITEM,
            ("root", "flow"),
            index,
        ),
    )


def _scope(
    node_id: str = "agent",
    *,
    frames: tuple[MemoryFrameRef, ...] | None = None,
    occurrence: tuple[MemoryOccurrenceSegment, ...] | None = None,
) -> MemoryScope:
    """Return one exact trace scope for a test node."""
    stack = _refs(node_id) if frames is None else frames
    return MemoryScope(
        root_flow_id="root",
        static_path=stack[-1].static_path,
        frame_stack=stack,
        occurrence_path=_occurrence() if occurrence is None else occurrence,
    )


def _layout(
    frames: tuple[MemoryFrameRef, ...],
) -> tuple[MemoryFrameSpec, ...]:
    """Return canonical layout order for exact frames."""
    return tuple(
        MemoryFrameSpec(ref, _POLICY, {"limit": 3})
        for ref in sorted(
            frames,
            key=lambda item: (
                item.root_flow_id,
                item.static_path,
                item.owner.value,
            ),
        )
    )


def _new_state(
    node: Node,
    *,
    layout: tuple[MemoryFrameSpec, ...] | None,
    learn: bool,
) -> tuple[RunState, MemoryRunLog]:
    """Initialize a direct reducer authority and leave START ready."""
    flow = _graph([node], [(START_ID, node.id), (node.id, END_ID)])
    state = RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {})
    pool.add_system("run_id", _RUN_ID)
    snapshots = None if layout is None else tuple(
        MemorySnapshotEnvelope(policy_name, {"epoch": 4})
        for policy_name in sorted({spec.policy_name for spec in layout})
    )
    log = MemoryRunLog(_RUN_ID)
    commit_prepared(
        log,
        state,
        prepare_run_started(
            state,
            RunStartData(
                run_id=_RUN_ID,
                flow_fingerprint=_FINGERPRINT,
                initial_pool=pool,
                start_wiring_override={},
                effective_llm_by_static_path={},
                num_workers=0,
                memory_layout=layout,
                memory_snapshots=snapshots,
                learn=learn,
            ),
        ),
    )
    return state, log


def _start_trace_node(
    node: Node,
    *,
    layout: tuple[MemoryFrameSpec, ...] | None,
    learn: bool,
) -> tuple[RunState, MemoryRunLog]:
    """Drive START and leave ``node`` as the sole in-flight outcome."""
    state, log = _new_state(node, layout=layout, learn=learn)
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(
        state, START_ID, {},
        forward_capture=_forward_capture(state.flow.nodes[START_ID], {}) if learn else None,
    ))
    commit_prepared(log, state, prepare_node_started(state, node.id))
    return state, log


def _evidence(scope: MemoryScope | None) -> EvaluationEvidence:
    """Return deterministic visible evidence for one accepted outcome."""
    return EvaluationEvidence(
        scope,
        {"question": "q"},
        "system",
        "prompt",
        interactions=(ModelTextInteraction("answer"),),
    )


def _successful_terminal(
    *,
    frames: tuple[MemoryFrameRef, ...] | None = None,
) -> tuple[RunState, MemoryRunLog, tuple[MemoryFrameSpec, ...]]:
    """Return a terminal-success run with one accepted trace episode."""
    node = _TraceNode("agent")
    scope = _scope(node.id, frames=frames)
    node.memory_scope = scope
    layout = _layout(scope.frame_stack)
    state, log = _start_trace_node(node, layout=layout, learn=True)
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(
            state,
            node.id,
            {"answer": 7},
            evidence=_evidence(scope),
            forward_capture=_forward_capture(node, {"question": "q"}),
        ),
    )
    commit_prepared(log, state, prepare_node_started(state, END_ID))
    commit_prepared(log, state, prepare_node_succeeded(
        state, END_ID, None, forward_capture=_forward_capture(state.flow.nodes[END_ID], {}),
    ))
    assert isinstance(state.result, RunSucceeded)
    return state, log, layout


def _commit_record(state: RunState, log: MemoryRunLog, record) -> None:
    """Prepare and commit one reducer record, asserting no learning observers."""
    prepared = prepare_record(state, record)
    assert prepared.observers == ()
    assert commit_prepared(log, state, prepared) == ()


def test_runstarted_retains_defensive_memory_authority_and_projects_episodes() -> None:
    state, _log, layout = _successful_terminal()
    outer, agent = _refs()

    assert state.memory_layout == layout
    assert state.memory_layout is not layout
    assert state.memory_snapshots == (
        MemorySnapshotEnvelope(_POLICY, {"epoch": 4}),
    )
    assert state.learn is True
    assert state.learning_state.frame_order == (outer, agent)

    outer_episode = state.learning_state.episodes_for(outer)
    agent_episode = state.learning_state.episodes_for(agent)
    assert len(outer_episode) == len(agent_episode) == 1
    assert outer_episode[0].owner_static_path == agent.static_path
    assert outer_episode[0].occurrence == MemoryOccurrenceRef(outer, _occurrence())
    assert agent_episode[0].occurrence == MemoryOccurrenceRef(agent, _occurrence())
    assert outer_episode[0].output == {"answer": 7}
    assert outer_episode[0].inputs == {"question": "q"}
    assert outer_episode[0].interactions == (ModelTextInteraction("answer"),)
    context = state.learning_state.learn_context(_RUN_ID, outer)
    assert context.episodes == outer_episode
    assert context.feedback == ()

    detached = state.memory_layout[0].config
    detached["limit"] = 999
    snapshot = state.memory_snapshots[0].snapshot
    snapshot["epoch"] = 999
    assert state.memory_layout[0].config == {"limit": 3}
    assert state.memory_snapshots[0].snapshot == {"epoch": 4}


def test_frame_learning_and_commit_follow_record_authority_and_layout_order() -> None:
    state, log, layout = _successful_terminal()
    outer, agent = _refs()

    # The reducer allows independent frames in any execution order; aggregation
    # still follows the canonical recorded layout and each frame's proposal order.
    for ref, mutations in (
        (
            agent,
            (
                MemoryMutation("agent-2", agent, {"order": 2}),
                MemoryMutation("agent-3", agent, {"order": 3}),
            ),
        ),
        (outer, (MemoryMutation("flow-1", outer, {"order": 1}),)),
    ):
        occurrences = state.learning_state.occurrence_inventory(ref)
        _commit_record(
            state,
            log,
            MemoryFrameLearningStartedRecord(_POLICY, ref, occurrences),
        )
        assert state.learning_state.active_frame == ref
        _commit_record(
            state,
            log,
            MemoryFrameLearnedRecord(_POLICY, ref, mutations),
        )
        assert (
            state.learning_state.frame_state(ref).status
            is FrameLearningStatus.LEARNED
        )

    batch = aggregate_policy_mutations(layout, state.learning_state, _POLICY)
    assert [mutation.mutation_id for mutation in batch] == [
        "flow-1",
        "agent-2",
        "agent-3",
    ]
    commit_id = memory_commit_id(_RUN_ID, _POLICY, batch)
    _commit_record(
        state,
        log,
        MemoryCommitRequestedRecord(_POLICY, commit_id, batch),
    )
    requested = state.learning_state.commit_state(_POLICY)
    assert requested.status is PolicyCommitStatus.REQUESTED
    assert requested.commit_id == commit_id
    assert requested.mutations == batch

    _commit_record(
        state,
        log,
        MemoryCommitSucceededRecord(_POLICY, commit_id),
    )
    assert (
        state.learning_state.commit_state(_POLICY).status
        is PolicyCommitStatus.SUCCEEDED
    )


def test_learning_failure_retry_final_and_primary_result_are_independent() -> None:
    agent = _refs()[1]
    state, log, _layout_value = _successful_terminal(frames=(agent,))
    primary = state.result
    occurrences = state.learning_state.occurrence_inventory(agent)

    _commit_record(
        state,
        log,
        MemoryFrameLearningStartedRecord(_POLICY, agent, occurrences),
    )
    _commit_record(
        state,
        log,
        MemoryFrameLearningFailedRecord(
            _POLICY,
            agent,
            MemoryOperationFailure("learn", "temporary", "retry", True),
        ),
    )
    assert (
        state.learning_state.frame_state(agent).status
        is FrameLearningStatus.FAILED_RETRYABLE
    )
    assert state.result is primary

    # Segment timing is orchestration-owned; the reducer validates the durable
    # failed_retryable -> started edge because no post-terminal resume marker exists.
    _commit_record(
        state,
        log,
        MemoryFrameLearningStartedRecord(_POLICY, agent, occurrences),
    )
    _commit_record(
        state,
        log,
        MemoryFrameLearningFailedRecord(
            _POLICY,
            agent,
            MemoryOperationFailure("learn", "permanent", "stop", False),
        ),
    )
    assert (
        state.learning_state.frame_state(agent).status
        is FrameLearningStatus.FAILED_FINAL
    )
    with pytest.raises(ValueError, match="cannot start from failed_final"):
        prepare_record(
            state,
            MemoryFrameLearningStartedRecord(_POLICY, agent, occurrences),
        )
    assert state.result is primary


def test_failure_episode_uses_full_canonical_durable_failure_json() -> None:
    agent = _refs()[1]
    node = _TraceNode("agent")
    scope = _scope(node.id, frames=(agent,))
    node.memory_scope = scope
    layout = _layout((agent,))
    state, _log = _start_trace_node(node, layout=layout, learn=True)
    failure = Failure(
        "node.execute",
        "visible",
        node.id,
        trace=(OriginRef("outer"), OriginRef("inner")),
        cause_type="ValueError",
        traceback="debug trace",
    )
    prepared = prepare_node_failed(
        state, failure, evidence=_evidence(scope),
        forward_capture=_forward_capture(node, {"question": "q"}),
    )
    commit_prepared(_log, state, prepared)

    assert isinstance(state.result, RunFailed)
    episode = state.learning_state.episodes_for(agent)[0]
    assert episode.failure == {
        "code": "node.execute",
        "message": "visible",
        "node_id": "agent",
        "trace": [{"key": "outer"}, {"key": "inner"}],
        "cause_type": "ValueError",
        "traceback": "debug trace",
    }
    assert episode.output is None


@pytest.mark.parametrize("enabled,learn", [(False, False), (True, False)])
def test_evidence_is_rejected_when_learning_is_not_enabled(
    enabled: bool,
    learn: bool,
) -> None:
    node = _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    layout = _layout(node.memory_scope.frame_stack) if enabled else None
    state, _log = _start_trace_node(node, layout=layout, learn=learn)

    with pytest.raises(ValueError, match="ineligible for evaluation evidence"):
        prepare_node_succeeded(
            state,
            node.id,
            "done",
            evidence=_evidence(node.memory_scope),
        )
    assert prepare_node_succeeded(state, node.id, "done").record.evidence is None


def test_learning_requires_evidence_and_rejects_ineligible_or_unrecorded_scope(
) -> None:
    node = _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    layout = _layout(node.memory_scope.frame_stack)
    state, _log = _start_trace_node(node, layout=layout, learn=True)
    with pytest.raises(ValueError, match="requires evaluation evidence"):
        prepare_node_succeeded(
            state, node.id, "done", forward_capture=_forward_capture(node, {}),
        )

    ordinary = FuncNode("ordinary", lambda _inputs: "done")
    ordinary.memory_scope = _scope(ordinary.id)
    ordinary_layout = _layout(ordinary.memory_scope.frame_stack)
    ordinary_state, _ = _start_trace_node(
        ordinary,
        layout=ordinary_layout,
        learn=True,
    )
    with pytest.raises(ValueError, match="ineligible"):
        prepare_node_succeeded(
            ordinary_state,
            ordinary.id,
            "done",
            evidence=_evidence(ordinary.memory_scope),
            forward_capture=_forward_capture(ordinary, {"question": "q"}),
        )
    assert (
        prepare_node_succeeded(
            ordinary_state, ordinary.id, "done",
            forward_capture=_forward_capture(ordinary, {}),
        ).record.evidence
        is None
    )

    wrong_agent = MemoryFrameRef(
        "root",
        ("root", "flow", "wrong"),
        MemoryOwner.AGENT,
    )
    mismatched = _TraceNode("agent")
    mismatched.memory_scope = _scope(mismatched.id)
    mismatched_layout = _layout((_refs()[0], wrong_agent))
    mismatched_state, _ = _start_trace_node(
        mismatched,
        layout=mismatched_layout,
        learn=True,
    )
    with pytest.raises(ValueError, match="exact recorded frame stack"):
        prepare_node_succeeded(
            mismatched_state,
            mismatched.id,
            "done",
            evidence=_evidence(mismatched.memory_scope),
            forward_capture=_forward_capture(mismatched, {"question": "q"}),
        )


def test_empty_enabled_layout_retains_scope_free_evidence_without_episodes() -> None:
    node = _TraceNode("agent")
    state, _log = _start_trace_node(node, layout=(), learn=True)

    prepared = prepare_node_succeeded(
        state, node.id, "done", evidence=_evidence(None),
        forward_capture=_forward_capture(node, {"question": "q"}),
    )
    assert prepared.record.evidence.scope is None
    assert project_outcome_episodes(_RUN_ID, prepared.record) == ()
    with pytest.raises(ValueError, match="evidence scope does not match"):
        prepare_node_succeeded(
            state,
            node.id,
            "done",
            evidence=_evidence(_scope(node.id)),
            forward_capture=_forward_capture(node, {"question": "q"}),
        )


@pytest.mark.parametrize("retained", ["outer", "agent"])
def test_learning_rejects_trace_scope_that_omits_an_applicable_frame(
    retained: str,
) -> None:
    outer, agent = _refs()
    frame_stack = (outer,) if retained == "outer" else (agent,)
    node = _TraceNode("agent")
    node.memory_scope = MemoryScope(
        root_flow_id="root",
        static_path=agent.static_path,
        frame_stack=frame_stack,
        occurrence_path=_occurrence(),
    )
    state, _log = _start_trace_node(
        node,
        layout=_layout((outer, agent)),
        learn=True,
    )

    with pytest.raises(ValueError, match="exact recorded frame stack"):
        prepare_node_succeeded(
            state, node.id, "done", forward_capture=_forward_capture(node, {}),
        )


def test_learning_records_require_primary_terminal_and_enabled_learn_flag() -> None:
    agent = _refs()[1]
    node = _TraceNode("agent")
    scope = _scope(node.id, frames=(agent,))
    node.memory_scope = scope
    layout = _layout((agent,))
    state, log = _start_trace_node(node, layout=layout, learn=True)
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(
            state,
            node.id,
            "done",
            evidence=_evidence(scope),
            forward_capture=_forward_capture(node, {"question": "q"}),
        ),
    )
    occurrences = state.learning_state.occurrence_inventory(agent)
    start = MemoryFrameLearningStartedRecord(_POLICY, agent, occurrences)
    with pytest.raises(ValueError, match="primary terminal"):
        prepare_record(state, start)

    disabled_node = _TraceNode("agent")
    disabled_node.memory_scope = scope
    disabled, disabled_log = _start_trace_node(
        disabled_node,
        layout=layout,
        learn=False,
    )
    commit_prepared(
        disabled_log,
        disabled,
        prepare_node_succeeded(disabled, disabled_node.id, "done"),
    )
    commit_prepared(
        disabled_log,
        disabled,
        prepare_node_started(disabled, END_ID),
    )
    commit_prepared(
        disabled_log,
        disabled,
        prepare_node_succeeded(disabled, END_ID, None),
    )
    with pytest.raises(ValueError, match="enabled learning"):
        prepare_record(disabled, start)


def test_started_and_requested_crash_states_replay_without_duplicate_records() -> None:
    agent = _refs()[1]
    state, log, layout = _successful_terminal(frames=(agent,))
    occurrences = state.learning_state.occurrence_inventory(agent)
    _commit_record(
        state,
        log,
        MemoryFrameLearningStartedRecord(_POLICY, agent, occurrences),
    )

    replay_started = _replay_records(
        RunState(copy.deepcopy(state.flow)),
        tuple(log),
        grow_builder=lambda _node, _seed: pytest.fail("no growth expected"),
    )
    assert replay_started.learning_state == state.learning_state
    assert (
        replay_started.learning_state.frame_state(agent).status
        is FrameLearningStatus.STARTED
    )
    with pytest.raises(ValueError, match="active"):
        prepare_record(
            replay_started,
            MemoryFrameLearningStartedRecord(_POLICY, agent, occurrences),
        )

    mutation = MemoryMutation("m", agent, {"lesson": "kept"})
    _commit_record(
        state,
        log,
        MemoryFrameLearnedRecord(_POLICY, agent, (mutation,)),
    )
    batch = aggregate_policy_mutations(layout, state.learning_state, _POLICY)
    commit_id = memory_commit_id(_RUN_ID, _POLICY, batch)
    request = MemoryCommitRequestedRecord(_POLICY, commit_id, batch)
    _commit_record(state, log, request)

    replay_requested = _replay_records(
        RunState(copy.deepcopy(state.flow)),
        tuple(log),
        grow_builder=lambda _node, _seed: pytest.fail("no growth expected"),
    )
    assert replay_requested.learning_state == state.learning_state
    assert (
        replay_requested.learning_state.commit_state(_POLICY).status
        is PolicyCommitStatus.REQUESTED
    )
    with pytest.raises(ValueError, match="cannot be requested from requested"):
        prepare_record(replay_requested, request)


def test_expanded_map_occurrence_projects_and_replays_from_live_node_scope() -> None:
    outer = _refs()[0]
    child_agent = AgentNode("agent", prompt="visible")
    child = _graph(
        [child_agent],
        [(START_ID, child_agent.id), (child_agent.id, END_ID)],
    )
    for node_id, node in child.nodes.items():
        node.memory_scope = MemoryScope(
            root_flow_id="root",
            static_path=("root", "flow", "map", "child", node_id),
            frame_stack=(outer,),
        )
    spawner = MapNode(
        "map",
        flow_id="child",
        child=child,
        child_inputs=[],
    )
    spawner.memory_scope = MemoryScope(
        root_flow_id="root",
        static_path=("root", "flow", "map"),
        frame_stack=(outer,),
    )
    state, log = _start_trace_node(
        spawner,
        layout=_layout((outer,)),
        learn=True,
    )
    pristine_flow = copy.deepcopy(state.flow)

    commit_prepared(
        log,
        state,
        prepare_expanded(
            state, spawner.id, Grow(seed=[{}]), build_grow,
            forward_capture=_forward_capture(spawner, {"over": [{}]}),
        ),
    )

    def succeed(node_id: str, value, *, evidence=None) -> None:
        assert state.ready[0] == node_id
        commit_prepared(log, state, prepare_node_started(state, node_id))
        commit_prepared(
            log,
            state,
            prepare_node_succeeded(
                state, node_id, value, evidence=evidence,
                forward_capture=_forward_capture(
                    state.flow.nodes[node_id], {} if evidence is None else evidence.bound_inputs,
                ),
            ),
        )

    succeed("map/__start__", {})
    succeed("map#0/__start__", {})
    dynamic_agent = state.flow.nodes["map#0/agent"]
    dynamic_scope = dynamic_agent.memory_scope
    assert type(dynamic_scope) is MemoryScope
    succeed(
        dynamic_agent.id,
        "visible",
        evidence=_evidence(dynamic_scope),
    )
    succeed("map#0/__end__", None)
    succeed("map/__end__", [None])
    succeed(END_ID, None)
    assert isinstance(state.result, RunSucceeded)

    episode = state.learning_state.episodes_for(outer)[0]
    assert episode.owner_static_path == dynamic_scope.static_path
    assert episode.occurrence.occurrence_path == (
        MemoryOccurrenceSegment(
            MemoryOccurrenceKind.MAP_ITEM,
            spawner.memory_scope.static_path,
            0,
        ),
    )

    replayed = _replay_records(
        RunState(pristine_flow),
        tuple(log),
        grow_builder=build_grow,
    )
    assert replayed.learning_state == state.learning_state
    assert replayed.learning_state.episodes_for(outer) == (episode,)


def test_retryable_and_final_commit_failures_control_re_request() -> None:
    agent = _refs()[1]
    state, log, layout = _successful_terminal(frames=(agent,))
    occurrences = state.learning_state.occurrence_inventory(agent)
    _commit_record(
        state,
        log,
        MemoryFrameLearningStartedRecord(_POLICY, agent, occurrences),
    )
    mutation = MemoryMutation("m", agent, {"lesson": 1})
    _commit_record(
        state,
        log,
        MemoryFrameLearnedRecord(_POLICY, agent, (mutation,)),
    )
    batch = aggregate_policy_mutations(layout, state.learning_state, _POLICY)
    commit_id = memory_commit_id(_RUN_ID, _POLICY, batch)
    request = MemoryCommitRequestedRecord(_POLICY, commit_id, batch)
    _commit_record(state, log, request)
    _commit_record(
        state,
        log,
        MemoryCommitFailedRecord(
            _POLICY,
            commit_id,
            MemoryOperationFailure("commit", "temporary", "retry", True),
        ),
    )
    assert (
        state.learning_state.commit_state(_POLICY).status
        is PolicyCommitStatus.FAILED_RETRYABLE
    )

    _commit_record(state, log, request)
    _commit_record(
        state,
        log,
        MemoryCommitFailedRecord(
            _POLICY,
            commit_id,
            MemoryOperationFailure("commit", "permanent", "stop", False),
        ),
    )
    assert (
        state.learning_state.commit_state(_POLICY).status
        is PolicyCommitStatus.FAILED_FINAL
    )
    with pytest.raises(ValueError, match="cannot be requested from failed_final"):
        prepare_record(state, request)


def test_learning_records_validate_active_frame_inventory_policy_batch_and_id() -> None:
    state, log, layout = _successful_terminal()
    outer, agent = _refs()
    outer_occurrences = state.learning_state.occurrence_inventory(outer)
    agent_occurrences = state.learning_state.occurrence_inventory(agent)

    with pytest.raises(ValueError, match="accepted inventory"):
        prepare_record(
            state,
            MemoryFrameLearningStartedRecord(_POLICY, outer, ()),
        )
    with pytest.raises(ValueError, match="does not match recorded frame policy"):
        prepare_record(
            state,
            MemoryFrameLearningStartedRecord("wrong", outer, outer_occurrences),
        )
    _commit_record(
        state,
        log,
        MemoryFrameLearningStartedRecord(_POLICY, outer, outer_occurrences),
    )
    with pytest.raises(ValueError, match="another memory frame"):
        prepare_record(
            state,
            MemoryFrameLearningStartedRecord(_POLICY, agent, agent_occurrences),
        )
    with pytest.raises(ValueError, match="active memory frame"):
        prepare_record(
            state,
            MemoryFrameLearnedRecord(
                _POLICY,
                agent,
                (MemoryMutation("wrong", agent, {}),),
            ),
        )
    _commit_record(
        state,
        log,
        MemoryFrameLearnedRecord(
            _POLICY,
            outer,
            (MemoryMutation("outer", outer, {}),),
        ),
    )
    with pytest.raises(ValueError, match="every frame"):
        prepare_record(
            state,
            MemoryCommitRequestedRecord(
                _POLICY,
                memory_commit_id(_RUN_ID, _POLICY, ()),
                (),
            ),
        )

    _commit_record(
        state,
        log,
        MemoryFrameLearningStartedRecord(_POLICY, agent, agent_occurrences),
    )
    collision = MemoryFrameLearnedRecord(
        _POLICY,
        agent,
        (MemoryMutation("outer", agent, {}),),
    )
    with pytest.raises(ValueError, match="unique within one policy"):
        prepare_record(state, collision)

    forged = MemoryRunLog(_RUN_ID)
    for envelope in log:
        forged.append(envelope.record)
    forged.append(collision)
    with pytest.raises(ValueError, match="unique within one policy"):
        _replay_records(
            RunState(copy.deepcopy(state.flow)),
            tuple(forged),
            grow_builder=build_grow,
        )

    _commit_record(
        state,
        log,
        MemoryFrameLearnedRecord(
            _POLICY,
            agent,
            (MemoryMutation("agent", agent, {}),),
        ),
    )
    batch = aggregate_policy_mutations(layout, state.learning_state, _POLICY)
    with pytest.raises(ValueError, match="exactly aggregate"):
        prepare_record(
            state,
            MemoryCommitRequestedRecord(
                _POLICY,
                memory_commit_id(_RUN_ID, _POLICY, tuple(reversed(batch))),
                tuple(reversed(batch)),
            ),
        )
    with pytest.raises(ValueError, match="canonical batch"):
        prepare_record(
            state,
            MemoryCommitRequestedRecord(_POLICY, "wrong-id", batch),
        )


def test_learning_append_failure_leaves_projection_sequence_and_result_unchanged(
) -> None:
    agent = _refs()[1]
    state, log, _layout_value = _successful_terminal(frames=(agent,))
    record = MemoryFrameLearningStartedRecord(
        _POLICY,
        agent,
        state.learning_state.occurrence_inventory(agent),
    )
    prepared = prepare_record(state, record)
    original_learning = state.learning_state
    original_sequence = state.next_sequence
    original_result = state.result

    class _FailingLog:
        run_id = log.run_id
        next_sequence = log.next_sequence

        def append(self, _record):
            raise OSError("append failed")

    with pytest.raises(OSError, match="append failed"):
        commit_prepared(_FailingLog(), state, prepared)
    assert state.learning_state is original_learning
    assert state.next_sequence == original_sequence
    assert state.result is original_result


def test_route_and_growth_projection_use_closed_source_neutral_envelopes() -> None:
    agent = _refs()[1]
    scope = _scope(frames=(agent,))
    evidence = _evidence(scope)
    route = project_outcome_episodes(
        _RUN_ID,
        NodeRoutedRecord("agent", "yes", evidence=evidence),
    )[0][1]
    growth = project_outcome_episodes(
        _RUN_ID,
        NodeExpandedRecord(
            "agent",
            {"value": (1, 2)},
            ("a", "z"),
            "b" * 64,
            evidence=evidence,
        ),
    )[0][1]

    assert route.output == {"kind": "route", "handle": "yes"}
    assert growth.output == {
        "kind": "grow",
        "seed": {"value": [1, 2]},
        "prune_ids": ["a", "z"],
    }


def test_episode_order_is_record_order_while_occurrence_inventory_is_structural(
) -> None:
    agent = _refs()[1]
    layout = _layout((agent,))
    learning = RuntimeLearningState.for_layout(layout)
    for index, handle in ((2, "second-structurally"), (0, "first-structurally")):
        scope = _scope(frames=(agent,), occurrence=_occurrence(index))
        record = NodeRoutedRecord(
            "agent",
            handle,
            evidence=_evidence(scope),
        )
        learning = learning.append_episodes(
            project_outcome_episodes(_RUN_ID, record)
        )

    assert [episode.output["handle"] for episode in learning.episodes_for(agent)] == [
        "second-structurally",
        "first-structurally",
    ]
    assert [
        occurrence.occurrence_path[0].index
        for occurrence in learning.occurrence_inventory(agent)
    ] == [0, 2]


def test_episode_append_validates_only_new_persistent_history(monkeypatch) -> None:
    agent = _refs()[1]
    learning = RuntimeLearningState.for_layout(_layout((agent,)))
    original = learning_module._revalidate_episode
    validations = 0

    def count_validation(episode) -> None:
        nonlocal validations
        validations += 1
        original(episode)

    monkeypatch.setattr(learning_module, "_revalidate_episode", count_validation)
    for index in range(129):
        scope = _scope(frames=(agent,), occurrence=_occurrence(index))
        learning = learning.append_episodes(
            project_outcome_episodes(
                _RUN_ID,
                NodeRoutedRecord("agent", str(index), evidence=_evidence(scope)),
            )
        )

    assert validations == 129
    episodes = learning.episodes_for(agent)
    assert [episode.output["handle"] for episode in episodes] == [
        str(index) for index in range(129)
    ]
    with pytest.raises(ValueError, match="run_id disagrees"):
        learning.learn_context("another-run", agent)
    with pytest.raises(ValueError, match="run_id disagrees"):
        learning.append_episodes(
            project_outcome_episodes(
                "another-run",
                NodeRoutedRecord(
                    "agent",
                    "foreign",
                    evidence=_evidence(_scope(frames=(agent,))),
                ),
            )
        )


def test_large_frame_inventory_uses_indexed_lookup(monkeypatch) -> None:
    refs = tuple(
        MemoryFrameRef(
            "root",
            ("root", "agents", f"agent-{index:04d}"),
            MemoryOwner.AGENT,
        )
        for index in range(1024)
    )
    learning = RuntimeLearningState.for_layout(_layout(refs))
    original_equality = MemoryFrameRef.__eq__
    comparisons = 0

    def count_equality(left, right) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original_equality(left, right)

    monkeypatch.setattr(MemoryFrameRef, "__eq__", count_equality)
    for ref in refs:
        assert learning.frame_state(ref).status is FrameLearningStatus.PENDING

    assert comparisons <= len(refs) * 2


def test_graph_outcome_uses_constant_memory_roots_and_indexed_scope(
    monkeypatch,
) -> None:
    refs = tuple(
        MemoryFrameRef(
            "root",
            ("root", "agents", f"agent-{index:04d}"),
            MemoryOwner.AGENT,
        )
        for index in range(1024)
    )
    target = refs[-1]
    node = _TraceNode("agent")
    node.memory_scope = MemoryScope(
        root_flow_id="root",
        static_path=target.static_path,
        frame_stack=(target,),
    )
    state, log = _start_trace_node(node, layout=_layout(refs), learn=True)

    def reject_full_layout_validation(*_args, **_kwargs) -> None:
        pytest.fail("hot graph transition rescanned the complete memory layout")

    original_applicable = RuntimeLearningState.applicable_frames
    applicable_calls = 0

    def count_applicable(self, root_flow_id, static_path):
        nonlocal applicable_calls
        applicable_calls += 1
        return original_applicable(self, root_flow_id, static_path)

    original_frame_key = learning_module.frame_key
    key_calls = 0

    def count_frame_key(ref):
        nonlocal key_calls
        key_calls += 1
        return original_frame_key(ref)

    monkeypatch.setattr(
        RuntimeLearningState,
        "validate_layout",
        reject_full_layout_validation,
    )
    monkeypatch.setattr(
        RuntimeLearningState,
        "applicable_frames",
        count_applicable,
    )
    monkeypatch.setattr(learning_module, "frame_key", count_frame_key)

    prepared = prepare_node_succeeded(
        state,
        node.id,
        "done",
        evidence=_evidence(node.memory_scope),
        forward_capture=_forward_capture(node, {"question": "q"}),
    )
    commit_prepared(log, state, prepared)

    assert applicable_calls == 1
    assert key_calls < 100


def test_policy_aggregation_uses_cached_frame_partition(monkeypatch) -> None:
    refs = tuple(
        MemoryFrameRef(
            "root",
            ("root", "agents", f"agent-{index:04d}"),
            MemoryOwner.AGENT,
        )
        for index in range(512)
    )
    layout = tuple(
        MemoryFrameSpec(ref, f"policy-{index:04d}", {})
        for index, ref in enumerate(refs)
    )
    learning = RuntimeLearningState.for_layout(layout)
    for ref in refs:
        learning = learning.with_frame_state(
            ref,
            FrameLearningState(FrameLearningStatus.STARTED),
        )
        learning = learning.with_frame_state(
            ref,
            FrameLearningState(FrameLearningStatus.LEARNED),
        )

    def reject_layout_scan(*_args, **_kwargs) -> None:
        pytest.fail("trusted policy aggregation rescanned the complete layout")

    monkeypatch.setattr(learning_module, "_validate_layout", reject_layout_scan)
    monkeypatch.setattr(
        RuntimeLearningState,
        "validate_layout",
        reject_layout_scan,
    )
    for index in range(len(refs)):
        assert learning.policy_mutations(f"policy-{index:04d}") == ()

    assert learning.policy_frame_indices == tuple(
        (index,) for index in range(len(refs))
    )


def test_commit_id_has_one_documented_canonical_encoding() -> None:
    ref = MemoryFrameRef("root", ("root", "agent"), MemoryOwner.AGENT)
    mutations = (
        MemoryMutation("m", ref, {"z": 1, "a": ["é", None]}),
    )

    assert memory_commit_id("run", "policy", mutations) == (
        "ac-memory-v1:"
        "6c8c1507c9c8c0d6ca4a6239ec397ca950fae2f206146db324ce22098e22f5d6"
    )
