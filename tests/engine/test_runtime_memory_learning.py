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
    MemoryOwner,
    MemoryScope,
    MemorySnapshotEnvelope,
    ModelTextInteraction,
)
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Grow, Node, NodeKind
from agent_composer.nodes.map import MapNode
from agent_composer.runtime.learning import (
    RuntimeLearningState,
    memory_commit_id,
    project_outcome_episodes,
)
from agent_composer.runtime.log import (
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


def test_owner_contexts_include_general_occurrences_and_empty_frames(monkeypatch):
    from agent_composer.compile.model import FlowOutput
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.nodes.contracts import Parameter
    from agent_composer.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode
    from tests.engine._graph_builder import _engine

    outer = MemoryFrameRef("root", ("root",), MemoryOwner.FLOW)
    local = MemoryFrameRef("root", ("root", "work"), MemoryOwner.AGENT)
    empty = MemoryFrameRef("root", ("root", "unused"), MemoryOwner.FLOW)
    layout = _layout((outer, local, empty))
    node = RecordingBackwardNode(
        "work", lambda inputs: inputs["topic"],
        lambda _context, feedback: BackwardResult(
            {"topic": feedback, "literal": ("keep literal",)},
            self_feedback=("first", "second"), variable_feedback={"tone": ("one", "two")},
        ),
        reads={"topic": "${input.topic}", "literal": {"default": [1]}},
    )
    node.declared_variables = lambda: (Parameter("tone"),)
    flow = _graph([node], [(START_ID, "work")], outputs=[FlowOutput("answer", "${work.output}")])
    for current in flow.nodes.values():
        stack = (outer, local) if current is node else (outer,)
        current.memory_scope = MemoryScope("root", ("root", current.id), stack)
    engine = _engine(flow, {"topic": "reefs"}, learnable=True, memory_layout=layout,
                     memory_snapshots=(MemorySnapshotEnvelope(_POLICY, {}),))
    list(engine.run())
    trace = engine.state.trace
    root, results = compute_backward(trace, "review")
    monkeypatch.setattr(node, "callback", lambda *_: pytest.fail("projection invoked backward"))

    contexts = learning_module.project_owner_contexts(trace, layout, "review", root, results)

    assert tuple(context.frame_ref for context in contexts) == tuple(spec.ref for spec in layout)
    by_frame = {context.frame_ref: context for context in contexts}
    assert len(by_frame[outer].episodes) == len(trace.occurrences) == 3
    assert by_frame[empty].episodes == () and by_frame[empty].feedback == ()
    local_episode, = by_frame[local].episodes
    assert local_episode.inputs == {"topic": "reefs", "literal": {"default": [1]}}
    assert local_episode.interactions == ()
    inherited = next(episode for episode in by_frame[outer].episodes
                     if episode.execution_seq == local_episode.execution_seq)
    assert inherited.owner_static_path == local_episode.owner_static_path == ("root", "work")
    assert inherited.occurrence.frame_ref == outer
    assert local_episode.occurrence.frame_ref == local
    entries = by_frame[local].feedback
    assert [(entry["channel"], entry["target_name"], entry["value"]) for entry in entries] == [
        ("output", None, "review"), ("input", "topic", "review"),
        ("input", "literal", "keep literal"), ("self", None, "first"), ("self", None, "second"),
        ("variable", "tone", "one"), ("variable", "tone", "two"),
    ]
    occurrence = next(item for item in trace.occurrences if item.node_id == "work")
    for entry in entries:
        assert entry["execution_seq"] == occurrence.source.execution_seq
        assert entry["owner_static_path"] == ["work"]
        assert entry["occurrence_path"] == []
        assert entry["execution_scope"] == {
            "static_path": ["work"], "flow_path": [], "occurrence_path": [], "boundary": None,
        }
    assert [(entry["message_index"], entry["contribution_context"]) for entry in entries[3:]] == [
        (0, {}), (1, {}), (0, {}), (1, {}),
    ]
    assert entries[2]["contribution_context"]["contributions"] == []
    assert [entry for entry in entries if entry["channel"] == "stopped"] == []
    entries[0]["value"] = "mutated"
    local_episode.inputs["literal"]["default"].append(2)
    assert by_frame[local].feedback[0]["value"] == "review"
    assert by_frame[local].episodes[0].inputs["literal"] == {"default": [1]}


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


def _owner_engine(flow, inputs=None, **kwargs):
    """Use the real static scope pass with the existing direct engine fixture."""
    from agent_composer.compile.llm_cascade import apply_effective_llm_map
    from agent_composer.compile.memory_layout import apply_memory_layout, resolve_memory_layout
    from agent_composer.memory import MemoryPolicySpec
    from tests.engine._graph_builder import _engine

    flow.flow_id = flow.flow_id or "root"
    flow.memory_policy = MemoryPolicySpec(_POLICY, {})
    apply_effective_llm_map(flow, {})
    resolved = resolve_memory_layout(flow)
    apply_memory_layout(flow, resolved)
    layout = tuple(frame.spec for frame in resolved.frames)
    engine = _engine(flow, inputs, learnable=True, memory_layout=layout,
                     memory_snapshots=(MemorySnapshotEnvelope(_POLICY, {}),), **kwargs)
    return engine, layout


@pytest.mark.parametrize("opaque", [False, True])
def test_owner_feedback_retains_literals_and_opaque_local_stops(opaque, monkeypatch):
    from agent_composer.compile.model import FlowOutput
    from agent_composer.expr.builtins import TEMPLATE_FNS
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode

    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "owner_identity", lambda value: calls.append(value) or value)
    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": "${owner_identity(input.topic)}" if opaque else "literal"})
    flow = _graph([node], [(START_ID, "work")], outputs=[FlowOutput("answer", "${work.output}")])
    engine, layout = _owner_engine(flow, {"topic": "reef"})
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "unchanged")
    contexts = learning_module.project_owner_contexts(engine.state.trace, layout, "unchanged", root, results)
    work = [entry for entry in contexts[0].feedback if entry["owner_static_path"] == ["work"]]
    expected = [("output", "unchanged"), ("input", "unchanged")]
    if opaque:
        expected.append(("stopped", "unchanged"))
    assert [(entry["channel"], entry["value"]) for entry in work] == expected
    assert work[1]["target_name"] == "x" and work[1]["contribution_context"]["contributions"] == []
    if opaque:
        stopped = work[2]["contribution_context"]
        assert "contribution_index" not in stopped
        assert any(detail["role"] == "opaque" and detail["result"] == "reef"
                   and detail["sources"][0]["value"] == "reef" for detail in stopped["contexts"])
    assert calls == (["reef"] if opaque else [])
    assert root.input_feedback == {}


@pytest.mark.parametrize("failure_kind", ["binding", "body", "route"])
def test_owner_contexts_keep_unavailable_inputs_and_causal_failures(failure_kind):
    from agent_composer.compile.model import FlowOutput
    from agent_composer.memory import MemoryEpisodeStatus
    from agent_composer.nodes.base import Route
    from agent_composer.nodes.contracts import Parameter
    from agent_composer.runtime.backward import compute_backward

    def execute(inputs):
        if failure_kind == "body":
            raise ValueError("expected body failure")
        return Route("unselected")

    node = FuncNode("work", execute, reads={"x": [1]})
    flow = _graph([node], [(START_ID, "work")], outputs=[FlowOutput("answer", "${work.output}")])
    if failure_kind == "binding":
        node.params = [Parameter("x", required=True)]
        flow.wiring["work"] = {}
    elif failure_kind == "route":
        from tests.engine.test_run_state_outcomes import _terminal_skip_route_flow

        flow = _terminal_skip_route_flow()
        node = flow.nodes["router"]
        node.forward = lambda _: Route("go")
        node.params = [Parameter("x")]
        flow.wiring[node.id] = {"x": [1]}
    engine, layout = _owner_engine(flow)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review failure")
    context, = learning_module.project_owner_contexts(engine.state.trace, layout, "review failure", root, results)
    episode = next(item for item in context.episodes if item.owner_static_path == ("root", node.id))
    assert episode.status is MemoryEpisodeStatus.FAILED and episode.output is None
    assert episode.inputs == (None if failure_kind == "binding" else {"x": [1]})
    assert episode.failure == learning_module.failure_json(engine.state.trace.terminal.failure)
    assert episode.failure["code"].startswith({"binding": "binding.", "body": "node.", "route": "graph."}[failure_kind])
    entries = [entry for entry in context.feedback if entry["execution_seq"] == episode.execution_seq]
    assert [(entry["channel"], entry["value"]) for entry in entries] == [("output", "review failure")]
    assert entries[0]["contribution_context"]["contexts"] == []


def test_owner_context_root_default_is_owned_once_by_accepted_start():
    from agent_composer.nodes.contracts import Parameter
    from agent_composer.nodes.end import EndNode
    from agent_composer.nodes.start import StartNode
    from agent_composer.runtime.backward import compute_backward

    start = StartNode(START_ID, params=[Parameter("topic", has_default=True, default="reef")])
    end = EndNode.record(END_ID, output_names=["answer"])
    engine, layout = _owner_engine(_graph([start, end], [(START_ID, END_ID)],
                                         wiring={END_ID: {"answer": "${input.topic}"}}))
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "keep default")
    context, = learning_module.project_owner_contexts(engine.state.trace, layout, "keep default", root, results)
    assert root.input_feedback == {"topic": ("keep default",)}
    assert not any(entry.get("root") for entry in context.feedback)
    start_inputs = [entry for entry in context.feedback if entry["channel"] == "input"
                    and entry["execution_seq"] == engine.state.trace.root_start_source.execution_seq]
    assert [(entry["target_name"], entry["value"]) for entry in start_inputs] == [("topic", "keep default")]


def test_owner_context_projects_synthetic_source_free_failure_only_to_root_flow():
    from agent_composer.runtime.backward import compute_backward
    from agent_composer.runtime.trace import RetainedTrace, TraceTerminal

    trace = RetainedTrace("synthetic", START_ID, END_ID)
    trace.terminal = TraceTerminal(False, None, None, Failure("graph.terminal_unreachable", "no producer", None))
    root_frame = MemoryFrameRef("root", ("root",), MemoryOwner.FLOW)
    child_frame = MemoryFrameRef("root", ("root", "child"), MemoryOwner.FLOW)
    layout = _layout((root_frame, child_frame))
    root, results = compute_backward(trace, "original text")
    parent, child = learning_module.project_owner_contexts(trace, layout, "original text", root, results)
    assert parent.episodes == child.episodes == () and child.feedback == ()
    assert parent.feedback == ({
        "execution_seq": None, "owner_static_path": None, "occurrence_path": None,
        "execution_scope": None, "root": True, "channel": "self", "target_name": None,
        "value": "original text", "message_index": 0, "contribution_context": {},
    },)


@pytest.mark.parametrize("same_boundary", [False, True])
def test_owner_context_root_fallback_uses_only_actual_source_zero_fields(same_boundary):
    from agent_composer.compile.model import CompiledFlow
    from agent_composer.nodes.base import Route
    from agent_composer.nodes.contracts import Parameter
    from agent_composer.nodes.start import StartNode
    from agent_composer.runtime.backward import compute_backward
    from agent_composer.runtime.engine import FlowEngine

    start = StartNode(START_ID, params=[Parameter("topic"), Parameter("omitted", has_default=True, default="unused")])
    start.forward = lambda _inputs: Route("default")
    frame = MemoryFrameRef("root", ("root",), MemoryOwner.FLOW)
    start.memory_scope = MemoryScope("root", ("root", START_ID), (frame,))
    flow = CompiledFlow({START_ID: start}, [])
    flow.end_id = START_ID if same_boundary else END_ID
    pool = VariablePool(start_id=START_ID)
    pool.set(START_ID, {"topic": "reef", "undeclared": "extra"})
    if not same_boundary:
        pool.set(END_ID, "preseeded result")
    pool.add_system("run_id", _RUN_ID)
    state, log = RunState(flow), MemoryRunLog(_RUN_ID)
    layout = _layout((frame,))
    commit_prepared(log, state, prepare_run_started(state, RunStartData(
        _RUN_ID, _FINGERPRINT, pool, {}, {}, 0, learnable=True,
        memory_layout=layout, memory_snapshots=(MemorySnapshotEnvelope(_POLICY, {}),),
    )))
    list(FlowEngine(state, log, grow_builder=build_grow, num_workers=0).run())
    root, results = compute_backward(state.trace, "review")
    context, = learning_module.project_owner_contexts(state.trace, layout, "review", root, results)
    assert all(not item.invoked for item in results)
    assert len(context.episodes) == 1
    assert [(entry["target_name"], entry["value"]) for entry in context.feedback] == (
        [("topic", "review")] if same_boundary else []
    )
    assert all(entry["root"] and entry["execution_seq"] is None for entry in context.feedback)


@pytest.mark.parametrize("material", ["episode", "contribution"])
def test_owner_context_error_identifies_receiver_without_changing_forward_acceptance(material):
    from agent_composer.compile.model import FlowOutput
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode

    value = "leaf"
    for _ in range(140 if material == "episode" else 123):
        value = [value]
    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": value})
    engine, layout = _owner_engine(_graph([node], [(START_ID, "work")],
                                         outputs=[FlowOutput("answer", "${work.output}")]))
    list(engine.run())
    trace = engine.state.trace
    assert trace.terminal.succeeded and trace.terminal.output == value
    feedback = None if material == "episode" else "review"
    root, results = compute_backward(trace, feedback)
    accepted = tuple(engine.log)
    if material == "contribution":
        empty_root, empty_results = compute_backward(trace, None)
        assert learning_module.project_owner_contexts(trace, layout, None, empty_root, empty_results)
    with pytest.raises(learning_module.OwnerContextError) as caught:
        learning_module.project_owner_contexts(trace, layout, feedback, root, results)
    assert caught.value.frame_ref == layout[0].ref
    assert type(caught.value.original) is ValueError and caught.value.__cause__ is caught.value.original
    assert "nesting" in str(caught.value.original)
    assert tuple(engine.log) == accepted and trace.terminal.output == value


def test_owner_context_authority_errors_are_not_context_construction_failures(monkeypatch):
    from agent_composer.compile.model import FlowOutput
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode

    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": "${input.topic}"})
    engine, layout = _owner_engine(_graph([node], [(START_ID, "work")],
                                         outputs=[FlowOutput("answer", "${work.output}")]), {"topic": "reef"})
    list(engine.run())
    trace = engine.state.trace
    root, results = compute_backward(trace, "review")
    with pytest.raises(ValueError, match="complete accepted occurrence order"):
        learning_module.project_owner_contexts(trace, layout, "review", root, results[:-1])
    with pytest.raises(ValueError, match="complete recorded layout"):
        learning_module.project_owner_contexts(trace, (), "review", root, results)
    for invalid_layout in (None, list(layout), tuple(reversed((*layout, *layout)))):
        with pytest.raises((TypeError, ValueError)):
            learning_module.project_owner_contexts(trace, invalid_layout, "review", root, results)
    occurrence = trace.occurrences[1]
    forged = copy.deepcopy(occurrence.memory_scope)
    object.__setattr__(forged, "frame_stack", list(forged.frame_stack))
    object.__setattr__(occurrence, "_memory_scope", forged)
    with pytest.raises(TypeError, match="frame_stack"):
        learning_module.project_owner_contexts(trace, layout, "review", root, results)
    object.__setattr__(occurrence, "_memory_scope", engine.state.flow.nodes[occurrence.node_id].memory_scope)
    original_lookup = trace.lookup_source

    def broken_lookup(source):
        if source.path:
            raise KeyError("missing source authority")
        return original_lookup(source)

    monkeypatch.setattr(trace, "lookup_source", broken_lookup)
    with pytest.raises(KeyError, match="missing source authority"):
        learning_module.project_owner_contexts(trace, layout, "review", root, results)


def test_owner_context_projection_from_replay_does_not_reinvoke_handlers(monkeypatch):
    from agent_composer.compile.model import FlowOutput
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode

    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": "${input.topic}"})
    engine, layout = _owner_engine(_graph([node], [(START_ID, "work")],
                                         outputs=[FlowOutput("answer", "${work.output}")]), {"topic": "reef"})
    original = copy.deepcopy(engine.state.flow)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review")
    expected = learning_module.project_owner_contexts(engine.state.trace, layout, "review", root, results)
    for current in original.nodes.values():
        current.forward = lambda *_: pytest.fail("replayed forward")
        current.backward = lambda *_: pytest.fail("replayed backward")
        current.capture_backward_context = lambda *_: pytest.fail("recaptured backward")
    replayed = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    monkeypatch.setattr(type(replayed.trace), "backward_handlers", property(lambda _: pytest.fail("read callback")))
    assert learning_module.project_owner_contexts(replayed.trace, layout, "review", root, results) == expected


def test_owner_context_empty_map_retains_growth_and_unexecuted_child_frame():
    from agent_composer.compile.model import FlowOutput
    from agent_composer.memory import MemoryPolicySpec
    from agent_composer.nodes.contracts import Parameter
    from agent_composer.runtime.backward import compute_backward

    child = _graph([FuncNode("work", lambda inputs: inputs["x"], reads={"x": "${input.x}"})],
                   [(START_ID, "work")], outputs=[FlowOutput("answer", "${work.output}")])
    child.memory_policy = MemoryPolicySpec(_POLICY, {})
    mapper = MapNode("map", flow_id="child", child=child)
    mapper.params = [Parameter("x")]
    engine, layout = _owner_engine(_graph([mapper], [(START_ID, "map")],
                                         wiring={"map": {"over": [], "x": "${item}"}},
                                         outputs=[FlowOutput("answer", "${map.output}")]))
    list(engine.run())
    root, results = compute_backward(engine.state.trace, None)
    outer, child_context = learning_module.project_owner_contexts(engine.state.trace, layout, None, root, results)
    assert len(outer.episodes) == len(engine.state.trace.occurrences) == 5
    assert outer.feedback == ()
    assert child_context.episodes == child_context.feedback == ()
    grown, = [episode for episode in outer.episodes if episode.output == {"kind": "grow", "seed": [], "prune_ids": []}]
    assert grown.owner_static_path == ("root", "map")


@pytest.mark.parametrize("scope_available", [False, True])
def test_owner_feedback_uses_recorded_execution_coordinates_independently_of_memory(scope_available):
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.nodes.contracts import ExecutionPosition, ExecutionScope
    from agent_composer.runtime.backward import compute_backward

    node = _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    node.execution_scope = (ExecutionScope(
        ("authored", "worker"), ("authored",), (ExecutionPosition("map_item", ("authored",), 3),),
    ) if scope_available else None)
    node.backward = lambda *_: BackwardResult(self_feedback=("local",))
    layout = _layout(node.memory_scope.frame_stack)
    state, log = _start_trace_node(node, layout=layout, learnable=True)
    commit_prepared(log, state, prepare_node_failed(
        state, Failure("node.execute", "failed", node.id), evidence=_evidence(node.memory_scope),
        forward_capture=_forward_capture(node, {"question": "q"}),
    ))
    occurrence = next(item for item in state.trace.occurrences if item.node_id == "agent")
    recorded_scope = occurrence.record.forward_capture.scope
    assert recorded_scope == node.execution_scope
    node.execution_scope = ExecutionScope(("changed",))
    root, results = compute_backward(state.trace, "review")
    contexts = learning_module.project_owner_contexts(state.trace, layout, "review", root, results)
    for context in contexts:
        for entry in context.feedback:
            assert entry["owner_static_path"] == (["authored", "worker"] if scope_available else None)
            assert entry["occurrence_path"] == ([{"kind": "map_item", "static_path": ["authored"], "index": 3}]
                                                 if scope_available else None)
            if scope_available:
                assert entry["owner_static_path"] == entry["execution_scope"]["static_path"]
                assert entry["occurrence_path"] == entry["execution_scope"]["occurrence_path"]
            else:
                assert entry["execution_scope"] is None
    for context in contexts:
        episode, = context.episodes
        assert episode.inputs == {"question": "q"}
        assert episode.owner_static_path == ("root", "flow", "agent")
        assert episode.occurrence.occurrence_path == _occurrence()
        assert episode.interactions == (ModelTextInteraction("answer"),)
        assert episode.base_system_prompt == "system" and episode.prompt == "prompt"


@pytest.mark.parametrize("reverse_targets", [False, True])
def test_owner_variable_batches_follow_captured_inventory_across_wire_round_trip(reverse_targets):
    from agent_composer.compile.model import FlowOutput
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.nodes.contracts import Parameter
    from agent_composer.runtime.backward import compute_backward
    from agent_composer.runtime.log import BackwardReadyRecord, RecordEnvelope, SCHEMA_VERSION, decode_record, encode_record
    from tests.engine._backward_fakes import RecordingBackwardNode

    batches = {"zeta": ("z2", "z1"), "alpha": ("a2", "a1")}
    if reverse_targets:
        batches = dict(reversed(tuple(batches.items())))
    node = RecordingBackwardNode("work", lambda _: "answer", lambda *_: BackwardResult(
        self_feedback=("second", "first"), variable_feedback=batches,
    ))
    node.declared_variables = lambda: (Parameter("zeta"), Parameter("alpha"))
    engine, layout = _owner_engine(_graph([node], [(START_ID, "work")],
                                         outputs=[FlowOutput("answer", "${work.output}")]))
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review")
    node.declared_variables = lambda: pytest.fail("read mutable declarations")
    contexts = learning_module.project_owner_contexts(engine.state.trace, layout, "review", root, results)
    assert [(entry["target_name"], entry["value"], entry["message_index"])
            for entry in contexts[0].feedback if entry["channel"] == "variable"] == [
        ("zeta", "z2", 0), ("zeta", "z1", 1), ("alpha", "a2", 0), ("alpha", "a1", 1),
    ]
    assert [(entry["value"], entry["message_index"]) for entry in contexts[0].feedback if entry["channel"] == "self"] == [
        ("second", 0), ("first", 1),
    ]
    assert all(entry["contribution_context"] == {} for entry in contexts[0].feedback
               if entry["channel"] in ("self", "variable"))
    ready = BackwardReadyRecord(1, root, results, ())
    decoded = decode_record(encode_record(RecordEnvelope(SCHEMA_VERSION, engine.state.run_id, 0, ready))).record
    assert learning_module.project_owner_contexts(engine.state.trace, layout, "review", decoded.root_result,
                                                 decoded.attributed_results) == contexts


def _new_state(
    node: Node,
    *,
    layout: tuple[MemoryFrameSpec, ...] | None,
    learnable: bool,
    route_handle: str | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Initialize a direct reducer authority and leave START ready."""
    flow = _graph([node], [(START_ID, node.id), (node.id, END_ID, route_handle)])
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
                learnable=learnable,
            ),
        ),
    )
    return state, log


def _start_trace_node(
    node: Node,
    *,
    layout: tuple[MemoryFrameSpec, ...] | None,
    learnable: bool,
    route_handle: str | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Drive START and leave ``node`` as the sole in-flight outcome."""
    state, log = _new_state(node, layout=layout, learnable=learnable, route_handle=route_handle)
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(
        state, START_ID, {},
        forward_capture=_forward_capture(state.flow.nodes[START_ID], {}) if learnable else None,
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
    state, log = _start_trace_node(node, layout=layout, learnable=True)
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
    assert state.learnable is True
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


@pytest.mark.parametrize("kind", ["success", "route", "failure", "growth"])
def test_engine_episodes_use_the_physical_start_sequence_for_every_outcome(kind):
    from agent_composer.runtime.state import prepare_node_routed
    from tests.engine.test_run_state_growth import _fragment

    class Spawner(_TraceNode):
        is_spawner = True

    node = Spawner("agent") if kind == "growth" else _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    state, log = _start_trace_node(node, layout=_layout(node.memory_scope.frame_stack), learnable=True,
                                   route_handle="go" if kind == "route" else None)
    physical_sequence = state.trace.started[node.id]
    outcome_sequence = state.next_sequence
    kwargs = {"evidence": _evidence(node.memory_scope),
              "forward_capture": _forward_capture(node, {"question": "q"})}
    if kind == "success":
        prepared = prepare_node_succeeded(state, node.id, {"answer": 7}, **kwargs)
    elif kind == "route":
        prepared = prepare_node_routed(state, node.id, "go", **kwargs)
    elif kind == "failure":
        prepared = prepare_node_failed(state, Failure("node.execute", "Failed.", node.id), **kwargs)
    else:
        prepared = prepare_expanded(state, node.id, Grow(seed={}), lambda spawner, _seed: _fragment(spawner.id), **kwargs)
    commit_prepared(log, state, prepared)
    assert physical_sequence != outcome_sequence
    for frame_ref in node.memory_scope.frame_stack:
        episode = state.learning_state.episodes_for(frame_ref)[-1]
        assert episode.execution_seq == physical_sequence
    assert state.trace.occurrences[-1].source.execution_seq == physical_sequence


def test_causal_route_episode_retains_the_accepted_failure():
    from agent_composer.memory import MemoryEpisodeStatus

    scope = _scope()
    failure = Failure("graph.terminal_unreachable", "No terminal route.", "agent")
    record = NodeRoutedRecord("agent", "go", evidence=_evidence(scope), terminal_failure=failure)
    for _frame_ref, episode in project_outcome_episodes(_RUN_ID, record):
        assert episode.status is MemoryEpisodeStatus.FAILED
        assert episode.output is None
        assert episode.failure["code"] == failure.code


def test_failure_episode_uses_full_canonical_durable_failure_json() -> None:
    agent = _refs()[1]
    node = _TraceNode("agent")
    scope = _scope(node.id, frames=(agent,))
    node.memory_scope = scope
    layout = _layout((agent,))
    state, _log = _start_trace_node(node, layout=layout, learnable=True)
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


@pytest.mark.parametrize("enabled,learnable", [(False, False), (True, False)])
def test_evidence_is_rejected_when_learning_is_not_enabled(
    enabled: bool,
    learnable: bool,
) -> None:
    node = _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    layout = _layout(node.memory_scope.frame_stack) if enabled else None
    state, _log = _start_trace_node(node, layout=layout, learnable=learnable)

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
    state, _log = _start_trace_node(node, layout=layout, learnable=True)
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
        learnable=True,
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
        learnable=True,
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
    state, _log = _start_trace_node(node, layout=(), learnable=True)

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
        learnable=True,
    )

    with pytest.raises(ValueError, match="exact recorded frame stack"):
        prepare_node_succeeded(
            state, node.id, "done", forward_capture=_forward_capture(node, {}),
        )


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
        learnable=True,
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
        assert learning.episodes_for(ref) == ()

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
    state, log = _start_trace_node(node, layout=_layout(refs), learnable=True)

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


def test_commit_id_has_one_documented_canonical_encoding() -> None:
    ref = MemoryFrameRef("root", ("root", "agent"), MemoryOwner.AGENT)
    mutations = (
        MemoryMutation("m", ref, {"z": 1, "a": ["é", None]}),
    )

    assert memory_commit_id("run", "policy", mutations) == (
        "ac-memory-v1:"
        "6c8c1507c9c8c0d6ca4a6239ec397ca950fae2f206146db324ce22098e22f5d6"
    )


def test_commit_id_with_an_attempt_uses_a_separate_versioned_encoding() -> None:
    ref = MemoryFrameRef("root", ("root", "agent"), MemoryOwner.AGENT)
    mutations = (
        MemoryMutation("m", ref, {"z": 1, "a": ["é", None]}),
    )

    assert memory_commit_id("run", "policy", mutations, attempt_id=None) == (
        memory_commit_id("run", "policy", mutations)
    )
    assert memory_commit_id("run", "policy", mutations, attempt_id=2) == (
        "ac-memory-v2:"
        "8d3dd7eb8bc36b1eacff069eea6374b5955dba8ff1787291f4a3e71e3271433a"
    )
    assert memory_commit_id("run", "policy", (), attempt_id=2) != (
        memory_commit_id("run", "policy", (), attempt_id=3)
    )
    for invalid in (0, -1, True, "2", 2.0):
        with pytest.raises(ValueError, match="attempt_id"):
            memory_commit_id("run", "policy", mutations, attempt_id=invalid)
