"""Authoritative runtime projection and reducer transitions for memory learning."""

from __future__ import annotations

import copy

import pytest

import yian.runtime.learning as learning_module
import yian.runtime.state as state_module
from yian.compile.expand import build_grow
from yian.compile.model import END_ID, START_ID
from yian.diagnostics import Failure, OriginRef
from yian.events import RunFailed, RunSucceeded
from yian.memory import (
    DeclSite,
    EvaluationEvidence,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryScope,
    MemorySnapshotEnvelope,
    MemorySpec,
    MemoryStack,
    ModelTextInteraction,
)
from yian.nodes.agent import AgentNode
from yian.nodes.base import Grow, Node, NodeKind
from yian.nodes.map import MapNode
from yian.runtime.learning import _episode_outcome, memory_commit_id
from yian.runtime.log import (
    MemoryRunLog,
    NodeExpandedRecord,
    NodeRoutedRecord,
)
from yian.runtime.state import (
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
from yian.typesys import VariablePool
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _forward_capture, _graph


def test_owner_contexts_include_general_occurrences_and_empty_memories(monkeypatch):
    from yian.compile.model import FlowOutput
    from yian.nodes.base import BackwardResult
    from yian.nodes.contracts import Parameter
    from yian.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode
    from tests.engine._graph_builder import _engine

    memories = (
        MemorySpec("empty", "empty", {}, (DeclSite((), ("def", "unused"), None),), False),
        MemorySpec("local", "local", {}, (DeclSite((), ("root",), "work"),), True),
        MemorySpec("outer", "outer", {}, (_ROOT,), False),
    )
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
        keys = ("outer", "local") if current is node else ("outer",)
        current.memory_scope = MemoryScope((current.id,), keys)
    stacks = _stacks(*(current.memory_scope for current in flow.nodes.values()))
    engine = _engine(flow, {"topic": "reefs"}, learnable=True, memories=memories, memory_stacks=stacks,
                     memory_snapshots=tuple(MemorySnapshotEnvelope(spec.key, {}) for spec in memories))
    list(engine.run())
    trace = engine.state.trace
    root, results = compute_backward(trace, "review")
    monkeypatch.setattr(node, "callback", lambda *_: pytest.fail("projection invoked backward"))

    contexts = learning_module.project_owner_contexts(trace, memories, stacks, "review", root, results)

    assert tuple(context.memory for context in contexts) == ("empty", "local", "outer")
    by_key = {context.memory: context for context in contexts}
    assert len(by_key["outer"].episodes) == len(trace.occurrences) == 3
    assert by_key["empty"].episodes == () and by_key["empty"].feedback == ()
    local_episode, = by_key["local"].episodes
    assert local_episode.inputs == {"topic": "reefs", "literal": {"default": [1]}}
    assert local_episode.interactions == ()
    inherited = next(episode for episode in by_key["outer"].episodes
                     if episode.execution_seq == local_episode.execution_seq)
    assert inherited.owner_static_path == local_episode.owner_static_path == ("work",)
    assert inherited.occurrence_path == local_episode.occurrence_path == ()
    entries = by_key["local"].feedback
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
    assert by_key["local"].feedback[0]["value"] == "review"
    assert by_key["local"].episodes[0].inputs["literal"] == {"default": [1]}


_RUN_ID = "learning-run"
_FINGERPRINT = "a" * 64
_POLICY = "reflective"


class _TraceNode(Node):
    """Trait-capable AGENT-kind node that evaluates without calling a model."""

    kind = NodeKind.AGENT
    needs_evaluation_trace = True

    def forward(self, inputs, **_caps):
        return inputs


_ROOT = DeclSite((), ("root",), None)


def _keys() -> tuple[str, ...]:
    """Return one outer and one local memory key in stack order."""
    return ("outer", "local")


def _occurrence(index: int = 2) -> tuple[MemoryOccurrenceSegment, ...]:
    """Return one canonical dynamic MAP occurrence."""
    return (MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("flow",), index),)


def _scope(
    node_id: str = "agent",
    *,
    memories: tuple[str, ...] | None = None,
    occurrence: tuple[MemoryOccurrenceSegment, ...] | None = None,
) -> MemoryScope:
    """Return one exact trace scope for a test node."""
    return MemoryScope(
        ("flow", node_id),
        _keys() if memories is None else memories,
        _occurrence() if occurrence is None else occurrence,
    )


def _table(keys: tuple[str, ...]) -> tuple[MemorySpec, ...]:
    """Return the sorted key table for ``keys``, all owned by the root flow."""
    return tuple(MemorySpec(key, key, {"limit": 3}, (_ROOT,), False) for key in sorted(keys))


def _stacks(*scopes: MemoryScope) -> tuple[MemoryStack, ...]:
    """Return the recorded stacks for ``scopes``, deduplicated and sorted by locator."""
    stacks = {scope.static_path: MemoryStack(scope.static_path, scope.memories) for scope in scopes}
    return tuple(stacks[locator] for locator in sorted(stacks))


def _owner_engine(flow, inputs=None, **kwargs):
    """Use the real static scope pass with the existing direct engine fixture."""
    from yian.compile.llm_cascade import apply_effective_llm_map
    from yian.compile.memory_layout import apply_memory_layout, resolve_memory_layout
    from yian.memory import MemoryDecl
    from tests.engine._graph_builder import _engine, memory_record

    flow.flow_id = flow.flow_id or "root"
    flow.memories = (MemoryDecl(_POLICY, _POLICY, {}, _ROOT),)
    apply_effective_llm_map(flow, {})
    apply_memory_layout(flow, resolve_memory_layout(flow))
    memories, stacks = memory_record(flow)
    engine = _engine(
        flow, inputs, learnable=True, memories=memories, memory_stacks=stacks,
        memory_snapshots=tuple(MemorySnapshotEnvelope(spec.key, {}) for spec in memories),
        **kwargs,
    )
    return engine, memories, stacks


@pytest.mark.parametrize("opaque", [False, True])
def test_owner_feedback_retains_literals_and_opaque_local_stops(opaque, monkeypatch):
    from yian.compile.model import FlowOutput
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.nodes.base import BackwardResult
    from yian.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode

    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "owner_identity", lambda value: calls.append(value) or value)
    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": "${owner_identity(input.topic)}" if opaque else "literal"})
    flow = _graph([node], [(START_ID, "work")], outputs=[FlowOutput("answer", "${work.output}")])
    engine, memories, stacks = _owner_engine(flow, {"topic": "reef"})
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "unchanged")
    contexts = learning_module.project_owner_contexts(engine.state.trace, memories, stacks, "unchanged", root, results)
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
    from yian.compile.model import FlowOutput
    from yian.memory import MemoryEpisodeStatus
    from yian.nodes.base import Route
    from yian.nodes.contracts import Parameter
    from yian.runtime.backward import compute_backward

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
    engine, memories, stacks = _owner_engine(flow)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review failure")
    context, = learning_module.project_owner_contexts(engine.state.trace, memories, stacks, "review failure", root, results)
    episode = next(item for item in context.episodes if item.owner_static_path == (node.id,))
    assert episode.status is MemoryEpisodeStatus.FAILED and episode.output is None
    assert episode.inputs == (None if failure_kind == "binding" else {"x": [1]})
    assert episode.failure == learning_module.failure_json(engine.state.trace.terminal.failure)
    assert episode.failure["code"].startswith({"binding": "binding.", "body": "node.", "route": "graph."}[failure_kind])
    entries = [entry for entry in context.feedback if entry["execution_seq"] == episode.execution_seq]
    assert [(entry["channel"], entry["value"]) for entry in entries] == [("output", "review failure")]
    assert entries[0]["contribution_context"]["contexts"] == []


def test_owner_context_root_default_is_owned_once_by_accepted_start():
    from yian.nodes.contracts import Parameter
    from yian.nodes.end import EndNode
    from yian.nodes.start import StartNode
    from yian.runtime.backward import compute_backward

    start = StartNode(START_ID, params=[Parameter("topic", has_default=True, default="reef")])
    end = EndNode.value(END_ID, output_name="answer")
    engine, memories, stacks = _owner_engine(_graph([start, end], [(START_ID, END_ID)],
                                                  wiring={END_ID: {"answer": "${input.topic}"}}))
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "keep default")
    context, = learning_module.project_owner_contexts(engine.state.trace, memories, stacks, "keep default", root, results)
    assert root.input_feedback == {"topic": ("keep default",)}
    assert not any(entry.get("root") for entry in context.feedback)
    start_inputs = [entry for entry in context.feedback if entry["channel"] == "input"
                    and entry["execution_seq"] == engine.state.trace.root_start_source.execution_seq]
    assert [(entry["target_name"], entry["value"]) for entry in start_inputs] == [("topic", "keep default")]


def test_owner_context_projects_synthetic_source_free_failure_only_to_root_flow():
    from yian.runtime.backward import compute_backward
    from yian.runtime.trace import RetainedTrace, TraceTerminal

    trace = RetainedTrace("synthetic", START_ID, END_ID)
    trace.terminal = TraceTerminal(False, None, None, Failure("graph.terminal_unreachable", "no producer", None))
    memories = (
        MemorySpec("inner", "inner", {}, (DeclSite((), ("def", "child"), None),), False),
        MemorySpec("outer", "outer", {}, (_ROOT,), False),
    )
    root, results = compute_backward(trace, "original text")
    inner, outer = learning_module.project_owner_contexts(trace, memories, (), "original text", root, results)
    assert outer.episodes == inner.episodes == () and inner.feedback == ()
    assert outer.feedback == ({
        "execution_seq": None, "owner_static_path": None, "occurrence_path": None,
        "execution_scope": None, "root": True, "channel": "self", "target_name": None,
        "value": "original text", "message_index": 0, "contribution_context": {},
    },)


@pytest.mark.parametrize("same_boundary", [False, True])
def test_owner_context_root_fallback_uses_only_actual_source_zero_fields(same_boundary):
    from yian.compile.model import CompiledFlow
    from yian.nodes.base import ELSE_HANDLE, Route
    from yian.nodes.contracts import Parameter
    from yian.nodes.start import StartNode
    from yian.runtime.backward import compute_backward
    from yian.runtime.engine import FlowEngine

    start = StartNode(START_ID, params=[Parameter("topic"), Parameter("omitted", has_default=True, default="unused")])
    start.forward = lambda _inputs: Route(ELSE_HANDLE)
    start.memory_scope = MemoryScope((START_ID,), (_POLICY,))
    flow = CompiledFlow({START_ID: start}, [])
    flow.end_id = START_ID if same_boundary else END_ID
    pool = VariablePool(start_id=START_ID)
    pool.set(START_ID, {"topic": "reef", "undeclared": "extra"})
    if not same_boundary:
        pool.set(END_ID, "preseeded result")
    pool.add_system("run_id", _RUN_ID)
    state, log = RunState(flow), MemoryRunLog(_RUN_ID)
    memories, stacks = _table((_POLICY,)), _stacks(start.memory_scope)
    commit_prepared(log, state, prepare_run_started(state, RunStartData(
        run_id=_RUN_ID, flow_fingerprint=_FINGERPRINT, initial_pool=pool,
        start_wiring_override={}, effective_llm_by_static_path={}, num_workers=0,
        learnable=True, memories=memories, memory_stacks=stacks,
        memory_snapshots=(MemorySnapshotEnvelope(_POLICY, {}),),
    )))
    list(FlowEngine(state, log, grow_builder=build_grow, num_workers=0).run())
    root, results = compute_backward(state.trace, "review")
    context, = learning_module.project_owner_contexts(state.trace, memories, stacks, "review", root, results)
    assert all(not item.invoked for item in results)
    assert len(context.episodes) == 1
    assert [(entry["target_name"], entry["value"]) for entry in context.feedback] == (
        [("topic", "review")] if same_boundary else []
    )
    assert all(entry["root"] and entry["execution_seq"] is None for entry in context.feedback)


@pytest.mark.parametrize("material", ["episode", "contribution"])
def test_owner_context_error_identifies_receiver_without_changing_forward_acceptance(material, monkeypatch):
    from yian.compile.model import FlowOutput
    from yian.nodes.base import BackwardResult
    from yian.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode

    # The limit is raised so the live run produces a value deep enough to exceed the
    # memory learn-context bound, which is what this test exercises.
    monkeypatch.setattr("yian.typesys.values.MAX_VALUE_NESTING", 500)
    value = "leaf"
    for _ in range(140 if material == "episode" else 123):
        value = [value]
    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": value})
    engine, memories, stacks = _owner_engine(_graph([node], [(START_ID, "work")],
                                                  outputs=[FlowOutput("answer", "${work.output}")]))
    list(engine.run())
    trace = engine.state.trace
    assert trace.terminal.succeeded and trace.terminal.output == value
    feedback = None if material == "episode" else "review"
    root, results = compute_backward(trace, feedback)
    accepted = tuple(engine.log)
    if material == "contribution":
        empty_root, empty_results = compute_backward(trace, None)
        assert learning_module.project_owner_contexts(trace, memories, stacks, None, empty_root, empty_results)
    with pytest.raises(learning_module.OwnerContextError) as caught:
        learning_module.project_owner_contexts(trace, memories, stacks, feedback, root, results)
    assert caught.value.memory == memories[0].key
    assert type(caught.value.original) is ValueError and caught.value.__cause__ is caught.value.original
    assert "nesting" in str(caught.value.original)
    assert tuple(engine.log) == accepted and trace.terminal.output == value


def test_owner_context_authority_errors_are_not_context_construction_failures(monkeypatch):
    from yian.compile.model import FlowOutput
    from yian.nodes.base import BackwardResult
    from yian.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode

    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": "${input.topic}"})
    engine, memories, stacks = _owner_engine(_graph([node], [(START_ID, "work")],
                                                  outputs=[FlowOutput("answer", "${work.output}")]), {"topic": "reef"})
    list(engine.run())
    trace = engine.state.trace
    root, results = compute_backward(trace, "review")
    with pytest.raises(ValueError, match="complete accepted occurrence order"):
        learning_module.project_owner_contexts(trace, memories, stacks, "review", root, results[:-1])
    with pytest.raises(ValueError, match="recorded memory stack"):
        learning_module.project_owner_contexts(trace, (), (), "review", root, results)
    for invalid_table in (None, list(memories), tuple(reversed((*memories, *memories)))):
        with pytest.raises((TypeError, ValueError)):
            learning_module.project_owner_contexts(trace, invalid_table, stacks, "review", root, results)
    with pytest.raises(TypeError, match="memory stacks"):
        learning_module.project_owner_contexts(trace, memories, list(stacks), "review", root, results)
    with pytest.raises(ValueError, match="unique locators"):
        learning_module.project_owner_contexts(trace, memories, (*stacks, stacks[0]), "review", root, results)
    occurrence = trace.occurrences[1]
    forged = copy.deepcopy(occurrence.memory_scope)
    object.__setattr__(forged, "memories", list(forged.memories))
    object.__setattr__(occurrence, "_memory_scope", forged)
    with pytest.raises(TypeError, match="memories"):
        learning_module.project_owner_contexts(trace, memories, stacks, "review", root, results)
    object.__setattr__(forged, "memories", ())
    with pytest.raises(ValueError, match="recorded memory stack"):
        learning_module.project_owner_contexts(trace, memories, stacks, "review", root, results)
    object.__setattr__(occurrence, "_memory_scope", engine.state.flow.nodes[occurrence.node_id].memory_scope)
    original_lookup = trace.lookup_source

    def broken_lookup(source):
        if source.path:
            raise KeyError("missing source authority")
        return original_lookup(source)

    monkeypatch.setattr(trace, "lookup_source", broken_lookup)
    with pytest.raises(KeyError, match="missing source authority"):
        learning_module.project_owner_contexts(trace, memories, stacks, "review", root, results)


def test_owner_context_projection_from_replay_does_not_reinvoke_handlers(monkeypatch):
    from yian.compile.model import FlowOutput
    from yian.nodes.base import BackwardResult
    from yian.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode

    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": "${input.topic}"})
    engine, memories, stacks = _owner_engine(_graph([node], [(START_ID, "work")],
                                                  outputs=[FlowOutput("answer", "${work.output}")]), {"topic": "reef"})
    original = copy.deepcopy(engine.state.flow)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review")
    expected = learning_module.project_owner_contexts(engine.state.trace, memories, stacks, "review", root, results)
    for current in original.nodes.values():
        current.forward = lambda *_: pytest.fail("replayed forward")
        current.backward = lambda *_: pytest.fail("replayed backward")
        current.capture_backward_context = lambda *_: pytest.fail("recaptured backward")
    replayed = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    monkeypatch.setattr(type(replayed.trace), "backward_handlers", property(lambda _: pytest.fail("read callback")))
    assert learning_module.project_owner_contexts(replayed.trace, memories, stacks, "review", root, results) == expected


def test_owner_context_empty_map_retains_growth_and_unexecuted_child_memory():
    from yian.compile.model import FlowOutput
    from yian.memory import DeclSite, MemoryDecl
    from yian.nodes.contracts import Parameter
    from yian.runtime.backward import compute_backward

    child = _graph([FuncNode("work", lambda inputs: inputs["x"], reads={"x": "${input.x}"})],
                   [(START_ID, "work")], outputs=[FlowOutput("answer", "${work.output}")])
    child.memories = (MemoryDecl("child_notes", "child_notes", {}, DeclSite((), ("def", "child"), None)),)
    mapper = MapNode("map", flow_id="child", child=child)
    mapper.params = [Parameter("x")]
    engine, memories, stacks = _owner_engine(_graph([mapper], [(START_ID, "map")],
                                                  wiring={"map": {"over": [], "x": "${item}"}},
                                                  outputs=[FlowOutput("answer", "${map.output}")]))
    list(engine.run())
    root, results = compute_backward(engine.state.trace, None)
    child_context, outer = learning_module.project_owner_contexts(engine.state.trace, memories, stacks, None, root, results)
    assert len(outer.episodes) == len(engine.state.trace.occurrences) == 5
    assert outer.feedback == ()
    assert child_context.episodes == child_context.feedback == ()
    grown, = [episode for episode in outer.episodes if episode.output == {"kind": "grow", "seed": [], "prune_ids": []}]
    assert grown.owner_static_path == ("map",)


@pytest.mark.parametrize("scope_available", [False, True])
def test_owner_feedback_uses_recorded_execution_coordinates_independently_of_memory(scope_available):
    from yian.nodes.base import BackwardResult
    from yian.nodes.contracts import ExecutionPosition, ExecutionScope
    from yian.runtime.backward import compute_backward

    node = _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    node.execution_scope = (ExecutionScope(
        ("authored", "worker"), ("authored",), (ExecutionPosition("map_item", ("authored",), 3),),
    ) if scope_available else None)
    node.backward = lambda *_: BackwardResult(self_feedback=("local",))
    memories, stacks = _table(node.memory_scope.memories), _stacks(node.memory_scope)
    state, log = _start_trace_node(node, keys=node.memory_scope.memories, learnable=True)
    commit_prepared(log, state, prepare_node_failed(
        state, Failure("node.execute", "failed", node.id), evidence=_evidence(node.memory_scope),
        forward_capture=_forward_capture(node, {"question": "q"}),
    ))
    occurrence = next(item for item in state.trace.occurrences if item.node_id == "agent")
    recorded_scope = occurrence.record.forward_capture.scope
    assert recorded_scope == node.execution_scope
    node.execution_scope = ExecutionScope(("changed",))
    root, results = compute_backward(state.trace, "review")
    contexts = learning_module.project_owner_contexts(state.trace, memories, stacks, "review", root, results)
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
        assert episode.owner_static_path == ("flow", "agent")
        assert episode.occurrence_path == _occurrence()
        assert episode.interactions == (ModelTextInteraction("answer"),)
        assert episode.base_system_prompt == "system" and episode.prompt == "prompt"


@pytest.mark.parametrize("reverse_targets", [False, True])
def test_owner_variable_batches_follow_captured_inventory_across_wire_round_trip(reverse_targets):
    from yian.compile.model import FlowOutput
    from yian.nodes.base import BackwardResult
    from yian.nodes.contracts import Parameter
    from yian.runtime.backward import compute_backward
    from yian.runtime.log import BackwardReadyRecord, RecordEnvelope, SCHEMA_VERSION, decode_record, encode_record
    from tests.engine._backward_fakes import RecordingBackwardNode

    batches = {"zeta": ("z2", "z1"), "alpha": ("a2", "a1")}
    if reverse_targets:
        batches = dict(reversed(tuple(batches.items())))
    node = RecordingBackwardNode("work", lambda _: "answer", lambda *_: BackwardResult(
        self_feedback=("second", "first"), variable_feedback=batches,
    ))
    node.declared_variables = lambda: (Parameter("zeta"), Parameter("alpha"))
    engine, memories, stacks = _owner_engine(_graph([node], [(START_ID, "work")],
                                                  outputs=[FlowOutput("answer", "${work.output}")]))
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review")
    node.declared_variables = lambda: pytest.fail("read mutable declarations")
    contexts = learning_module.project_owner_contexts(engine.state.trace, memories, stacks, "review", root, results)
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
    assert learning_module.project_owner_contexts(
        engine.state.trace, memories, stacks, "review", decoded.root_result, decoded.attributed_results,
    ) == contexts


def _new_state(
    node: Node,
    *,
    keys: tuple[str, ...] | None,
    learnable: bool,
    route_handle: str | None = None,
    stacks: tuple[MemoryStack, ...] | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Initialize a direct reducer authority and leave START ready.

    ``keys=None`` disables memory. Otherwise the table holds ``keys`` and the
    stacks default to the node's own scope.
    """
    flow = _graph([node], [(START_ID, node.id), (node.id, END_ID, route_handle)])
    state = RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {})
    pool.add_system("run_id", _RUN_ID)
    if keys is None:
        memories = snapshots = recorded_stacks = None
    else:
        memories = _table(keys)
        snapshots = tuple(MemorySnapshotEnvelope(spec.key, {"epoch": 4}) for spec in memories)
        if stacks is not None:
            recorded_stacks = stacks
        elif node.memory_scope is not None:
            recorded_stacks = _stacks(node.memory_scope)
        else:
            recorded_stacks = ()
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
                memory_snapshots=snapshots,
                learnable=learnable,
                memories=memories,
                memory_stacks=recorded_stacks,
            ),
        ),
    )
    return state, log


def _start_trace_node(
    node: Node,
    *,
    keys: tuple[str, ...] | None,
    learnable: bool,
    route_handle: str | None = None,
    stacks: tuple[MemoryStack, ...] | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Drive START and leave ``node`` as the sole in-flight outcome."""
    state, log = _new_state(
        node, keys=keys, learnable=learnable, route_handle=route_handle, stacks=stacks,
    )
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
    memories: tuple[str, ...] | None = None,
) -> tuple[RunState, MemoryRunLog, tuple[MemorySpec, ...]]:
    """Return a terminal-success run with one accepted trace episode."""
    node = _TraceNode("agent")
    scope = _scope(node.id, memories=memories)
    node.memory_scope = scope
    state, log = _start_trace_node(node, keys=scope.memories, learnable=True)
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
    return state, log, _table(scope.memories)


def _commit_record(state: RunState, log: MemoryRunLog, record) -> None:
    """Prepare and commit one reducer record, asserting no learning observers."""
    prepared = prepare_record(state, record)
    assert prepared.observers == ()
    assert commit_prepared(log, state, prepared) == ()


def _contexts(state: RunState) -> dict:
    """Return the learner contexts of a terminal run's retained trace, keyed by memory."""
    from yian.runtime.backward import compute_backward

    root, results = compute_backward(state.trace, None)
    contexts = learning_module.project_owner_contexts(
        state.trace, state.memories, state.memory_stacks, None, root, results,
    )
    return {context.memory: context for context in contexts}


def test_runstarted_retains_defensive_memory_authority_and_projects_episodes() -> None:
    state, _log, table = _successful_terminal()

    assert state.memories == table
    assert state.memory_snapshots == (
        MemorySnapshotEnvelope("local", {"epoch": 4}),
        MemorySnapshotEnvelope("outer", {"epoch": 4}),
    )
    assert state.learnable is True
    assert state.backward_state.memory_order == ("local", "outer")

    contexts = _contexts(state)
    outer_episode = contexts["outer"].episodes
    local_episode = contexts["local"].episodes
    assert len(outer_episode) == len(local_episode) == 1
    assert outer_episode[0].owner_static_path == ("flow", "agent")
    assert outer_episode[0].occurrence_path == _occurrence()
    assert local_episode[0].occurrence_path == _occurrence()
    assert outer_episode[0].output == {"answer": 7}
    assert outer_episode[0].inputs == {"question": "q"}
    assert outer_episode[0].interactions == (ModelTextInteraction("answer"),)

    detached = state.memories[0].config
    detached["limit"] = 999
    snapshot = state.memory_snapshots[0].snapshot
    snapshot["epoch"] = 999
    assert state.memories[0].config == {"limit": 3}
    assert state.memory_snapshots[0].snapshot == {"epoch": 4}


@pytest.mark.parametrize("kind", ["success", "route", "failure", "growth"])
def test_engine_episodes_use_the_physical_start_sequence_for_every_outcome(kind):
    from yian.runtime.state import prepare_node_routed
    from tests.engine.test_run_state_growth import _fragment

    class Spawner(_TraceNode):
        is_spawner = True

    node = Spawner("agent") if kind == "growth" else _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    state, log = _start_trace_node(node, keys=node.memory_scope.memories, learnable=True,
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
    assert state.trace.occurrences[-1].source.execution_seq == physical_sequence
    if kind == "failure":
        for context in _contexts(state).values():
            assert context.episodes[-1].execution_seq == physical_sequence


def test_causal_route_episode_retains_the_accepted_failure():
    from yian.memory import MemoryEpisodeStatus

    scope = _scope()
    failure = Failure("graph.terminal_unreachable", "No terminal route.", "agent")
    record = NodeRoutedRecord("agent", "go", evidence=_evidence(scope), terminal_failure=failure)
    status, output, failure_payload = _episode_outcome(record)
    assert status is MemoryEpisodeStatus.FAILED
    assert output is None
    assert failure_payload["code"] == failure.code


def test_failure_episode_uses_full_canonical_durable_failure_json() -> None:
    node = _TraceNode("agent")
    scope = _scope(node.id, memories=("local",))
    node.memory_scope = scope
    state, _log = _start_trace_node(node, keys=("local",), learnable=True)
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
    episode, = _contexts(state)["local"].episodes
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
    keys = node.memory_scope.memories if enabled else None
    state, _log = _start_trace_node(node, keys=keys, learnable=learnable)

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
    state, _log = _start_trace_node(node, keys=_keys(), learnable=True)
    with pytest.raises(ValueError, match="requires evaluation evidence"):
        prepare_node_succeeded(
            state, node.id, "done", forward_capture=_forward_capture(node, {}),
        )

    ordinary = FuncNode("ordinary", lambda _inputs: "done")
    ordinary.memory_scope = _scope(ordinary.id)
    ordinary_state, _ = _start_trace_node(ordinary, keys=_keys(), learnable=True)
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

    mismatched = _TraceNode("agent")
    mismatched.memory_scope = _scope(mismatched.id)
    mismatched_state, _ = _start_trace_node(
        mismatched,
        keys=_keys(),
        learnable=True,
        stacks=(MemoryStack(("flow", "agent"), ("outer",)),),
    )
    with pytest.raises(ValueError, match="exact recorded memory stack"):
        prepare_node_succeeded(
            mismatched_state,
            mismatched.id,
            "done",
            evidence=_evidence(mismatched.memory_scope),
            forward_capture=_forward_capture(mismatched, {"question": "q"}),
        )


def test_learning_rejects_a_scope_without_a_recorded_stack() -> None:
    node = _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    state, _log = _start_trace_node(node, keys=_keys(), learnable=True, stacks=())

    with pytest.raises(ValueError, match="has no recorded memory stack"):
        prepare_node_succeeded(
            state,
            node.id,
            "done",
            evidence=_evidence(node.memory_scope),
            forward_capture=_forward_capture(node, {"question": "q"}),
        )


def test_empty_enabled_table_retains_scope_free_evidence_without_episodes() -> None:
    node = _TraceNode("agent")
    state, _log = _start_trace_node(node, keys=(), learnable=True)

    prepared = prepare_node_succeeded(
        state, node.id, "done", evidence=_evidence(None),
        forward_capture=_forward_capture(node, {"question": "q"}),
    )
    assert prepared.record.evidence.scope is None
    with pytest.raises(ValueError, match="evidence scope does not match"):
        prepare_node_succeeded(
            state,
            node.id,
            "done",
            evidence=_evidence(_scope(node.id)),
            forward_capture=_forward_capture(node, {"question": "q"}),
        )


@pytest.mark.parametrize("retained", ["outer", "local"])
def test_learning_rejects_trace_scope_that_omits_a_recorded_memory(
    retained: str,
) -> None:
    node = _TraceNode("agent")
    node.memory_scope = _scope(node.id, memories=(retained,))
    state, _log = _start_trace_node(
        node,
        keys=_keys(),
        learnable=True,
        stacks=_stacks(_scope(node.id)),
    )

    with pytest.raises(ValueError, match="exact recorded memory stack"):
        prepare_node_succeeded(
            state, node.id, "done", forward_capture=_forward_capture(node, {}),
        )


def test_expanded_map_occurrence_projects_and_replays_from_live_node_scope() -> None:
    child_agent = AgentNode("agent", prompt="visible")
    child = _graph(
        [child_agent],
        [(START_ID, child_agent.id), (child_agent.id, END_ID)],
    )
    for node_id, node in child.nodes.items():
        node.memory_scope = MemoryScope(("flow", "map", "child", node_id), ("outer",))
    spawner = MapNode(
        "map",
        flow_id="child",
        child=child,
        child_inputs=[],
    )
    spawner.memory_scope = MemoryScope(("flow", "map"), ("outer",))
    state, log = _start_trace_node(
        spawner,
        keys=("outer",),
        learnable=True,
        stacks=_stacks(
            spawner.memory_scope, *(node.memory_scope for node in child.nodes.values()),
        ),
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
    assert dynamic_scope.static_path == ("flow", "map", "child", "agent")
    succeed(
        dynamic_agent.id,
        "visible",
        evidence=_evidence(dynamic_scope),
    )
    succeed("map#0/__end__", None)
    succeed("map/__end__", [None])
    succeed(END_ID, None)
    assert isinstance(state.result, RunSucceeded)

    episodes = _contexts(state)["outer"].episodes
    episode, = (item for item in episodes if item.owner_static_path == dynamic_scope.static_path)
    assert episode.occurrence_path == (
        MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("flow", "map"), 0),
    )
    assert episode.interactions == (ModelTextInteraction("answer"),)

    replayed = _replay_records(
        RunState(pristine_flow),
        tuple(log),
        grow_builder=build_grow,
    )
    assert _contexts(replayed)["outer"].episodes == episodes


def test_route_and_growth_projection_use_closed_source_neutral_envelopes() -> None:
    _status, route, _failure = _episode_outcome(NodeRoutedRecord("agent", "yes"))
    _status, growth, _failure = _episode_outcome(
        NodeExpandedRecord("agent", {"value": (1, 2)}, ("a", "z"), "b" * 64),
    )

    assert route == {"kind": "route", "handle": "yes"}
    assert growth == {
        "kind": "grow",
        "seed": {"value": [1, 2]},
        "prune_ids": ["a", "z"],
    }


def test_graph_outcome_uses_constant_memory_roots_and_indexed_scope(
    monkeypatch,
) -> None:
    keys = tuple(f"agent-{index:04d}" for index in range(1024))
    node = _TraceNode("agent")
    node.memory_scope = MemoryScope(("agents", "agent-1023"), (keys[-1],))
    stacks = tuple(
        MemoryStack(("agents", key), (key,)) for key in keys
    )
    state, log = _start_trace_node(node, keys=keys, learnable=True, stacks=stacks)

    original_recorded_stack = state_module._recorded_stack
    stack_calls = 0

    def count_recorded_stack(recorded, locator):
        nonlocal stack_calls
        stack_calls += 1
        return original_recorded_stack(recorded, locator)

    monkeypatch.setattr(state_module, "_recorded_stack", count_recorded_stack)

    prepared = prepare_node_succeeded(
        state,
        node.id,
        "done",
        evidence=_evidence(node.memory_scope),
        forward_capture=_forward_capture(node, {"question": "q"}),
    )
    commit_prepared(log, state, prepared)

    assert stack_calls == 1


def test_commit_id_has_one_documented_canonical_encoding() -> None:
    mutations = (MemoryMutation("m", {"z": 1, "a": ["é", None]}),)

    assert memory_commit_id("run", "notes", mutations, 1) == (
        "ac-memory-v3:"
        "250e0a4c357acfd6ef4d37a84a5207bffdb3df2e343816df5e7219401a7bf47b"
    )
    assert memory_commit_id("run", "notes", mutations, 2) == (
        "ac-memory-v3:"
        "1c87e547bc72a30942158a6c0c0bc3dc4fa21245e45564ac7571175e48cdf64a"
    )


def test_commit_id_requires_a_positive_attempt_and_separates_attempts() -> None:
    mutations = (MemoryMutation("m", {"z": 1, "a": ["é", None]}),)

    assert memory_commit_id("run", "notes", (), 2) != memory_commit_id("run", "notes", (), 3)
    for invalid in (None, 0, -1, True, "2", 2.0):
        with pytest.raises(ValueError, match="attempt_id"):
            memory_commit_id("run", "notes", mutations, invalid)
    with pytest.raises(TypeError):
        memory_commit_id("run", "notes", list(mutations), 1)


def test_root_fallback_reaches_only_keys_owned_by_the_root_flow_site():
    from yian.runtime.backward import compute_backward
    from yian.runtime.trace import RetainedTrace, TraceTerminal

    trace = RetainedTrace("synthetic", START_ID, END_ID)
    trace.terminal = TraceTerminal(
        False, None, None, Failure("graph.terminal_unreachable", "no producer", None)
    )
    memories = (
        MemorySpec("agent_notes", "agent_notes", {}, (DeclSite((), ("root",), "agent"),), True),
        MemorySpec("def_notes", "def_notes", {}, (DeclSite((), ("def", "child"), None),), False),
        # An imported file's root flow is not the run's root flow.
        MemorySpec("lib.notes", "lib.notes", {}, (DeclSite(("lib",), ("root",), None),), False),
        MemorySpec("shared", "shared", {}, (DeclSite((), ("def", "child"), None), _ROOT), False),
    )
    root, results = compute_backward(trace, "original text")

    agent_notes, def_notes, imported, shared = learning_module.project_owner_contexts(
        trace, memories, (), "original text", root, results,
    )

    assert agent_notes.feedback == def_notes.feedback == imported.feedback == ()
    assert [entry["value"] for entry in shared.feedback] == ["original text"]
    assert all(entry["root"] for entry in shared.feedback)


def test_replay_rejects_an_eligible_node_scope_that_differs_from_its_recorded_stack():
    node = _TraceNode("agent")
    node.memory_scope = _scope(node.id)
    state, log = _start_trace_node(node, keys=_keys(), learnable=True)
    original = copy.deepcopy(state.flow)
    commit_prepared(log, state, prepare_node_succeeded(
        state, node.id, "done", evidence=_evidence(node.memory_scope),
        forward_capture=_forward_capture(node, {"question": "q"}),
    ))
    replayed = _replay_records(RunState(copy.deepcopy(original)), tuple(log), grow_builder=build_grow)
    assert replayed.next_sequence == state.next_sequence

    original.nodes["agent"].memory_scope = _scope(node.id, memories=("outer",))
    with pytest.raises(ValueError, match="exact recorded memory stack"):
        _replay_records(RunState(original), tuple(log), grow_builder=build_grow)


def test_owner_contexts_are_the_only_episode_projection() -> None:
    state, _log, table = _successful_terminal()

    assert not hasattr(state, "learning_state")
    for name in ("RuntimeLearningState", "project_outcome_episodes", "occurrence_key"):
        assert not hasattr(learning_module, name)
    episode, = _contexts(state)["outer"].episodes
    assert episode.owner_static_path == ("flow", "agent")
    assert state.backward_state.memory_order == tuple(spec.key for spec in table)
