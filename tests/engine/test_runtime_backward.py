"""One accepted reverse walk over retained data routes and boundary projections."""

from copy import deepcopy

import pytest

from yian.compile.model import END_ID, START_ID, FlowOutput
from yian.nodes.base import BackwardResult
from yian.nodes.contracts import Parameter
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.runtime.learning import validate_attributed_backward
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _engine, _graph
from tests.engine._backward_fakes import RecordingBackwardNode

_POLICY = "reflective"


@pytest.mark.parametrize("value", [None, {"answer": ["stable"]}])
def test_terminal_trace_keeps_actual_root_writer_until_quiescence(value):
    observations = []

    def late(_inputs):
        observations.append(engine.state.trace.terminal)
        return "last outcome"

    end = EndNode.value(END_ID, output_name="answer")
    flow = _graph([end, FuncNode("late", late)],
                  [(START_ID, END_ID), (START_ID, "late")],
                  wiring={END_ID: {"answer": value}, "late": {}})
    engine = _engine(flow, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    end_occurrence = next(item for item in trace.occurrences if item.node_id == END_ID)
    assert observations == [None]
    assert trace.occurrences[-1].node_id == "late"
    assert trace.end_id == END_ID
    assert trace.terminal.succeeded is True
    assert trace.terminal.source == end_occurrence.source
    assert trace.terminal.output == value
    assert trace.terminal.failure is None


def test_root_inventory_is_retained_without_running_capture_or_defaults():
    start = StartNode(START_ID, params=[Parameter("topic", has_default=True, default=[1])])
    start.capture_backward_context = lambda: pytest.fail("initialization captured an invocation")
    engine = _engine(_graph([start], [(START_ID, END_ID)]), learnable=True)
    assert engine.state.trace.root_input_names == ("topic",)
    assert engine.state.trace.root_start_source is None
    assert engine.state.trace.lookup_source(next(iter(engine.state.trace.sources))) == {}
    start.params.clear()
    assert engine.state.trace.root_input_names == ("topic",)


def test_diamond_calls_shared_producer_once_with_two_ordered_messages():
    def passthrough(_context, feedback):
        return BackwardResult({"x": feedback})

    producer = RecordingBackwardNode("producer", lambda inputs: inputs["x"], passthrough,
                                     reads={"x": "${input.topic}"})
    left = RecordingBackwardNode("left", lambda inputs: inputs["x"], passthrough,
                                 reads={"x": "${producer.output}"})
    right = RecordingBackwardNode("right", lambda inputs: inputs["x"], passthrough,
                                  reads={"x": "${producer.output}"})
    flow = _graph([producer, left, right],
                  [(START_ID, "producer"), ("producer", "left"), ("producer", "right")],
                  outputs=[FlowOutput("left", "${left.output}"), FlowOutput("right", "${right.output}")])
    engine = _engine(flow, {"topic": "reefs"}, learnable=True)
    list(engine.run())
    from yian.runtime.backward import compute_backward

    root, results = compute_backward(engine.state.trace, {"left": "L", "right": "R"})
    validate_attributed_backward(engine.state.trace, {"left": "L", "right": "R"}, root, results)
    assert len(producer.calls) == len(left.calls) == len(right.calls) == 1
    assert producer.calls[0][1] == ("L", "R")
    assert root.input_feedback == {"topic": ("L", "R")}
    assert root.self_feedback == () and root.variable_feedback == {}
    assert tuple(item.source for item in results) == tuple(item.source for item in engine.state.trace.occurrences)
    assert all(item.invoked for item in results)


@pytest.mark.parametrize("bad", [
    lambda: BackwardResult({"x": ("valid",), "unknown": ("bad",)}),
    lambda: BackwardResult({"x": ("valid",)}, variable_feedback={"foreign": ("bad",)}),
    lambda: BackwardResult({"x": ([None],)}),
])
def test_entire_callback_result_is_validated_before_routing_any_input(bad):
    from yian.runtime.backward import _BackwardComputeError, compute_backward

    producer = RecordingBackwardNode("producer", lambda _: [1, 2], lambda *_: pytest.fail("routed partial result"))
    consumer = RecordingBackwardNode("consumer", lambda inputs: inputs["x"], lambda *_: bad(),
                                     reads={"x": "${producer.output}"})
    engine = _engine(_graph([producer, consumer], [(START_ID, "producer"), ("producer", "consumer")],
                            outputs=[FlowOutput("result", "${consumer.output}")]), learnable=True)
    list(engine.run())
    with pytest.raises(_BackwardComputeError) as caught:
        compute_backward(engine.state.trace, "review")
    assert caught.value.failure.code == "backward.result"
    assert caught.value.failure.node_id == "consumer"
    assert producer.calls == []


@pytest.mark.parametrize("feedback", [None, [None, None]])
def test_empty_feedback_keeps_complete_history_without_invoking_callbacks(feedback):
    from yian.runtime.backward import compute_backward

    node = RecordingBackwardNode("work", lambda _: [1, 2], lambda *_: pytest.fail("empty callback"))
    engine = _engine(_graph([node], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")]), learnable=True)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, feedback)
    validate_attributed_backward(engine.state.trace, feedback, root, results)
    assert root == BackwardResult()
    assert len(results) == len(engine.state.trace.occurrences)
    assert all(not result.invoked and result.feedback == () and result.result == BackwardResult() for result in results)
    with pytest.raises(ValueError, match="exactly 2"):
        compute_backward(engine.state.trace, [None])
    assert node.calls == []


def test_root_default_feedback_survives_its_literal_route():
    from yian.runtime.backward import compute_backward

    start = StartNode(START_ID, params=[Parameter("topic", has_default=True, default="reefs"), Parameter("absent", required=False)])
    engine = _engine(_graph([start], [], outputs=[FlowOutput("result", "${input.topic}")]), learnable=True)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "")
    validate_attributed_backward(engine.state.trace, "", root, results)
    assert root.input_feedback == {"topic": ("",)}


@pytest.mark.parametrize("early", [False, True])
def test_accepted_failure_uses_recorded_inputs_and_diagnostics(early):
    from yian.runtime.backward import compute_backward

    def body(_):
        raise RuntimeError("body broke")

    def backward(context, feedback):
        if context.inputs is None:
            return BackwardResult(self_feedback=("binding unavailable",))
        return BackwardResult({"x": feedback}, self_feedback=("body failure",))

    node = RecordingBackwardNode("work", body, backward, reads={"x": "${input.topic}"})
    if early:
        node.params.append(Parameter("missing", required=True))
    engine = _engine(_graph([node], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")]),
                     {"topic": "reefs"}, learnable=True)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review failure")
    validate_attributed_backward(engine.state.trace, "review failure", root, results)
    context, received = node.calls[0]
    assert received == ("review failure",)
    assert context.output is None and context.failure is not None
    assert (context.inputs is None) == early
    assert root.input_feedback == ({} if early else {"topic": ("review failure",)})
    assert root.self_feedback == ()
    assert next(item for item in results if item.source == engine.state.trace.terminal.source).invoked
    with pytest.raises(ValueError):
        compute_backward(engine.state.trace, {})


@pytest.mark.parametrize("explicit", [False, True])
def test_terminal_trace_latches_causal_route_and_later_failure_wins(explicit):
    from yian.compile.expand import build_grow
    from yian.nodes.base import Output, Route
    from yian.runtime.backward import compute_backward
    from yian.runtime.state import RunState, _replay_records
    from tests.engine.test_run_state_outcomes import _terminal_skip_route_flow

    flow = _terminal_skip_route_flow(sibling=True)
    flow.nodes["router"].forward = lambda _: Route("go")
    calls = []
    flow.nodes["router"].backward = lambda context, feedback: calls.append(context) or BackwardResult(self_feedback=("route",))
    observed = []

    def later(_):
        observed.append(engine.state.trace.terminal)
        if explicit:
            raise RuntimeError("later failure")
        return Output("done")

    flow.nodes["unrelated"].forward = later
    engine = _engine(flow, learnable=True)
    original = deepcopy(engine.state.flow)
    list(engine.run())
    trace = engine.state.trace
    chosen = next(item for item in trace.occurrences if item.node_id == ("unrelated" if explicit else "router"))
    assert observed == [None]
    assert trace.terminal.source == chosen.source and not trace.terminal.succeeded
    root, attributed = compute_backward(trace, "explain failure")
    validate_attributed_backward(trace, "explain failure", root, attributed)
    assert root == BackwardResult()
    assert len(calls) == (0 if explicit else 1)
    if calls:
        assert calls[0].failure == trace.terminal.failure
    for length in range(1, len(tuple(engine.log)) + 1):
        replay = _replay_records(RunState(deepcopy(original)), tuple(engine.log)[:length], grow_builder=build_grow)
        assert replay.trace.terminal == (trace.terminal if length == len(tuple(engine.log)) else None)


def test_explicitly_synthetic_source_free_failure_keeps_direct_root_text():
    from yian.diagnostics import Failure
    from yian.runtime.backward import compute_backward
    from yian.runtime.trace import RetainedTrace, TraceTerminal

    trace = RetainedTrace("synthetic", START_ID, END_ID)
    trace.terminal = TraceTerminal(False, None, None, Failure("graph.terminal_unreachable", "no executing producer", None))
    root, results = compute_backward(trace, "original text")
    validate_attributed_backward(trace, "original text", root, results)
    assert root == BackwardResult(self_feedback=("original text",))
    assert results == ()


def test_map_whole_list_context_and_item_order_survive_reversed_completion(monkeypatch):
    import threading
    from yian.events import NodeSucceeded
    from yian.nodes.map import MapNode
    from yian.nodes.code import CodeNode
    from yian.runtime.backward import compute_backward
    from yian.runtime.learning import project_owner_contexts
    from tests.engine.test_runtime_memory_learning import _owner_engine

    second_done = threading.Event()

    original_forward = CodeNode.forward
    calls = {}

    def forward(node, inputs):
        if node.execution_scope.occurrence_path[-1].index == 0:
            assert second_done.wait(5)
        return original_forward(node, inputs)

    def backward(_node, context, feedback):
        calls[context.node_started_seq] = (context, feedback)
        return BackwardResult({"x": feedback})

    monkeypatch.setattr(CodeNode, "forward", forward)
    monkeypatch.setattr(CodeNode, "backward", backward)
    work = CodeNode("work", code="return inputs['x']")
    work.params = [Parameter("x")]
    child = _graph([work], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")],
                   wiring={"work": {"x": "${input.x}"}})
    mapper = MapNode("map", flow_id="child", child=child)
    mapper.params = [Parameter("x")]
    producer = RecordingBackwardNode("producer", lambda inputs: inputs["items"],
                                     lambda _context, feedback: BackwardResult({"items": feedback}),
                                     reads={"items": "${input.items}"})
    flow = _graph([producer, mapper], [(START_ID, "producer"), ("producer", "map")],
                  outputs=[FlowOutput("result", "${map.output}")],
                  wiring={"producer": {"items": "${input.items}"}, "map": {"over": "${producer.output}", "x": "${item}"}})

    def observe(event):
        if isinstance(event, NodeSucceeded) and event.node_id == f"map#1/{END_ID}":
            second_done.set()

    engine, memories, stacks = _owner_engine(flow, {"items": [7, 7]}, num_workers=2, on_event=observe)
    list(engine.run())
    trace = engine.state.trace
    ends = [item.node_id for item in trace.occurrences if item.node_id in (f"map#0/{END_ID}", f"map#1/{END_ID}")]
    assert ends == [f"map#1/{END_ID}", f"map#0/{END_ID}"]
    root, results = compute_backward(trace, "review both items")
    validate_attributed_backward(trace, "review both items", root, results)
    assert producer.calls[0][1] == (["review both items", None], [None, "review both items"])
    assert root.input_feedback == {"items": (["review both items", None], [None, "review both items"])}
    for index in (0, 1):
        occurrence = next(item for item in trace.occurrences if item.node_id == f"map#{index}/work")
        context, feedback = calls[occurrence.source.execution_seq]
        assert feedback == ("review both items",)
        details = [detail for entry in context.downstream_context for detail in entry["contexts"]]
        assert any(detail["role"] == "output" and detail["result"] == [7, 7]
                   and detail["positions"] == [[index]] for detail in details)
    assert len(results) == len(trace.occurrences)
    owner_context, = project_owner_contexts(trace, memories, stacks, "review both items", root, results)
    assert len(owner_context.episodes) == len(trace.occurrences)
    producer_received = [entry for entry in owner_context.feedback if entry["channel"] == "output"
                         and entry["owner_static_path"] == ["producer"]]
    assert [entry["value"] for entry in producer_received] == [["review both items", None], [None, "review both items"]]
    for index in (0, 1):
        occurrence = next(item for item in trace.occurrences if item.node_id == f"map#{index}/work")
        received, = [entry for entry in owner_context.feedback if entry["channel"] == "output"
                     and entry["execution_seq"] == occurrence.source.execution_seq]
        assert received["occurrence_path"] == [{"kind": "map_item", "static_path": ["map"], "index": index}]
        assert received["execution_scope"]["occurrence_path"] == [{"kind": "map_item", "static_path": ["map"], "index": index}]
        assert received["contribution_context"] == calls[occurrence.source.execution_seq][0].downstream_context[0]
        assert any(detail["role"] == "output" and detail["result"] == [7, 7] and detail["positions"] == [[index]]
                   for detail in received["contribution_context"]["contexts"])


@pytest.mark.parametrize("transform", [False, True])
def test_structural_ancestry_requires_exact_metadata_prescribed_transfer(transform):
    from yian.runtime.backward import compute_backward
    from yian.runtime.learning import project_owner_contexts
    from tests.engine.test_runtime_memory_learning import _owner_engine

    producer = RecordingBackwardNode("producer", lambda _: "value", lambda *_: BackwardResult())
    middle = RecordingBackwardNode("middle", lambda inputs: inputs["x"],
                                   lambda _context, feedback: BackwardResult({"x": tuple(m + "!" for m in feedback) if transform else feedback}),
                                   reads={"x": "${producer.output}"})
    middle.capture_backward_context = lambda: {"input_output_paths": {"x": []}}
    left = RecordingBackwardNode("left", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": "${middle.output}"})
    right = RecordingBackwardNode("right", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                  reads={"x": "${middle.output}"})
    engine, memories, stacks = _owner_engine(_graph([producer, middle, left, right],
                            [(START_ID, "producer"), ("producer", "middle"), ("middle", "left"), ("middle", "right")],
                            outputs=[FlowOutput("left", "${left.output}"), FlowOutput("right", "${right.output}")]))
    list(engine.run())
    root, results = compute_backward(engine.state.trace, {"left": "same", "right": "same"})
    validate_attributed_backward(engine.state.trace, {"left": "same", "right": "same"}, root, results)
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "producer")
    result = next(item for item in results if item.source == occurrence.source)
    assert result.feedback == (("same!", "same!") if transform else ("same", "same"))
    assert tuple(item.parents for item in result.attribution) == (((0, 1), (0, 1)) if transform else ((0,), (1,)))
    context, = project_owner_contexts(engine.state.trace, memories, stacks, {"left": "same", "right": "same"}, root, results)
    received = [entry for entry in context.feedback if entry["channel"] == "output"
                and entry["execution_seq"] == occurrence.source.execution_seq]
    assert tuple(entry["value"] for entry in received) == result.feedback
    assert tuple(tuple(entry["contribution_context"]["parents"]) for entry in received) == tuple(
        attribution.parents for attribution in result.attribution
    )
    assert [entry["contribution_context"] for entry in received] == list(producer.calls[0][0].downstream_context)


def test_diamond_crossing_call_uses_one_walk_and_preserves_child_default_projection(monkeypatch):
    from yian.nodes.call import CallNode
    from yian.nodes.code import CodeNode
    from yian.runtime.backward import compute_backward
    from yian.memory import DeclSite, MemoryDecl
    from yian.runtime.learning import project_owner_contexts
    from tests.engine.test_runtime_memory_learning import _owner_engine

    calls = []

    def child_backward(_node, context, feedback):
        calls.append(context)
        return BackwardResult({"x": feedback, "literal": ("child default",)}, self_feedback=("child only",))

    monkeypatch.setattr(CodeNode, "backward", child_backward)
    start = StartNode(START_ID, params=[Parameter("x"), Parameter("literal", has_default=True, default="default")])
    work = CodeNode("work", code="return inputs['x']")
    work.params = [Parameter("x"), Parameter("literal")]
    child = _graph([start, work], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")],
                   wiring={"work": {"x": "${input.x}", "literal": "${input.literal}"}})
    child.memories = (MemoryDecl("child_notes", "child_notes", {}, DeclSite((), ("def", "child"), None)),)
    call = CallNode("call", flow_id="child", child=child)
    call.params = [Parameter("x")]
    producer = RecordingBackwardNode("producer", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                     reads={"x": "${input.topic}"})
    right = RecordingBackwardNode("right", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                  reads={"x": "${producer.output}"})
    flow = _graph([producer, call, right], [(START_ID, "producer"), ("producer", "call"), ("producer", "right")],
                  outputs=[FlowOutput("left", "${call.output}"), FlowOutput("right", "${right.output}")],
                  wiring={"producer": {"x": "${input.topic}"}, "call": {"x": "${producer.output}"}, "right": {"x": "${producer.output}"}})
    engine, memories, stacks = _owner_engine(flow, {"topic": "reefs"})
    list(engine.run())
    root, results = compute_backward(engine.state.trace, {"left": "L", "right": "R"})
    validate_attributed_backward(engine.state.trace, {"left": "L", "right": "R"}, root, results)
    assert len(calls) == 1 and producer.calls[0][1] == ("L", "R")
    assert len(producer.calls) == 1
    assert root.input_feedback == {"topic": ("L", "R")}
    boundary, = engine.state.trace.boundaries
    by_source = {item.source: item for item in results}
    assert by_source[boundary.start_source].result.input_feedback == {"x": ("L",), "literal": ("child default",)}
    assert not by_source[boundary.owner].invoked
    assert root.self_feedback == () and root.variable_feedback == {}
    contexts = project_owner_contexts(engine.state.trace, memories, stacks, {"left": "L", "right": "R"}, root, results)
    assert len(contexts) == 2
    for context in contexts:
        default, = [entry for entry in context.feedback if entry["execution_seq"] == boundary.start_source.execution_seq
                    and entry["channel"] == "input" and entry["target_name"] == "literal"]
        assert default["value"] == "child default"
        assert default["contribution_context"]["contributions"] == []
        self_entry, = [entry for entry in context.feedback if entry["channel"] == "self"]
        assert self_entry["value"] == "child only" and self_entry["message_index"] == 0
        assert self_entry["owner_static_path"] == ["call", "work"]
        assert not self_entry.get("root")


@pytest.mark.parametrize("same_boundary", [False, True])
def test_preseeded_root_output_does_not_invent_an_executed_producer(same_boundary):
    from yian.compile.expand import build_grow
    from yian.compile.model import CompiledFlow
    from yian.nodes.base import Route
    from yian.runtime.backward import compute_backward
    from yian.runtime.engine import FlowEngine
    from yian.runtime.log import MemoryRunLog
    from yian.runtime.state import RunStartData, RunState, commit_prepared, prepare_run_started
    from yian.typesys.pool import VariablePool

    start = StartNode(START_ID, params=[Parameter("topic"), Parameter("omitted", has_default=True, default="unused")])
    start.forward = lambda _inputs: Route("default")
    end_id = START_ID if same_boundary else END_ID
    flow = CompiledFlow({START_ID: start}, [])
    flow.end_id = end_id
    pool = VariablePool(start_id=START_ID)
    pool.set(START_ID, {"topic": "reefs", "undeclared": "extra"})
    if not same_boundary:
        pool.set(END_ID, "preseeded result")
    pool.add_system("run_id", "preseed")
    state, log = RunState(flow), MemoryRunLog("preseed")
    commit_prepared(log, state, prepare_run_started(state, RunStartData(
        "preseed", "f" * 64, pool, {}, {}, 0, learnable=True,
    )))
    list(FlowEngine(state, log, grow_builder=build_grow, num_workers=0).run())
    root, results = compute_backward(state.trace, "review")
    validate_attributed_backward(state.trace, "review", root, results)
    assert root.input_feedback == ({"topic": ("review",)} if same_boundary else {})
    assert root.self_feedback == ()
    assert all(not result.invoked for result in results)
    assert (state.trace.terminal.source.execution_seq == 0) if same_boundary else state.trace.terminal.source is None


def test_terminal_trace_append_rejection_leaves_all_indexes_unchanged(monkeypatch):
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from tests.engine._graph_builder import _forward_capture
    from tests.engine.test_forward_trace import _trace_snapshot

    engine = _engine(_graph([], [(START_ID, END_ID)]), learnable=True)
    state, log = engine.state, engine.log
    for node_id, value in ((START_ID, {}), (END_ID, None)):
        commit_prepared(log, state, prepare_node_started(state, node_id))
        prepared = prepare_node_succeeded(state, node_id, value, forward_capture=_forward_capture(state.flow.nodes[node_id], {}))
        if node_id == START_ID:
            commit_prepared(log, state, prepared)
    before = _trace_snapshot(state.trace)
    append = log.append
    monkeypatch.setattr(log, "append", lambda _: (_ for _ in ()).throw(OSError("reject terminal")))
    with pytest.raises(OSError, match="reject terminal"):
        commit_prepared(log, state, prepared)
    assert _trace_snapshot(state.trace) == before
    monkeypatch.setattr(log, "append", append)
    commit_prepared(log, state, prepared)
    assert state.trace.terminal.succeeded


def test_opaque_expression_stops_are_recoverable_without_rerunning_functions(monkeypatch):
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.expr.lineage import route_feedback
    from yian.runtime.backward import compute_backward

    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "backward_opaque", lambda value: calls.append(value) or value)
    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                 reads={"x": "${backward_opaque(input.topic)}"})
    engine = _engine(_graph([node], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")]),
                     {"topic": "reefs"}, learnable=True)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review opaque")
    validate_attributed_backward(engine.state.trace, "review opaque", root, results)
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "work")
    result = next(item.result for item in results if item.source == occurrence.source)
    recovered = route_feedback(dict(zip(occurrence.input_route.names, occurrence.input_route.children))["x"],
                               result.input_feedback["x"][0], engine.state.trace.lookup_source)
    assert recovered.contributions == () and recovered.stopped[0].feedback == "review opaque"
    assert recovered.stopped[0].context[-1].result == "reefs"
    assert root.input_feedback == {} and calls == ["reefs"]


def test_case_control_and_unselected_branch_do_not_receive_feedback():
    from yian.nodes.case import Case, CaseNode
    from yian.runtime.backward import compute_backward

    control = RecordingBackwardNode("control", lambda _: True, lambda *_: pytest.fail("control feedback"))
    value = RecordingBackwardNode("value", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                  reads={"x": "${input.topic}"})
    case = CaseNode("case", [Case("go", "${__r0}")])
    case.params = [Parameter("__r0")]
    selected = RecordingBackwardNode("selected", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                     reads={"x": "${value.output}"})
    unused = RecordingBackwardNode("unused", lambda _: pytest.fail("unselected forward"), lambda *_: pytest.fail("unselected backward"))
    flow = _graph([control, value, case, selected, unused],
                  [(START_ID, "control"), (START_ID, "value"), ("control", "case"),
                   ("case", "selected", "go"), ("case", "unused", "default"), ("value", "selected")],
                  outputs=[FlowOutput("result", "${selected.output | unused.output}")],
                  wiring={"value": {"x": "${input.topic}"}, "case": {"__r0": "${control.output}"},
                          "selected": {"x": "${value.output}"}})
    engine = _engine(flow, {"topic": "reefs"}, learnable=True)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review selected")
    validate_attributed_backward(engine.state.trace, "review selected", root, results)
    assert root.input_feedback == {"topic": ("review selected",)}
    assert control.calls == [] and unused.calls == []
    by_node = {item.node_id: item for item in engine.state.trace.occurrences}
    assert "unused" not in by_node
    assert by_node["case"].record.forward_capture.input_names == ()
    assert not next(item for item in results if item.source == by_node["case"].source).invoked


def test_agent_inventory_is_recorded_and_empty_feedback_does_no_backward_work(monkeypatch):
    from yian.compose.helpers.types import read_variables
    from yian.nodes.agent import AgentNode
    from yian.nodes.base import Output
    from yian.runtime.backward import compute_backward

    calls = []
    monkeypatch.setattr(AgentNode, "forward", lambda _node, inputs: Output(inputs["topic"]))

    def backward(_node, context, feedback):
        calls.append(context)
        return BackwardResult({"topic": feedback}, self_feedback=("agent note",), variable_feedback={"instruction": ("local note",)})

    monkeypatch.setattr(AgentNode, "backward", backward)
    node = AgentNode("agent", prompt="${topic}", variables=read_variables({"instruction": "str = concise"}, {}))
    node.params = [Parameter("topic")]
    engine = _engine(_graph([node], [(START_ID, "agent")], outputs=[FlowOutput("result", "${agent.output}")],
                            wiring={"agent": {"topic": "${input.topic}"}}), {"topic": "reefs"}, learnable=True)
    list(engine.run())
    node.params.clear()
    node.variables.clear()
    node.capture_backward_context = lambda: pytest.fail("recaptured invocation")
    _, inactive = compute_backward(engine.state.trace, None)
    assert calls == [] and len(inactive) == len(engine.state.trace.occurrences)
    root, active = compute_backward(engine.state.trace, "review")
    assert calls[0].input_names == ("topic",) and calls[0].variable_names == ("instruction",)
    assert root.input_feedback == {"topic": ("review",)} and root.self_feedback == () and root.variable_feedback == {}
    assert any(result.result.variable_feedback == {"instruction": ("local note",)} for result in active)


@pytest.mark.parametrize("kind", ["callback", "host", "cancel", "memory", "infrastructure"])
def test_callback_failures_are_attributed_without_swallowing_host_faults(kind, monkeypatch):
    from yian.nodes.base import _CapabilityFault, _HostStop
    from yian.runtime.backward import _BackwardComputeError, compute_backward

    original = _HostStop("stop") if kind == "cancel" else MemoryError("memory") if kind == "memory" else RuntimeError("broken")

    def backward(*_):
        raise _CapabilityFault(original) if kind == "host" else original

    node = RecordingBackwardNode("work", lambda _: "value", backward)
    engine = _engine(_graph([node], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")]), learnable=True)
    list(engine.run())
    if kind == "infrastructure":
        def broken_lookup(_source):
            raise original
        monkeypatch.setattr(engine.state.trace, "lookup_source", broken_lookup)
    if kind == "callback":
        with pytest.raises(_BackwardComputeError) as caught:
            compute_backward(engine.state.trace, "review")
        assert caught.value.failure.code == "backward.execute"
        assert caught.value.failure.node_id == "work"
        assert caught.value.__cause__ is original
    else:
        with pytest.raises(type(original)) as caught:
            compute_backward(engine.state.trace, "review")
        assert caught.value is original


def test_pooled_reverse_walk_uses_outcomes_instead_of_start_sequence(monkeypatch):
    import threading
    from yian.events import NodeSucceeded
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.runtime.backward import compute_backward

    committed = threading.Event()

    def wait_peer():
        assert committed.wait(5)
        return None

    monkeypatch.setitem(TEMPLATE_FNS, "backward_wait_peer", wait_peer)
    peer = RecordingBackwardNode("peer", lambda _: 7, lambda *_: BackwardResult(self_feedback=("peer reached",)))
    consumer = RecordingBackwardNode("consumer", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}),
                                     reads={"x": "${backward_wait_peer() | peer.output}"})

    def observe(event):
        if isinstance(event, NodeSucceeded) and event.node_id == "peer":
            committed.set()

    engine = _engine(_graph([consumer, peer], [(START_ID, "consumer"), (START_ID, "peer")],
                            outputs=[FlowOutput("result", "${consumer.output}")]),
                     learnable=True, num_workers=2, on_event=observe)
    list(engine.run())
    occurrences = {item.node_id: item for item in engine.state.trace.occurrences}
    assert occurrences["consumer"].source.execution_seq < occurrences["peer"].source.execution_seq
    assert occurrences["consumer"].outcome_seq > occurrences["peer"].outcome_seq
    root, attributed = compute_backward(engine.state.trace, "review")
    validate_attributed_backward(engine.state.trace, "review", root, attributed)
    assert len(consumer.calls) == len(peer.calls) == 1
    assert peer.calls[0][1] == ("review",)


def test_causal_route_trace_latch_is_installed_only_after_append(monkeypatch):
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded, prepare_node_routed
    from tests.engine._graph_builder import _forward_capture
    from tests.engine.test_forward_trace import _trace_snapshot
    from tests.engine.test_run_state_outcomes import _terminal_skip_route_flow

    engine = _engine(_terminal_skip_route_flow(sibling=True), learnable=True)
    state, log = engine.state, engine.log
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(state, START_ID, {}, forward_capture=_forward_capture(state.flow.nodes[START_ID], {})))
    for node_id in ("router", "unrelated"):
        commit_prepared(log, state, prepare_node_started(state, node_id))
    prepared = prepare_node_routed(state, "router", "go", forward_capture=_forward_capture(state.flow.nodes["router"], {}))
    before = _trace_snapshot(state.trace)
    append = log.append
    monkeypatch.setattr(log, "append", lambda _: (_ for _ in ()).throw(OSError("reject causal route")))
    with pytest.raises(OSError):
        commit_prepared(log, state, prepared)
    assert _trace_snapshot(state.trace) == before
    monkeypatch.setattr(log, "append", append)
    commit_prepared(log, state, prepared)
    assert state.trace.terminal is None
    causal = state.trace.occurrences[-1].source
    commit_prepared(log, state, prepare_node_succeeded(state, "unrelated", "done", forward_capture=_forward_capture(state.flow.nodes["unrelated"], {})))
    assert state.trace.terminal.source == causal


def test_failed_end_routes_text_without_fabricating_output_context():
    from yian.nodes.contracts import Assertion
    from yian.runtime.backward import compute_backward

    producer = RecordingBackwardNode("producer", lambda _: [1, 2], lambda *_: BackwardResult())
    engine = _engine(_graph([producer], [(START_ID, "producer")], outputs=[FlowOutput("result", "${producer.output}")]), learnable=True)
    engine.state.flow.nodes[END_ID].pre_asserts = [Assertion("false", "fail END")]
    list(engine.run())
    root, _ = compute_backward(engine.state.trace, "review failed END")
    assert root == BackwardResult()
    context, messages = producer.calls[0]
    assert messages == ("review failed END",)
    assert not any(detail["role"] == "output" for entry in context.downstream_context for detail in entry["contexts"])


def test_backward_context_is_trace_owned_detached_and_survives_replay(monkeypatch):
    from yian.compile.expand import build_grow
    from yian.nodes.base import Node
    from yian.runtime.backward import compute_backward
    from yian.runtime.state import RunState, _replay_records

    node = RecordingBackwardNode("work", lambda inputs: inputs["x"], lambda _c, f: BackwardResult({"x": f}), reads={"x": "${input.topic}"})
    engine = _engine(_graph([node], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")]), {"topic": [1, 2]}, learnable=True)
    original = deepcopy(engine.state.flow)
    list(engine.run())
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "work")
    context = engine.state.trace.backward_context(occurrence.source)
    context.inputs["x"].append(3)
    assert engine.state.trace.backward_context(occurrence.source).inputs == {"x": [1, 2]}
    before = tuple(engine.log)
    live_result = compute_backward(engine.state.trace, ["one", None])
    monkeypatch.setattr(Node, "capture_backward_context", lambda _: pytest.fail("replay recaptured"))
    replay = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    assert replay.trace.backward_context(occurrence.source) == context
    assert compute_backward(replay.trace, ["one", None]) == live_result
    assert tuple(engine.log) == before


def test_empty_map_retains_synthetics_without_inventing_item_contributors():
    from yian.nodes.code import CodeNode
    from yian.nodes.map import MapNode
    from yian.runtime.backward import compute_backward

    work = CodeNode("work", code="raise AssertionError('empty map ran a child')\nreturn None")
    work.params = [Parameter("x")]
    child = _graph([work], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")],
                   wiring={"work": {"x": "${input.x}"}})
    mapper = MapNode("map", flow_id="child", child=child)
    mapper.params = [Parameter("x")]
    engine = _engine(_graph([mapper], [(START_ID, "map")], outputs=[FlowOutput("result", "${map.output}")],
                            wiring={"map": {"over": [], "x": "${item}"}}), learnable=True)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review empty list")
    validate_attributed_backward(engine.state.trace, "review empty list", root, results)
    assert root == BackwardResult()
    assert engine.state.trace.boundaries == ()
    assert len(results) == len(engine.state.trace.occurrences) == 5


def test_loop_carried_routes_use_the_same_fold_after_live_pruning(monkeypatch):
    from yian.compose import load_flow
    from yian.nodes.code import CodeNode
    from yian.runtime.backward import compute_backward
    from yian.runtime.learning import project_owner_contexts
    from yian.runtime.state import RunState, _replay_records
    from yian.compile.expand import build_grow
    from tests.engine.test_runtime_memory_learning import _owner_engine

    calls = []

    def backward(_node, context, feedback):
        calls.append(context.node_started_seq)
        return BackwardResult({"n": tuple(message if type(message) is str else message["n"] for message in feedback)})

    monkeypatch.setattr(CodeNode, "backward", backward)
    flow = load_flow("""id: carried
name: carried
input: {n: int}
defs:
  body:
    long_term_memory: body_notes
    input: {n: int}
    nodes:
      work:
        kind: code
        input: {n: '${input.n}'}
        code: "return {'n': inputs['n'] - 1}"
        output: {n: int}
    output: ${work.output}
nodes:
  loop:
    kind: loop
    call: body
    times: 2
    input: {n: '${input.n}'}
output: ${loop.output}
""").compiled
    engine, memories, stacks = _owner_engine(flow, {"n": 3})
    original = deepcopy(engine.state.flow)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review carried value")
    validate_attributed_backward(engine.state.trace, "review carried value", root, results)
    assert root.input_feedback == {"n": ("review carried value",)}
    assert len(calls) == len(set(calls)) == 2
    drivers = [item for item in engine.state.trace.occurrences if item.record.forward_capture.scope.static_path == ("loop",)]
    assert len(drivers) == 3
    reached = {result.source for result in results if result.invoked}
    assert drivers[-1].source in reached and all(item.source not in reached for item in drivers[:-1])
    contexts = project_owner_contexts(engine.state.trace, memories, stacks, "review carried value", root, results)
    assert len(contexts) == 2
    by_key = {context.memory: context for context in contexts}
    assert len(by_key[_POLICY].episodes) == len(engine.state.trace.occurrences)
    assert len(by_key["body_notes"].episodes) == 6
    assert len({episode.execution_seq for episode in by_key[_POLICY].episodes}) == len(by_key[_POLICY].episodes)
    assert any(item.node_id not in engine.state.flow.nodes for item in engine.state.trace.occurrences)
    for owner_context in contexts:
        for episode in owner_context.episodes:
            occurrence = next(item for item in engine.state.trace.occurrences
                              if item.source.execution_seq == episode.execution_seq)
            assert episode.owner_static_path == occurrence.memory_scope.static_path
            assert episode.occurrence_path == occurrence.memory_scope.occurrence_path
    for driver in drivers[:-1]:
        episode = next(item for item in by_key[_POLICY].episodes if item.execution_seq == driver.source.execution_seq)
        assert episode.output["kind"] == "grow"
        assert not any(entry["execution_seq"] == driver.source.execution_seq for entry in by_key[_POLICY].feedback)
    monkeypatch.setattr(CodeNode, "backward", lambda *_: pytest.fail("projection replay invoked callback"))
    replay = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    assert project_owner_contexts(replay.trace, memories, stacks, "review carried value", root, results) == contexts


def test_callback_message_mutation_does_not_change_root_or_recorded_feedback():
    from yian.runtime.backward import compute_backward

    def backward(_context, feedback):
        feedback[0][0] = "mutated by callback"
        return BackwardResult()

    end = RecordingBackwardNode(END_ID, lambda _: [1, 2], backward)
    engine = _engine(_graph([end], [(START_ID, END_ID)]), learnable=True)
    list(engine.run())
    message = [None, "review second"]
    _, results = compute_backward(engine.state.trace, message)
    assert message == [None, "review second"]
    reached = next(item for item in results if item.invoked)
    assert reached.feedback == ([None, "review second"],)


@pytest.mark.parametrize("opaque", [False, True])
def test_canonical_file_input_feedback_keeps_literal_and_opaque_stops(opaque, monkeypatch):
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.expr.lineage import route_feedback
    from yian.runtime.backward import compute_backward
    from yian.typesys import FileRef
    from yian.runtime.learning import project_owner_contexts
    from tests.engine.test_runtime_memory_learning import _owner_engine

    value = {"files": [FileRef(uri="file:///tmp/example.txt")]}
    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "backward_file", lambda: calls.append(None) or value)
    feedback = {"files": [{"uri": "Check the URI."}]}
    node = RecordingBackwardNode("work", lambda _: "done", lambda *_: BackwardResult({"file": (feedback,)}),
                                 reads={"file": "${backward_file()}" if opaque else value})
    engine, memories, stacks = _owner_engine(_graph([node], [(START_ID, "work")], outputs=[FlowOutput("result", "${work.output}")]))
    list(engine.run())
    root, results = compute_backward(engine.state.trace, "review")
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "work")
    result = next(item.result for item in results if item.source == occurrence.source)
    capture = occurrence.record.forward_capture
    recovered = route_feedback(occurrence.input_route.children[0], result.input_feedback["file"][0],
                               engine.state.trace.lookup_source, recorded_value=capture.bound_inputs["file"])
    assert root == BackwardResult()
    assert recovered.contributions == ()
    assert [stop.feedback for stop in recovered.stopped] == ([feedback] if opaque else [])
    context, = project_owner_contexts(engine.state.trace, memories, stacks, "review", root, results)
    returned, = [entry for entry in context.feedback if entry["target_name"] == "file" and entry["channel"] == "input"]
    assert returned["value"] == feedback
    assert [entry["value"] for entry in context.feedback if entry["channel"] == "stopped"] == ([feedback] if opaque else [])
    assert calls == ([None] if opaque else [])


@pytest.mark.parametrize("opaque", [False, True])
def test_map_canonical_item_shape_preserves_its_projected_seed_stop(opaque, monkeypatch):
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.expr.lineage import route_feedback
    from yian.nodes.map import MapNode
    from yian.runtime.backward import compute_backward
    from yian.typesys import FileRef
    from yian.runtime.learning import project_owner_contexts
    from tests.engine.test_runtime_memory_learning import _owner_engine

    files = [FileRef(uri="file:///tmp/same.txt"), FileRef(uri="file:///tmp/same.txt")]
    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "backward_files", lambda: calls.append(None) or files)
    child = _graph([StartNode(START_ID, params=[Parameter("file")])], [],
                   outputs=[FlowOutput("uri", "${input.file.uri}")])
    mapper = MapNode("map", flow_id="child", child=child)
    mapper.params = [Parameter("file")]
    flow = _graph([mapper], [(START_ID, "map")], outputs=[FlowOutput("files", "${map.output}")],
                  wiring={"map": {"over": "${backward_files()}" if opaque else files, "file": "${item}"}})
    engine, memories, stacks = _owner_engine(flow)
    list(engine.run())
    root, results = compute_backward(engine.state.trace, [None, "Check second URI."])
    boundary = engine.state.trace.boundaries[1]
    start = engine.state.trace.occurrence(boundary.start_source)
    result = next(item.result for item in results if item.source == start.source)
    assert result.input_feedback == {"file": ({"uri": "Check second URI."},)}
    routed = route_feedback(start.input_route.children[0], result.input_feedback["file"][0],
                            engine.state.trace.lookup_source,
                            recorded_value=start.record.forward_capture.bound_inputs["file"])
    assert root == BackwardResult() and routed.contributions == ()
    assert [stop.feedback for stop in routed.stopped] == (
        [[None, {"uri": "Check second URI."}]] if opaque else []
    )
    context, = project_owner_contexts(engine.state.trace, memories, stacks, [None, "Check second URI."], root, results)
    assert [entry["value"] for entry in context.feedback if entry["channel"] == "stopped"] == (
        [[None, {"uri": "Check second URI."}]] if opaque else []
    )
    assert calls == ([None] if opaque else [])


@pytest.mark.parametrize("operator", ["+ []", "* 2"])
def test_map_bounded_projected_file_lists_compose_before_stopping(operator, monkeypatch):
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.expr.lineage import route_feedback
    from yian.nodes.map import MapNode
    from yian.runtime.backward import compute_backward
    from yian.typesys import FileRef

    raw = {"files": [FileRef(uri="file:///tmp/x")]}
    for _ in range(40):
        raw = {"nested": raw}
    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "deep_map_files", lambda: calls.append(None) or [raw])
    child = _graph([StartNode(START_ID, params=[Parameter("files")])], [],
                   outputs=[FlowOutput("files", "${input.files}")])
    mapper = MapNode("map", flow_id="child", child=child)
    mapper.params = [Parameter("files")]
    expression = "${item." + "nested." * 40 + "files " + operator + "}"
    flow = _graph([mapper], [(START_ID, "map")], outputs=[FlowOutput("files", "${map.output}")],
                  wiring={"map": {"over": "${deep_map_files()}", "files": expression}})
    engine = _engine(flow, learnable=True)
    list(engine.run())
    assert engine.state.trace.terminal.succeeded
    feedback = [{"uri": "Review."}]
    if operator == "* 2":
        feedback.append(None)
    root, results = compute_backward(engine.state.trace, [feedback])
    start = engine.state.trace.occurrence(engine.state.trace.boundaries[0].start_source)
    result = next(item.result for item in results if item.source == start.source)
    routed = route_feedback(start.input_route.children[0], result.input_feedback["files"][0],
                            engine.state.trace.lookup_source,
                            recorded_value=start.record.forward_capture.bound_inputs["files"])
    assert root == BackwardResult() and routed.contributions == ()
    assert [stop.feedback for stop in routed.stopped] == [[{"uri": "Review."}]]
    assert calls == [None]


_PASSTHROUGH_FLOWS = {
    "loop-output": ("""id: passthrough
name: passthrough
input: {{n: int, tag: str}}
defs:
  body:
    input: {{n: int}}
    nodes:
      work:
        kind: code
        input: {{n: '${{input.n}}'}}
        code: "return {{'n': inputs['n'] - 1}}"
        output: {{n: int}}
    output: ${{work.output}}
nodes:
  loop:
    kind: loop
    call: body
    times: {times}
    input: {{n: '${{input.n}}', tag: '${{input.tag}}'}}
output: ${{loop.output}}
""", {"tag": "review tag"}, {"tag": ("review tag",)}),
    "chat-shaped": ("""id: passthrough
name: passthrough
input: {{n: int, tag: str}}
defs:
  body:
    input: {{n: int, tag: str}}
    nodes:
      work:
        kind: code
        input: {{n: '${{input.n}}', tag: '${{input.tag}}'}}
        code: "return {{'n': inputs['n'] - 1, 'seen': inputs['tag']}}"
        output: {{n: int, seen: str}}
    output: ${{work.output}}
nodes:
  loop:
    kind: loop
    call: body
    times: {times}
    input: {{n: '${{input.n}}', tag: '${{input.tag}}', seen: ''}}
output: ${{loop.output}}
""", {"seen": "review seen"}, {"tag": ("review seen",)}),
    "nested-call": ("""id: passthrough
name: passthrough
input: {{n: int, tag: str}}
defs:
  dec:
    input: {{n: int}}
    nodes:
      work:
        kind: code
        input: {{n: '${{input.n}}'}}
        code: "return {{'n': inputs['n'] - 1}}"
        output: {{n: int}}
    output: ${{work.output}}
  body:
    input: {{n: int}}
    nodes:
      inner:
        kind: call
        call: dec
        input: {{n: '${{input.n}}'}}
    output: ${{inner.output}}
nodes:
  loop:
    kind: loop
    call: body
    times: {times}
    input: {{n: '${{input.n}}', tag: '${{input.tag}}'}}
output: ${{loop.output}}
""", {"tag": "review tag"}, {"tag": ("review tag",)}),
}


@pytest.mark.parametrize("times", [1, 2, 3, 6])
@pytest.mark.parametrize("shape", sorted(_PASSTHROUGH_FLOWS))
def test_loop_passthrough_field_routes_feedback_to_its_seed(shape, times, monkeypatch):
    from yian.compile.expand import build_grow
    from yian.compose import load_flow
    from yian.nodes.code import CodeNode
    from yian.runtime.backward import compute_backward
    from yian.runtime.state import RunState, _replay_records
    from tests.engine.test_runtime_memory_learning import _owner_engine

    def backward(_node, _context, feedback):
        # The body maps its output `seen` back to the `tag` input it echoed.
        routed = {}
        for message in feedback:
            for output, target in (("n", "n"), ("seen", "tag")):
                if output in message:
                    routed.setdefault(target, []).append(message[output])
        return BackwardResult({name: tuple(values) for name, values in routed.items()})

    monkeypatch.setattr(CodeNode, "backward", backward)
    source, feedback, expected = _PASSTHROUGH_FLOWS[shape]
    flow = load_flow(source.format(times=times)).compiled
    engine, _, _ = _owner_engine(flow, {"n": 10, "tag": "red"})
    original = deepcopy(engine.state.flow)
    list(engine.run())
    expected_output = {"n": 10 - times, "tag": "red"}
    if shape == "chat-shaped":
        expected_output["seen"] = "red"
    assert engine.state.trace.terminal.output == expected_output
    if shape == "nested-call" and times >= 3:
        assert not any(i.startswith("loop#0/") for i in engine.state.flow.nodes)
        assert any(i.startswith(f"loop#{times - 1}/") for i in engine.state.flow.nodes)
    root, results = compute_backward(engine.state.trace, feedback)
    validate_attributed_backward(engine.state.trace, feedback, root, results)
    assert root.input_feedback == expected
    replay = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    assert compute_backward(replay.trace, feedback) == (root, results)
