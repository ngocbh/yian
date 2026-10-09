"""Accepted forward captures and producer identities survive reducer replay."""

from dataclasses import FrozenInstanceError
import json

import pytest

from yian.expr.lineage import SourceRef, ValueRoute
from yian.memory import ConversationInvocation, EvaluationEvidence
from yian.nodes.contracts import ExecutionScope
from yian.runtime.log import (
    SCHEMA_VERSION, NodeSucceededRecord, RecordEnvelope, canonical_typed,
    decode_record, encode_record,
)


@pytest.mark.parametrize("schema", [SCHEMA_VERSION])
def test_forward_capture_round_trips_invocation_names_and_structural_layout(schema):
    from yian.runtime.contracts import ForwardCapture

    data = {"input_output_paths": {"result": []}}
    capture = ForwardCapture(
        {"result": {"items": [1]}},
        ValueRoute.record((("result", ValueRoute.literal({"items": [1]})),)),
        node_data=data, input_names=("result",), variable_names=("instruction",),
    )
    data["input_output_paths"]["result"].append("changed")
    record = NodeSucceededRecord("end", "end", canonical_typed({"items": [1]}, None)[0], forward_capture=capture)
    envelope = RecordEnvelope(schema, "run", 3, record)
    decoded = decode_record(encode_record(envelope))
    assert decoded == envelope
    assert decoded.record.forward_capture.input_names == ("result",)
    assert decoded.record.forward_capture.variable_names == ("instruction",)
    assert decoded.record.forward_capture.node_data == {"input_output_paths": {"result": []}}
    assert decoded.schema == schema
    old = json.loads(encode_record(envelope))
    old["schema"] = 6
    with pytest.raises(ValueError, match="schema"):
        decode_record(json.dumps(old))


@pytest.mark.parametrize("field,value", [
    ("input_names", None), ("input_names", ["result", "result"]), ("input_names", [1]),
    ("variable_names", ["instruction", "instruction"]), ("variable_names", [False]),
    ("node_data", {"input_output_paths": []}),
    ("node_data", {"input_output_paths": {"unknown": []}}),
    ("node_data", {"input_output_paths": {"result": [True]}}),
    ("node_data", {"input_output_paths": {"result": [-1]}}),
    ("node_data", {"input_output_paths": {"result": "result"}}),
])
def test_forward_capture_rejects_malformed_invocation_inventory_and_paths(field, value):
    from yian.runtime.contracts import ForwardCapture

    capture = ForwardCapture({"result": 1}, ValueRoute.record((("result", ValueRoute.literal(1)),)),
                             input_names=("result",), variable_names=("instruction",))
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 3, NodeSucceededRecord(
        "end", "end", canonical_typed(1, None)[0], forward_capture=capture,
    ))
    wire = json.loads(encode_record(envelope))
    wire["payload"]["forward_capture"][field] = value
    with pytest.raises((TypeError, ValueError)):
        decode_record(json.dumps(wire))


@pytest.mark.parametrize("field", ["input_names", "variable_names"])
def test_schema_six_requires_explicit_invocation_inventories(field):
    from yian.runtime.contracts import ForwardCapture

    capture = ForwardCapture({}, ValueRoute.record(()))
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 3, NodeSucceededRecord(
        "work", "work", canonical_typed(None, None)[0], forward_capture=capture,
    ))
    wire = json.loads(encode_record(envelope))
    wire["payload"]["forward_capture"].pop(field)
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire))


def test_accepted_backward_inventories_and_layout_replay_without_recapturing(monkeypatch):
    import copy
    from yian.compile.expand import build_grow
    from yian.compile.model import END_ID, START_ID
    from yian.compose import load_flow
    from yian.nodes.base import BackwardContext, Node
    from yian.nodes.code import CodeNode
    from yian.nodes.end import EndNode
    from yian.nodes.start import StartNode
    from yian.runtime.state import RunState, _replay_records
    from tests.engine._graph_builder import _engine

    flow = load_flow("""id: captured_layout
name: captured_layout
input: {topic: 'str = reefs'}
nodes:
  work:
    kind: code
    input: {topic: '${input.topic}'}
    code: return inputs['topic']
    output: str
output: ${work.output}
asserts: ["${work.output} != null"]
""").compiled
    engine = _engine(flow, learnable=True)
    original = copy.deepcopy(engine.state.flow)
    list(engine.run())
    occurrences = {item.node_id: item for item in engine.state.trace.occurrences}
    start = occurrences[START_ID].record.forward_capture
    terminal = occurrences[END_ID].record.forward_capture
    assert start.input_names == ("topic",) and start.bound_inputs == {"topic": "reefs"}
    assert terminal.input_names == ("result",)
    assert len(terminal.bound_inputs) > 1
    assert terminal.node_data == {"input_output_paths": {"result": []}}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("replay executed or recaptured a node")

    monkeypatch.setattr(CodeNode, "forward", forbidden)
    input_names = Node.backward_input_names
    initialized_roots = []

    def root_inventory_only(node):
        assert node.id == START_ID
        initialized_roots.append(node.id)
        return input_names(node)

    monkeypatch.setattr(Node, "backward_input_names", root_inventory_only)
    monkeypatch.setattr(Node, "declared_variables", forbidden)
    for kind in (Node, StartNode, EndNode):
        monkeypatch.setattr(kind, "capture_backward_context", forbidden)
    replay = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    assert initialized_roots == [START_ID]
    assert replay.trace.occurrences == engine.state.trace.occurrences
    occurrence = next(item for item in replay.trace.occurrences if item.node_id == END_ID)
    handler = replay.trace.backward_handlers[occurrence.source]
    handler.__self__._names[:] = ["changed", "afterward"]
    handler.__self__.params.clear()
    capture = occurrence.record.forward_capture
    context = BackwardContext(
        capture.bound_inputs, input_names=capture.input_names, variable_names=capture.variable_names,
        output=replay.trace.sources[occurrence.source].value, node_data=capture.node_data,
    )
    assert handler(context, ("Review the answer.",)).input_feedback == {"result": ("Review the answer.",)}


def test_static_capture_host_fault_does_not_append_a_node_outcome():
    from yian.compile.model import END_ID, START_ID
    from yian.runtime.log import NodeStartedRecord
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    expected = ValueError("capture failed")
    node = FuncNode("work", lambda _inputs: "body must not run")

    def fail():
        raise expected

    node.capture_backward_context = fail
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learnable=True)
    with pytest.raises(ValueError) as caught:
        list(engine.run())
    assert caught.value is expected
    assert tuple(engine.log)[-1].record == NodeStartedRecord("work")
    assert all(item.node_id != "work" for item in engine.state.trace.occurrences)
    assert engine.state.pool.get("work") is None


def test_nested_agent_render_capture_uses_authored_model_map_and_replays_without_execution(monkeypatch):
    import copy
    import yian.nodes.agent.node as agent_module
    from yian.compile.expand import build_grow
    from yian.compose import load_flow
    from yian.events import RunSucceeded
    from yian.nodes.agent import AgentNode
    from yian.nodes.base import Output
    from yian.runtime.state import RunState, _replay_records
    from tests.engine._graph_builder import _engine

    flow = load_flow("""id: captured_models
name: captured_models
llm_config: {provider: test, model: root, temperature: 0.2}
defs:
  leaf:
    input: {topic: str}
    nodes:
      writer:
        kind: agent
        mode: plain
        input: {topic: '${input.topic}'}
        prompt: '${topic}: ${instruction}'
        variables: {instruction: 'str = clearly'}
        output: {topic: str}
    output: ${writer.output}
  repeat_left:
    llm_config: {model: left}
    input: {topic: str}
    nodes:
      twice:
        kind: loop
        call: leaf
        times: 2
        input: {topic: '${input.topic}'}
    output: ${twice.output}
  repeat_right:
    llm_config: {model: right}
    input: {topic: str}
    nodes:
      twice:
        kind: loop
        call: leaf
        times: 2
        input: {topic: '${input.topic}'}
    output: ${twice.output}
nodes:
  left:
    kind: call
    call: repeat_left
    input: {topic: reefs}
  right:
    kind: map
    call: repeat_right
    over: [reefs, reefs]
    input: {topic: '${item}'}
output: {left: '${left.output}', right: '${right.output}'}
""").compiled
    rendered = []

    def mode(ctx):
        rendered.append((ctx.prompt, dict(ctx.llm_config)))
        return Output({"topic": ctx.prompt})

    monkeypatch.setitem(agent_module.MODES, "plain", mode)
    engine = _engine(flow, learnable=True)
    original = copy.deepcopy(engine.state.flow)
    list(engine.run())
    assert isinstance(engine.state.result, RunSucceeded), engine.state.result
    trace = engine.state.trace
    agents = [item for item in trace.occurrences if item.record.forward_capture.variable_names == ("instruction",)]
    assert len(agents) == len(rendered) == 6
    assert engine.state.effective_llm_by_static_path == {
        "left/twice/writer": {"provider": "test", "model": "left", "temperature": 0.2},
        "right/twice/writer": {"provider": "test", "model": "right", "temperature": 0.2},
    }
    positions = set()
    for occurrence in agents:
        capture = occurrence.record.forward_capture
        scope, data = capture.scope, capture.node_data
        expected = engine.state.effective_llm_by_static_path["/".join(scope.static_path)]
        assert data["effective_llm_config"] == expected
        assert data["prompt_template"] == "${topic}: ${instruction}"
        assert data["variables"] == {"instruction": "clearly"}
        assert data["rendered_prompt"] == capture.bound_inputs["topic"] + ": clearly"
        assert scope.static_path in (("left", "twice", "writer"), ("right", "twice", "writer"))
        assert [item.kind for item in scope.occurrence_path] == (
            ["call", "loop_body"] if scope.static_path[0] == "left" else ["map_item", "loop_body"]
        )
        positions.add((scope.static_path, scope.occurrence_path))
    assert len(positions) == 6
    assert any(item.node_id not in engine.state.flow.nodes for item in agents)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("replay tried to resolve, render, capture, or invoke a model")

    monkeypatch.setattr(AgentNode, "_resolve_variables", forbidden)
    monkeypatch.setattr(AgentNode, "capture_backward_context", forbidden)
    monkeypatch.setattr(agent_module, "render_template_record", forbidden)
    monkeypatch.setitem(agent_module.MODES, "plain", forbidden)
    replay = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    assert replay.trace.occurrences == trace.occurrences
    for occurrence in agents:
        handler = replay.trace.backward_handlers[occurrence.source].__self__
        handler.prompt = "later prompt"
        handler.llm_config["model"] = "later model"
        data = replay.trace.backward_context(occurrence.source).node_data
        evidence = data.pop("_evaluation_evidence")
        assert evidence["interactions"] == []
        assert data == occurrence.record.forward_capture.node_data


@pytest.mark.parametrize("failure_at", ["capture", "append"])
def test_agent_render_capture_host_and_append_failures_do_not_install_worker_data(monkeypatch, failure_at):
    import yian.nodes.agent.node as agent_module
    import yian.runtime.eval_node as evaluator
    from yian.compile.model import END_ID, START_ID
    from yian.nodes.agent import AgentNode
    from yian.nodes.base import Output
    from yian.runtime.log import NodeStartedRecord
    from tests.engine._graph_builder import _engine, _graph

    expected = OSError("capture or append unavailable")
    node = AgentNode("writer", prompt="actual prompt")
    modes = []
    monkeypatch.setitem(agent_module.MODES, "plain", lambda ctx: modes.append(ctx.prompt) or Output(ctx.prompt))
    engine = _engine(_graph([node], [(START_ID, "writer"), ("writer", END_ID)]), learnable=True)
    canonical, append = evaluator.canonical_seed, engine.log.append

    def fail_capture(value, **kwargs):
        if type(value) is dict and "rendered_prompt" in value:
            raise expected
        return canonical(value, **kwargs)

    def fail_append(record):
        if type(record) is NodeSucceededRecord and record.node_id == "writer":
            assert record.forward_capture.node_data["rendered_prompt"] == "actual prompt"
            raise expected
        return append(record)

    if failure_at == "capture":
        monkeypatch.setattr(evaluator, "canonical_seed", fail_capture)
    else:
        monkeypatch.setattr(engine.log, "append", fail_append)
    with pytest.raises(OSError) as caught:
        list(engine.run())
    assert caught.value is expected
    assert modes == ([] if failure_at == "capture" else ["actual prompt"])
    assert tuple(engine.log)[-1].record == NodeStartedRecord("writer")
    assert all(item.node_id != "writer" for item in engine.state.trace.occurrences)
    assert "writer" not in engine.state.trace.current_writers
    assert engine.state.pool.get("writer") is None


def test_discarded_agent_worker_keeps_render_capture_out_of_accepted_trace(monkeypatch):
    import threading
    import yian.nodes.agent.node as agent_module
    from yian.compile.model import END_ID, START_ID
    from yian.nodes.agent import AgentNode
    from yian.nodes.base import Output
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    entered, stop = threading.Event(), threading.Event()
    rendered = []

    def mode(ctx):
        rendered.append(ctx.prompt)
        entered.set()
        assert stop.wait(5)
        return Output(ctx.prompt)

    def fail(_inputs):
        assert entered.wait(5)
        raise ValueError("sibling failed")

    monkeypatch.setitem(agent_module.MODES, "plain", mode)
    slow = AgentNode("slow", prompt="captured but discarded")
    flow = _graph([slow, FuncNode("bad", fail)], [
        (START_ID, "slow"), (START_ID, "bad"), ("slow", END_ID), ("bad", END_ID),
    ])
    engine = _engine(flow, learnable=True, num_workers=2, stop_event=stop)
    list(engine.run())
    assert rendered == ["captured but discarded"]
    assert "slow" in engine.state.trace.started
    assert [item.node_id for item in engine.state.trace.occurrences] == [START_ID, "bad"]
    assert "slow" not in engine.state.trace.current_writers


@pytest.mark.parametrize("schema", [SCHEMA_VERSION])
def test_forward_capture_round_trips_without_memory_scope(schema):
    from yian.runtime.contracts import ForwardCapture

    inputs = {"value": [1, None]}
    route = ValueRoute.record((("value", ValueRoute(
        "source", source=SourceRef("run", 1, ("nested",)),
    )),))
    item = ForwardCapture({"x": None}, ValueRoute.record((
        ("x", ValueRoute.literal(None)),
    )), evaluated_keys=frozenset({"x"}))
    capture = ForwardCapture(
        inputs, route, evaluated_keys=frozenset({"value"}), items=((0, item),),
        scope=ExecutionScope(("work",)), node_data={"saved": [2]},
    )
    inputs["value"].append(99)
    evidence = EvaluationEvidence(None, {"value": [1, None]}, None, None)
    record = NodeSucceededRecord(
        "work", "work", canonical_typed(None, None)[0],
        evidence=evidence, forward_capture=capture,
    )
    envelope = RecordEnvelope(schema, "run", 3, record)
    decoded = decode_record(encode_record(envelope))
    assert decoded == envelope
    assert decoded.record.forward_capture.bound_inputs == {"value": [1, None]}
    decoded.record.forward_capture.bound_inputs["value"].append(8)
    decoded.record.forward_capture.node_data["saved"].append(9)
    assert capture.bound_inputs == {"value": [1, None]}
    assert capture.node_data == {"saved": [2]}
    with pytest.raises(FrozenInstanceError):
        capture.scope = None
    assert decoded.schema == schema
    old = json.loads(encode_record(envelope))
    old["schema"] = 6
    with pytest.raises(ValueError, match="schema"):
        decode_record(json.dumps(old))


@pytest.mark.parametrize("field,value", [
    ("source", {"run_id": "run", "execution_seq": True, "path": []}),
    ("source", {"run_id": "run", "execution_seq": 1, "path": [-1]}),
    ("source", {"run_id": "run", "execution_seq": 1, "path": [False]}),
    ("source", {"run_id": "", "execution_seq": 1, "path": []}),
    ("source", {"run_id": "run", "execution_seq": 1, "path": [], "extra": 1}),
    ("children", [{}]), ("count", True), ("count", 2),
    ("kind", "unknown"), ("names", ["unused"]), ("sizes", [2]),
    ("value_json", "Infinity"), ("expression", 2),
    ("path", [0]), ("spans", [[0, 1]]),
])
def test_forward_capture_rejects_malformed_nested_route_wire(field, value):
    from yian.runtime.contracts import ForwardCapture

    capture = ForwardCapture({"x": 1}, ValueRoute.record(((
        "x", ValueRoute("source", source=SourceRef("run", 1)),
    ),)))
    record = NodeSucceededRecord("work", "work", canonical_typed(None, None)[0],
                                 forward_capture=capture)
    wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 3, record)))
    wire["payload"]["forward_capture"]["route"]["children"][0][field] = value
    with pytest.raises((TypeError, ValueError)):
        decode_record(json.dumps(wire))


@pytest.mark.parametrize("value_json", [
    pytest.param('{"a":1,"a":2}', id="duplicate-key"),
    pytest.param('{"nested":{"a":1,"a":2}}', id="nested-duplicate-key"),
    pytest.param('"\\ud800"', id="surrogate-string"),
    pytest.param('{"nested":"\\ud800"}', id="surrogate-value"),
    pytest.param('{"\\ud800":1}', id="surrogate-key"),
])
def test_forward_capture_rejects_noncanonical_embedded_json(value_json):
    from yian.runtime.contracts import ForwardCapture

    route = ValueRoute("literal", value_json=value_json)
    with pytest.raises((TypeError, ValueError)):
        ForwardCapture({"x": None}, ValueRoute.record((("x", route),)))

    capture = ForwardCapture({"x": None}, ValueRoute.record((("x", ValueRoute.literal(None)),)))
    record = NodeSucceededRecord("work", "work", canonical_typed(None, None)[0],
                                 forward_capture=capture)
    wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 3, record)))
    wire["payload"]["forward_capture"]["route"]["children"][0]["value_json"] = value_json
    with pytest.raises((TypeError, ValueError)):
        decode_record(json.dumps(wire))


def test_trace_source_is_the_actual_commit_as_producer():
    from yian.nodes.base import Node, Output
    from yian.compile.model import START_ID, END_ID
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    class Redirect(Node):
        def forward(self, inputs):
            return Output({"value": None}, commit_as="target")

    producer = Redirect("physical")
    consumer = FuncNode("read", lambda inputs: inputs,
                        reads={"x": "${target.output.value}"})
    flow = _graph([producer, FuncNode("target", lambda _: None), consumer], [
        (START_ID, "physical"), ("target", "read"), ("read", END_ID),
    ])
    engine = _engine(flow, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    occurrences = {item.node_id: item for item in trace.occurrences}
    physical = occurrences["physical"]
    entry = trace.current_writers["target"]
    assert entry.source == physical.source
    assert entry.source.execution_seq < physical.outcome_seq
    assert trace.lookup_source(entry.source) == {"value": None}
    consumer_capture = occurrences["read"].record.forward_capture
    assert consumer_capture.bound_inputs == {"x": None}
    assert consumer_capture.route.children[0].source == SourceRef(
        engine.state.run_id, physical.source.execution_seq, ("value",),
    )
    assert trace.backward_handlers[physical.source].__self__ is producer


def test_forward_capture_keeps_the_existing_canonical_depth_range():
    from yian.runtime.contracts import ForwardCapture
    from yian.runtime.log import canonical_seed

    value = 1
    for _ in range(140):
        value = [value]
    _, inputs = canonical_seed({"x": value})
    for route in (ValueRoute.literal(value), ValueRoute("source", source=SourceRef("run", 0, ("x",)))):
        capture = ForwardCapture(inputs, ValueRoute.record((("x", route),)))
        record = NodeSucceededRecord("work", "work", canonical_typed(None, None)[0],
                                     forward_capture=capture)
        envelope = RecordEnvelope(SCHEMA_VERSION, "run", 3, record)
        assert decode_record(encode_record(envelope)) == envelope
        assert capture.bound_inputs == inputs


@pytest.mark.parametrize("stage", ["binding", "body", "output", "route", "grow"])
def test_failed_binding_and_rejected_outcomes_retain_canonical_failures(stage):
    from yian.compile.model import START_ID, END_ID
    from yian.nodes.base import Node, Output, Route, Grow
    from yian.nodes.contracts import Parameter
    from tests.engine._graph_builder import _engine, _graph

    class Failing(Node):
        def forward(self, inputs):
            if stage == "body":
                raise RuntimeError("body failed")
            if stage == "output":
                return Output(1, commit_as="absent")
            if stage == "route":
                return Route("absent")
            return Grow(seed={1})

    node = Failing("work")
    node.params = [Parameter("x")]
    node.is_spawner = True
    wiring = {"work": {"x": "${input.x:?missing}" if stage == "binding" else 1}}
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)], wiring=wiring),
                     learnable=True, grow_builder=lambda *_: None)
    list(engine.run())
    accepted = engine.state.trace.occurrences[-1]
    assert accepted.node_id == "work"
    record = accepted.record
    assert record.failure == engine.state.result.failure
    assert record.forward_capture.bound_inputs == (None if stage == "binding" else {"x": 1})


def test_trace_is_applied_only_after_successful_append(monkeypatch):
    from yian.compile.model import START_ID, END_ID
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from tests.engine._graph_builder import _engine, _forward_capture, _graph

    engine = _engine(_graph([], [(START_ID, END_ID)]), learnable=True)
    state, log = engine.state, engine.log
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    trace = state.trace
    prior = (trace.occurrences, dict(trace.sources), dict(trace.current_writers),
             dict(trace.backward_handlers), dict(trace.started), trace.next_sequence)
    prepared = prepare_node_succeeded(state, START_ID, {},
                                      forward_capture=_forward_capture(state.flow.nodes[START_ID], {}))
    assert trace.occurrences == ()
    original_append = log.append

    def reject(_):
        raise OSError("append failed")

    monkeypatch.setattr(log, "append", reject)
    with pytest.raises(OSError, match="append failed"):
        commit_prepared(log, state, prepared)
    assert (trace.occurrences, dict(trace.sources), dict(trace.current_writers),
            dict(trace.backward_handlers), dict(trace.started), trace.next_sequence) == prior
    monkeypatch.setattr(log, "append", original_append)
    commit_prepared(log, state, prepared)
    assert len(trace.occurrences) == 1
    assert trace.current_writers[START_ID].source.execution_seq == 1


def test_pooled_lookup_keeps_value_and_producer_identity_together(monkeypatch):
    import threading
    from yian.compile.model import START_ID, END_ID
    from yian.events import NodeSucceeded
    from yian.expr.builtins import TEMPLATE_FNS
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    consumer_waiting = threading.Event()
    producer_committed = threading.Event()

    def pause():
        consumer_waiting.set()
        assert producer_committed.wait(5)
        return None

    def produce(_):
        assert consumer_waiting.wait(5)
        return {"value": 7, "null": None}

    def observe(event):
        if isinstance(event, NodeSucceeded) and event.node_id == "peer":
            producer_committed.set()

    monkeypatch.setitem(TEMPLATE_FNS, "await_peer", pause)
    consumer = FuncNode("consumer", lambda inputs: inputs, reads={
        "before": "${peer.output.value | 0}",
        "pause": "${await_peer()}",
        "after": "${peer.output.value | 0}",
        "null": "${peer.output.null}",
        "missing": "${peer.output.missing}",
    })
    flow = _graph([consumer, FuncNode("peer", produce)], [
        (START_ID, "consumer"), (START_ID, "peer"), ("consumer", END_ID),
    ])
    engine = _engine(flow, learnable=True, num_workers=2, on_event=observe)
    list(engine.run())
    occurrences = {entry.node_id: entry for entry in engine.state.trace.occurrences}
    peer, read = occurrences["peer"], occurrences["consumer"]
    assert read.source.execution_seq < peer.source.execution_seq
    assert peer.outcome_seq < read.outcome_seq
    capture = read.record.forward_capture
    assert capture.bound_inputs == {"before": 0, "pause": None, "after": 7, "null": None, "missing": None}
    routes = dict(zip(capture.route.names, capture.route.children))
    assert routes["before"].children[0].kind == "literal"
    assert routes["after"].children[0].source == SourceRef(engine.state.run_id, peer.source.execution_seq, ("value",))
    assert routes["null"].source == SourceRef(engine.state.run_id, peer.source.execution_seq, ("null",))
    assert routes["missing"].kind == "literal"


def test_discarded_pooled_outcomes_are_not_trace_occurrences():
    import threading
    from yian.compile.model import START_ID, END_ID
    from yian.runtime.backward import compute_backward
    from yian.runtime.contracts import BackwardOutcome
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord
    from yian.runtime.state import prepare_record, commit_prepared
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    stop = threading.Event()
    waiting = threading.Event()
    body_finished = []

    def discard(_):
        waiting.set()
        assert stop.wait(5)
        body_finished.append(True)
        return "discarded"

    def fail(_):
        assert waiting.wait(5)
        raise RuntimeError("failed")

    flow = _graph([FuncNode("slow", discard), FuncNode("bad", fail)], [
        (START_ID, "slow"), (START_ID, "bad"), ("slow", END_ID), ("bad", END_ID),
    ])
    engine = _engine(flow, learnable=True, num_workers=2, stop_event=stop)
    list(engine.run())
    assert body_finished == [True]
    trace = engine.state.trace
    assert "slow" in trace.started
    assert [entry.node_id for entry in trace.occurrences] == [START_ID, "bad"]
    assert "slow" not in trace.current_writers
    assert all(handler.__self__.id != "slow" for handler in trace.backward_handlers.values())
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardStartedRecord(1, [None]))
    root, attributed = compute_backward(trace, None)
    for record in (BackwardStartedRecord(1, None), BackwardReadyRecord(1, root, attributed, ()),
                   BackwardFinishedRecord(1, BackwardOutcome("succeeded", root))):
        assert commit_prepared(engine.log, engine.state, prepare_record(engine.state, record)) == ()
    assert len(attributed) == 2 and all(not item.invoked for item in attributed)
    assert engine.state.backward_state.outcome.feedback == root


def test_trace_memory_scope_is_detached_and_replayed_without_evidence():
    from copy import deepcopy
    from yian.compile.expand import build_grow
    from yian.compile.model import START_ID, FlowOutput
    from yian.memory import DeclSite, MemoryScope, MemorySnapshotEnvelope, MemorySpec, MemoryStack
    from yian.runtime.state import RunState, _replay_records
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    scope = MemoryScope(("code",), ("memory",))
    node = FuncNode("code", lambda _: {"items": [1]})
    node.memory_scope = scope
    engine = _engine(_graph([node], [(START_ID, "code")], outputs=[FlowOutput("value", "${code.output}")]),
                     learnable=True,
                     memories=(MemorySpec("memory", "memory", {}, (DeclSite((), ("root",), None),), False),),
                     memory_stacks=(MemoryStack(("code",), ("memory",)),),
                     memory_snapshots=(MemorySnapshotEnvelope("memory", {}),))
    original = deepcopy(engine.state.flow)
    list(engine.run())
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "code")
    expected = deepcopy(scope)
    assert occurrence.record.evidence is None
    assert occurrence.memory_scope == expected and occurrence.memory_scope is not scope
    assert "memory_scope" not in json.loads(occurrence._record_bytes)["payload"]
    engine.state.flow.nodes["code"].memory_scope = None
    object.__setattr__(scope, "memories", [])
    detached = occurrence.memory_scope
    object.__setattr__(detached, "static_path", ("foreign",))
    assert occurrence.memory_scope == expected
    replay = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    assert replay.trace.occurrence(occurrence.source).memory_scope == expected


@pytest.mark.parametrize("bad", ["carrier", "nested"])
def test_trace_rejects_forged_memory_scope_before_append_and_when_copied(bad):
    from copy import deepcopy
    from dataclasses import replace
    from yian.memory import MemoryScope
    from yian.runtime.state import prepare_node_succeeded, commit_prepared
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _forward_capture
    from tests.engine.test_runtime_memory_learning import _start_trace_node

    node = FuncNode("code", lambda _: "done")
    node.memory_scope = MemoryScope(("code",), ("memory",))
    state, log = _start_trace_node(node, keys=("memory",), learnable=True)
    prepared = prepare_node_succeeded(state, node.id, "done", forward_capture=_forward_capture(node, {}))
    occurrence = prepared._plan.trace_delta.occurrence
    malformed = object() if bad == "carrier" else deepcopy(node.memory_scope)
    if bad == "nested":
        object.__setattr__(malformed, "memories", [])
    with pytest.raises(TypeError):
        replace(occurrence, _memory_scope=malformed)
    object.__setattr__(occurrence, "_memory_scope", malformed)
    sequence = state.next_sequence
    with pytest.raises(TypeError):
        commit_prepared(log, state, prepared)
    assert state.next_sequence == state.trace.next_sequence == log.next_sequence == sequence


def _prepared_scoped_outcome(kind, *, scoped=True):
    """Prepare one physical outcome using the existing direct reducer fixture."""
    from copy import deepcopy
    from yian.compile.model import END_ID
    from yian.diagnostics import Failure
    from yian.memory import MemoryScope
    from yian.nodes.base import Grow
    from yian.runtime.state import prepare_expanded, prepare_node_failed, prepare_node_routed, prepare_node_succeeded
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _forward_capture
    from tests.engine.test_runtime_memory_learning import _start_trace_node
    from tests.engine.test_run_state_growth import _fragment

    class Spawner(FuncNode):
        is_spawner = True

    node = (Spawner if kind == "growth" else FuncNode)("physical", lambda _: "done")
    node.memory_scope = MemoryScope((node.id,), ("memory",)) if scoped else None
    state, log = _start_trace_node(node, keys=("memory",), learnable=True,
                                   route_handle="go" if kind == "route" else None)
    original = deepcopy(state.flow)
    kwargs = {"forward_capture": _forward_capture(node, {})}
    builder = lambda spawner, _seed: _fragment(spawner.id)
    if kind == "success":
        prepared = prepare_node_succeeded(state, node.id, "done", commit_as=END_ID, **kwargs)
    elif kind == "route":
        prepared = prepare_node_routed(state, node.id, "go", **kwargs)
    elif kind == "failure":
        prepared = prepare_node_failed(state, Failure("node.execute", "failed", node.id), **kwargs)
    else:
        prepared = prepare_expanded(state, node.id, Grow(seed={}), builder, **kwargs)
    return node, state, log, prepared, original, builder


def test_terminal_trace_output_must_match_the_prepared_outcome_before_append():
    from dataclasses import replace
    from yian.runtime.state import commit_prepared
    from yian.typesys.values import ANY_VALUE_ADAPTER
    from tests.engine.test_run_state_outcomes import _state_snapshot

    _node, state, log, prepared, _original, _builder = _prepared_scoped_outcome("success")
    delta = prepared._plan.trace_delta
    forged = replace(delta.terminal, output_bytes=ANY_VALUE_ADAPTER.dump_json(
        canonical_typed("forged", None)[1], warnings="error",
    ))
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, terminal=forged)))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises(ValueError, match="terminal trace"):
        commit_prepared(log, state, prepared)

    assert tuple(log) == records
    assert _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace
    assert log.next_sequence == state.next_sequence == state.trace.next_sequence


def _prepared_terminal_route(*, sibling=False):
    from copy import deepcopy
    from yian.compile.model import START_ID
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded, prepare_node_routed
    from tests.engine._graph_builder import _engine, _forward_capture
    from tests.engine.test_run_state_outcomes import _terminal_skip_route_flow

    engine = _engine(_terminal_skip_route_flow(sibling=sibling), learnable=True)
    state, log = engine.state, engine.log
    original = deepcopy(state.flow)
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(
        state, START_ID, {}, forward_capture=_forward_capture(state.flow.nodes[START_ID], {}),
    ))
    for node_id in (("router", "unrelated") if sibling else ("router",)):
        commit_prepared(log, state, prepare_node_started(state, node_id))
    prepared = prepare_node_routed(
        state, "router", "go", forward_capture=_forward_capture(state.flow.nodes["router"], {}),
    )
    return state, log, prepared, original


def _prepared_trace_frontier(kind):
    from copy import deepcopy
    from yian.compile.model import START_ID
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded, prepare_run_resumed
    from tests.engine._graph_builder import _engine, _forward_capture
    from tests.engine.test_run_state_outcomes import _terminal_skip_route_flow

    engine = _engine(_terminal_skip_route_flow(), learnable=True)
    state, log = engine.state, engine.log
    original = deepcopy(state.flow)
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(
        state, START_ID, {}, forward_capture=_forward_capture(state.flow.nodes[START_ID], {}),
    ))
    prepared = prepare_node_started(state, "router")
    if kind == "resumed":
        commit_prepared(log, state, prepared)
        prepared = prepare_run_resumed(state, num_workers=0)
    return state, log, prepared, original


def test_frontier_trace_cannot_inject_a_causal_terminal_source_before_append():
    from dataclasses import replace
    from yian.runtime.state import commit_prepared
    from tests.engine.test_run_state_outcomes import _state_snapshot

    state, log, prepared, _original = _prepared_trace_frontier("started")
    delta = replace(prepared._plan.trace_delta, causal_route_source=SourceRef(state.run_id, 0))
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=delta))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises(ValueError, match="frontier trace"):
        commit_prepared(log, state, prepared)

    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace
    assert log.next_sequence == state.next_sequence == state.trace.next_sequence


def test_outcome_trace_source_must_match_the_physical_accepted_start_before_append():
    from dataclasses import replace
    from yian.runtime.state import commit_prepared
    from tests.engine.test_run_state_outcomes import _state_snapshot

    _node, state, log, prepared, _original, _builder = _prepared_scoped_outcome("failure")
    delta = prepared._plan.trace_delta
    occurrence = replace(delta.occurrence, source=SourceRef(state.run_id, 0))
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, occurrence=occurrence)))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises(ValueError, match="physical.*occurrence|occurrence.*physical"):
        commit_prepared(log, state, prepared)

    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace
    assert log.next_sequence == state.next_sequence == state.trace.next_sequence


@pytest.mark.parametrize("kind", ["started", "resumed"])
@pytest.mark.parametrize("field", [
    "missing", "sequence", "initialized", "started", "resumed", "occurrence", "handler", "writer",
    "prune_ids", "connections", "consumed_connection", "boundaries", "boundary_update",
    "root_start_source", "root_writer", "causal_route_source", "terminal",
])
def test_frontier_trace_rejects_inappropriate_fields_before_append(kind, field):
    from dataclasses import replace
    from yian.runtime.state import commit_prepared
    from yian.runtime.trace import TraceTerminal
    from tests.engine.test_run_state_outcomes import _state_snapshot

    state, log, prepared, _original = _prepared_trace_frontier(kind)
    delta = prepared._plan.trace_delta

    class EmptyChanges(tuple):
        def __iter__(self):
            pytest.fail("iterated forged frontier changes")

    class UnavailableChange:
        def __iter__(self):
            pytest.fail("iterated forged frontier authority")

    writer = next(iter(state.trace.current_writers.items()))
    values = {
        "sequence": float(delta.sequence), "initialized": state.trace,
        "started": list(delta.started) if delta.started is not None else ("router", delta.sequence),
        "resumed": int(delta.resumed), "occurrence": state.trace.occurrences[-1],
        "handler": lambda *_: pytest.fail("invoked a forged frontier handler"), "writer": writer,
        "prune_ids": EmptyChanges(), "connections": EmptyChanges(), "boundaries": EmptyChanges(),
        "consumed_connection": UnavailableChange(), "boundary_update": UnavailableChange(),
        "root_start_source": SourceRef(state.run_id, 0), "root_writer": writer[1],
        "causal_route_source": SourceRef(state.run_id, 0),
        "terminal": TraceTerminal(True, writer[1].source, writer[1]._value_bytes, None),
    }
    delta = None if field == "missing" else replace(delta, **{field: values[field]})
    changed = replace(prepared, _plan=replace(prepared._plan, trace_delta=delta))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises(ValueError, match="frontier trace"):
        commit_prepared(log, state, changed)

    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace
    assert type(state.trace.next_sequence) is int
    assert log.next_sequence == state.next_sequence == state.trace.next_sequence


@pytest.mark.parametrize("change", ["node", "sequence", "float_sequence", "tuple_subclass"])
def test_frontier_trace_started_identity_is_exact(change):
    from dataclasses import replace
    from yian.compile.model import START_ID
    from yian.runtime.state import commit_prepared
    from tests.engine.test_run_state_outcomes import _state_snapshot

    state, log, prepared, _original = _prepared_trace_frontier("started")
    delta = prepared._plan.trace_delta

    class StartedAlias(tuple):
        def __iter__(self):
            return iter((START_ID, 0))

    values = {
        "node": (START_ID, delta.sequence), "sequence": ("router", 0),
        "float_sequence": ("router", float(delta.sequence)), "tuple_subclass": StartedAlias(delta.started),
    }
    changed = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, started=values[change])))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)
    with pytest.raises(ValueError, match="frontier trace start"):
        commit_prepared(log, state, changed)
    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace


@pytest.mark.parametrize("kind", ["success", "route", "failure", "growth"])
@pytest.mark.parametrize("method", ["replace", "mutate"])
@pytest.mark.parametrize("change", ["source", "source_float", "node", "sequence", "float_sequence", "record"])
def test_outcome_trace_occurrence_identity_matches_the_canonical_record(kind, method, change):
    from copy import deepcopy
    from dataclasses import replace
    from yian.compile.model import START_ID
    from yian.runtime.state import commit_prepared
    from tests.engine.test_run_state_outcomes import _state_snapshot

    _node, state, log, prepared, _original, _builder = _prepared_scoped_outcome(kind)
    delta = prepared._plan.trace_delta
    occurrence = deepcopy(delta.occurrence)
    if change == "source":
        object.__setattr__(occurrence, "source", SourceRef(state.run_id, 0))
    elif change == "source_float":
        object.__setattr__(occurrence.source, "execution_seq", float(occurrence.source.execution_seq))
    elif change == "node":
        object.__setattr__(occurrence, "node_id", START_ID)
    elif change == "sequence":
        object.__setattr__(occurrence, "outcome_seq", occurrence.outcome_seq + 1)
    elif change == "float_sequence":
        object.__setattr__(occurrence, "outcome_seq", float(occurrence.outcome_seq))
    else:
        object.__setattr__(occurrence, "_record_bytes", encode_record(tuple(log)[-1]))
    if method == "replace":
        changed = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, occurrence=occurrence)))
    else:
        object.__setattr__(delta, "occurrence", occurrence)
        changed = prepared
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises(ValueError, match="physical trace occurrence|source execution_seq"):
        commit_prepared(log, state, changed)

    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace
    assert log.next_sequence == state.next_sequence == state.trace.next_sequence


@pytest.mark.parametrize("kind", ["success", "route", "failure", "growth"])
@pytest.mark.parametrize("field", ["initialized", "started", "resumed", "sequence"])
def test_outcome_trace_cannot_override_dispatch_authority(kind, field):
    from dataclasses import replace
    from yian.runtime.state import commit_prepared
    from tests.engine.test_run_state_outcomes import _state_snapshot

    _node, state, log, prepared, _original, _builder = _prepared_scoped_outcome(kind)
    delta = prepared._plan.trace_delta
    value = {
        "initialized": state.trace, "started": ("physical", 0),
        "resumed": True, "sequence": float(delta.sequence),
    }[field]
    changed = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, **{field: value})))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)
    with pytest.raises(ValueError, match="outcome trace"):
        commit_prepared(log, state, changed)
    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace


@pytest.mark.parametrize("change", ["target", "source", "sequence", "float_sequence", "value", "missing", "tuple"])
def test_outcome_trace_writer_matches_the_same_accepted_physical_output(change):
    from dataclasses import replace
    from yian.runtime.state import commit_prepared, prepare_node_succeeded
    from tests.engine._graph_builder import _forward_capture
    from tests.engine.test_run_state_outcomes import _state_snapshot

    node, state, log, _prepared, _original, _builder = _prepared_scoped_outcome("success")
    prepared = prepare_node_succeeded(state, node.id, "done", forward_capture=_forward_capture(node, {}))
    delta = prepared._plan.trace_delta
    target, writer = delta.writer
    values = {
        "target": (state.flow.start_id, writer), "source": (target, replace(writer, source=SourceRef(state.run_id, 0))),
        "sequence": (target, replace(writer, outcome_seq=writer.outcome_seq + 1)),
        "float_sequence": (target, replace(writer, outcome_seq=float(writer.outcome_seq))),
        "value": (target, replace(writer, _value_bytes=next(iter(state.trace.sources.values()))._value_bytes)),
        "missing": None, "tuple": [target, writer],
    }
    changed = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, writer=values[change])))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)
    with pytest.raises(ValueError, match="physical trace writer"):
        commit_prepared(log, state, changed)
    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace


@pytest.mark.parametrize("kind", ["started", "resumed"])
def test_frontier_trace_and_terminal_source_match_live_replay_without_repreparing(kind, monkeypatch):
    from copy import deepcopy
    from dataclasses import replace
    from yian.compile.expand import build_grow
    from yian.runtime.backward import compute_backward
    import yian.runtime.state as state_module
    from tests.engine._graph_builder import _forward_capture

    state, log, prepared, original = _prepared_trace_frontier(kind)
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(prepared._plan.trace_delta)))
    with monkeypatch.context() as patch:
        patch.setattr(state_module, "_prepare_record_trace", lambda *_a, **_k: pytest.fail("trace was reprepared"))
        state_module.commit_prepared(log, state, prepared)
    if kind == "resumed":
        state_module.commit_prepared(log, state, state_module.prepare_node_started(state, "router"))
    physical = SourceRef(state.run_id, state.trace.started["router"])
    state_module.commit_prepared(log, state, state_module.prepare_node_routed(
        state, "router", "go", forward_capture=_forward_capture(state.flow.nodes["router"], {}),
    ))
    assert state.trace.terminal.source == physical and physical.execution_seq > 0
    root, results = compute_backward(state.trace, "Review.")
    assert next(item for item in results if item.source == physical).invoked
    replay = state_module._replay_records(state_module.RunState(deepcopy(original)), tuple(log), grow_builder=build_grow)
    assert _trace_snapshot(replay.trace) == _trace_snapshot(state.trace)
    assert compute_backward(replay.trace, "Review.") == (root, results)


@pytest.mark.parametrize("kind", ["success", "route", "failure"])
@pytest.mark.parametrize("method", ["replace", "mutate"])
@pytest.mark.parametrize("change", [
    "none", "carrier", "source_none", "source_sequence", "source_run", "source_float",
    "succeeded", "disposition", "output", "failure",
])
def test_terminal_trace_rejects_forged_fields_before_append(kind, method, change):
    from copy import deepcopy
    from dataclasses import replace
    from yian.diagnostics import Failure
    from yian.runtime.state import commit_prepared
    from yian.runtime.trace import TraceTerminal
    from yian.typesys.values import ANY_VALUE_ADAPTER
    from tests.engine.test_run_state_outcomes import _state_snapshot

    if kind == "route":
        state, log, prepared, _original = _prepared_terminal_route()
    else:
        _node, state, log, prepared, _original, _builder = _prepared_scoped_outcome(kind)
    delta = prepared._plan.trace_delta
    forged = deepcopy(delta.terminal)
    if change == "none":
        forged = None
    elif change == "carrier":
        forged = object()
    elif change.startswith("source_"):
        source = forged.source
        if change == "source_none":
            source = None
        elif change == "source_sequence":
            source = SourceRef(state.run_id, 0)
        elif change == "source_run":
            source = SourceRef("foreign-run", source.execution_seq)
        else:
            object.__setattr__(source, "execution_seq", float(source.execution_seq))
        object.__setattr__(forged, "source", source)
    elif change == "succeeded":
        object.__setattr__(forged, "succeeded", int(forged.succeeded))
    elif change == "disposition":
        value_bytes = ANY_VALUE_ADAPTER.dump_json(canonical_typed("forged", None)[1], warnings="error")
        forged = TraceTerminal(
            not forged.succeeded, forged.source,
            None if forged.succeeded else value_bytes,
            Failure("node.execute", "forged", "router") if forged.succeeded else None,
        )
    elif change == "output":
        object.__setattr__(forged, "output_bytes", ANY_VALUE_ADAPTER.dump_json(
            canonical_typed("forged", None)[1], warnings="error",
        ))
    else:
        object.__setattr__(forged, "failure", Failure("node.execute", "forged", "router"))
    if method == "replace":
        prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, terminal=forged)))
    else:
        object.__setattr__(delta, "terminal", forged)
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises((TypeError, ValueError)):
        commit_prepared(log, state, prepared)

    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace
    assert log.next_sequence == state.next_sequence == state.trace.next_sequence


@pytest.mark.parametrize("kind", ["success", "route", "growth"])
@pytest.mark.parametrize("field", ["terminal", "root_writer", "causal_route_source", "root_start_source"])
def test_nonterminal_outcome_rejects_fabricated_terminal_trace_authority(kind, field):
    from dataclasses import replace
    from yian.diagnostics import Failure
    from yian.runtime.state import commit_prepared, prepare_node_succeeded
    from yian.runtime.trace import TraceTerminal
    from tests.engine._graph_builder import _forward_capture
    from tests.engine.test_run_state_outcomes import _state_snapshot

    node, state, log, prepared, _original, _builder = _prepared_scoped_outcome(kind)
    if kind == "success":
        prepared = prepare_node_succeeded(state, node.id, "done", forward_capture=_forward_capture(node, {}))
    delta = prepared._plan.trace_delta
    assert delta.terminal is None
    source = delta.occurrence.source
    forged = {
        "terminal": TraceTerminal(False, source, None, Failure("node.execute", "forged", node.id)),
        "root_writer": next(iter(state.trace.current_writers.values())),
        "causal_route_source": source,
        "root_start_source": source,
    }[field]
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, **{field: forged})))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises(ValueError, match="terminal trace"):
        commit_prepared(log, state, prepared)

    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace


@pytest.mark.parametrize("change", ["none", "source", "sequence", "float_sequence", "value"])
def test_terminal_trace_root_writer_must_be_the_physical_accepted_output(change):
    from dataclasses import replace
    from yian.runtime.state import commit_prepared
    from tests.engine.test_run_state_outcomes import _state_snapshot

    _node, state, log, prepared, _original, _builder = _prepared_scoped_outcome("success")
    delta = prepared._plan.trace_delta
    writer = delta.root_writer
    if change == "none":
        forged = None
    elif change == "source":
        forged = replace(writer, source=SourceRef(state.run_id, 0))
    elif change == "sequence":
        forged = replace(writer, outcome_seq=writer.outcome_seq + 1)
    elif change == "float_sequence":
        forged = replace(writer, outcome_seq=float(writer.outcome_seq))
    else:
        forged = replace(writer, _value_bytes=next(iter(state.trace.current_writers.values()))._value_bytes)
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, root_writer=forged)))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises(ValueError, match="terminal trace root writer"):
        commit_prepared(log, state, prepared)

    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace


@pytest.mark.parametrize("change", ["none", "sequence", "run", "float_sequence"])
def test_terminal_trace_root_start_latch_must_match_the_accepted_start(change):
    from dataclasses import replace
    from yian.compile.model import START_ID, END_ID
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from tests.engine._graph_builder import _engine, _forward_capture, _graph
    from tests.engine.test_run_state_outcomes import _state_snapshot

    engine = _engine(_graph([], [(START_ID, END_ID)]), learnable=True)
    state, log = engine.state, engine.log
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    prepared = prepare_node_succeeded(
        state, START_ID, {}, forward_capture=_forward_capture(state.flow.nodes[START_ID], {}),
    )
    delta = prepared._plan.trace_delta
    source = delta.root_start_source
    if change == "none":
        forged = None
    elif change == "sequence":
        forged = SourceRef(state.run_id, 0)
    elif change == "run":
        forged = SourceRef("foreign-run", source.execution_seq)
    else:
        forged = replace(source)
        object.__setattr__(forged, "execution_seq", float(source.execution_seq))
    changed = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(delta, root_start_source=forged)))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)

    with pytest.raises(ValueError):
        commit_prepared(log, state, changed)

    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace
    commit_prepared(log, state, prepared)
    assert state.trace.root_start_source == source


@pytest.mark.parametrize("kind", ["success", "route", "failure"])
def test_terminal_trace_accepts_detached_authority_without_repreparing(kind, monkeypatch):
    from copy import deepcopy
    from dataclasses import replace
    from yian.compile.expand import build_grow
    import yian.runtime.state as state_module

    if kind == "route":
        state, log, prepared, original = _prepared_terminal_route()
        builder = build_grow
    else:
        _node, state, log, prepared, original, builder = _prepared_scoped_outcome(kind)
    delta = prepared._plan.trace_delta
    detached = replace(delta, terminal=deepcopy(delta.terminal), root_writer=deepcopy(delta.root_writer),
                       root_start_source=deepcopy(delta.root_start_source),
                       causal_route_source=deepcopy(delta.causal_route_source))
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=detached))
    with monkeypatch.context() as patch:
        patch.setattr(state_module, "_prepare_record_trace", lambda *_a, **_k: pytest.fail("trace was reprepared"))
        patch.setattr(state_module, "_prepare_terminal_delta", lambda *_a, **_k: pytest.fail("terminal was rederived"))
        state_module.commit_prepared(log, state, prepared)
    assert state.trace.terminal == detached.terminal
    replay = state_module._replay_records(state_module.RunState(original), tuple(log), grow_builder=builder)
    assert _trace_snapshot(replay.trace) == _trace_snapshot(state.trace)
    assert replay.result == state.result


@pytest.mark.parametrize("explicit_failure", [False, True])
def test_terminal_trace_uses_the_accepted_route_latch_until_an_explicit_failure(explicit_failure):
    from copy import deepcopy
    from dataclasses import replace
    from yian.compile.expand import build_grow
    from yian.diagnostics import Failure
    from yian.runtime.state import RunState, _replay_records, commit_prepared, prepare_node_failed, prepare_node_succeeded
    from tests.engine._graph_builder import _forward_capture
    from tests.engine.test_run_state_outcomes import _state_snapshot

    state, log, route, original = _prepared_terminal_route(sibling=True)
    delta = route._plan.trace_delta
    assert delta.terminal is None
    causal = delta.causal_route_source
    sibling = SourceRef(state.run_id, state.trace.started["unrelated"])
    for source in (None, sibling):
        forged = replace(route, _plan=replace(route._plan, trace_delta=replace(delta, causal_route_source=source)))
        records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)
        with pytest.raises(ValueError, match="terminal trace causal source"):
            commit_prepared(log, state, forged)
        assert tuple(log) == records and _state_snapshot(state) == before_state
        assert _trace_snapshot(state.trace) == before_trace
    commit_prepared(log, state, route)
    capture = _forward_capture(state.flow.nodes["unrelated"], {})
    if explicit_failure:
        final = prepare_node_failed(state, Failure("node.execute", "later failure", "unrelated"), forward_capture=capture)
    else:
        final = prepare_node_succeeded(state, "unrelated", "done", forward_capture=capture)
    expected_source, wrong_source = (sibling, causal) if explicit_failure else (causal, sibling)
    terminal = final._plan.trace_delta.terminal
    forged = replace(final, _plan=replace(final._plan, trace_delta=replace(
        final._plan.trace_delta, terminal=replace(terminal, source=wrong_source),
    )))
    records, before_state, before_trace = tuple(log), _state_snapshot(state), _trace_snapshot(state.trace)
    with pytest.raises(ValueError, match="terminal trace"):
        commit_prepared(log, state, forged)
    assert tuple(log) == records and _state_snapshot(state) == before_state
    assert _trace_snapshot(state.trace) == before_trace
    commit_prepared(log, state, replace(final, _plan=replace(final._plan, trace_delta=replace(
        final._plan.trace_delta, terminal=deepcopy(terminal),
    ))))
    assert state.trace.terminal.source == expected_source
    assert state.trace.terminal.failure == state.result.failure
    for length in range(1, len(tuple(log)) + 1):
        replay = _replay_records(RunState(deepcopy(original)), tuple(log)[:length], grow_builder=build_grow)
        assert replay.trace.terminal == (state.trace.terminal if length == len(tuple(log)) else None)


@pytest.mark.parametrize("kind", ["success", "route", "failure", "growth"])
@pytest.mark.parametrize("method", ["replace", "mutate"])
@pytest.mark.parametrize("change", ["none", "foreign_path", "foreign_memory"])
def test_outcome_rejects_valid_scope_substitution_before_append(kind, method, change):
    from dataclasses import replace
    from yian.memory import MemoryScope
    from yian.runtime.log import encode_pool
    from yian.runtime.state import commit_prepared

    node, state, log, prepared, _original, _builder = _prepared_scoped_outcome(kind)
    scope = node.memory_scope
    if change == "none":
        forged = None
    elif change == "foreign_path":
        forged = MemoryScope(("other",), scope.memories)
    else:
        forged = MemoryScope(scope.static_path, ("foreign",))
    delta = prepared._plan.trace_delta
    if method == "replace":
        prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(
            delta, occurrence=replace(delta.occurrence, _memory_scope=forged),
        )))
    else:
        object.__setattr__(delta.occurrence, "_memory_scope", forged)
    accepted, pool, nodes = tuple(log), encode_pool(state.pool), state.flow.nodes
    sequence = state.next_sequence
    with pytest.raises(ValueError, match="physical.*memory scope|memory scope.*physical"):
        commit_prepared(log, state, prepared)
    assert tuple(log) == accepted and encode_pool(state.pool) == pool and state.flow.nodes is nodes
    assert state.next_sequence == state.trace.next_sequence == log.next_sequence == sequence


@pytest.mark.parametrize("kind", ["success", "route", "failure", "growth"])
@pytest.mark.parametrize("change", ["display_node", "delta", "occurrence"])
def test_outcome_scope_authority_cannot_be_selected_or_omitted_by_trace_delta(kind, change):
    from dataclasses import replace
    from yian.compile.model import START_ID
    from yian.runtime.state import commit_prepared

    _node, state, log, prepared, _original, _builder = _prepared_scoped_outcome(kind)
    delta = prepared._plan.trace_delta
    if change == "display_node":
        delta = replace(delta, occurrence=replace(delta.occurrence, node_id=START_ID, _memory_scope=None))
    elif change == "delta":
        delta = None
    else:
        delta = replace(delta, occurrence=None)
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=delta))
    sequence = state.next_sequence
    with pytest.raises(ValueError):
        commit_prepared(log, state, prepared)
    assert state.next_sequence == state.trace.next_sequence == log.next_sequence == sequence


@pytest.mark.parametrize("kind", ["success", "route", "failure", "growth"])
@pytest.mark.parametrize("scoped", [False, True])
def test_outcome_accepts_detached_physical_scope_and_replays_it(kind, scoped):
    from copy import deepcopy
    from dataclasses import replace
    from yian.runtime.state import RunState, _replay_records, commit_prepared

    node, state, log, prepared, original, builder = _prepared_scoped_outcome(kind, scoped=scoped)
    delta = prepared._plan.trace_delta
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(
        delta, occurrence=replace(delta.occurrence, _memory_scope=deepcopy(node.memory_scope)),
    )))
    commit_prepared(log, state, prepared)
    occurrence = state.trace.occurrences[-1]
    assert occurrence.memory_scope == node.memory_scope
    replay = _replay_records(RunState(original), tuple(log), grow_builder=builder)
    assert replay.trace.occurrence(occurrence.source).memory_scope == occurrence.memory_scope


def test_learn_false_omits_backward_capture():
    from yian.compile.model import START_ID, END_ID
    from yian.runtime.log import NodeSucceededRecord
    from tests.engine._graph_builder import _engine, _graph

    flow = _graph([], [(START_ID, END_ID)])
    engine = _engine(flow, learnable=False)
    list(engine.run())
    assert engine.state.trace is None
    assert all(node.execution_scope is None for node in flow.nodes.values())
    for envelope in engine.log:
        if isinstance(envelope.record, NodeSucceededRecord):
            assert envelope.record.forward_capture is None


@pytest.mark.parametrize("path,value,present,source_path", [
    ("input.x", None, True, ("x",)), ("input.absent", None, False, None),
    ("input.x.more", None, False, None), ("input.items.0", None, False, None),
    ("__start__.output", {"x": None, "items": [1]}, True, ()),
    ("system.run_id.ignored", "direct-engine-test", True, None),
    ("system", None, False, None), ("input", None, False, None),
    (" input.x ", None, True, ("x",)), ("input. x", None, False, None),
])
def test_paired_resolver_preserves_pool_namespaces(path, value, present, source_path):
    from yian.compile.model import START_ID, END_ID
    from tests.engine._graph_builder import _engine, _graph

    engine = _engine(_graph([], [(START_ID, END_ID)]), {"x": None, "items": [1]}, learnable=True)
    resolved = engine.state.trace.resolve(path)
    assert resolved.value == value
    assert resolved.present is present
    assert resolved.source == (None if source_path is None else SourceRef(engine.state.run_id, 0, source_path))
    head, *rest = path.strip().split(".")
    assert resolved.value == engine.state.pool.resolve(head, rest)


def test_reducer_requires_capture_exactly_when_learning():
    from yian.compile.model import START_ID, END_ID
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from tests.engine._graph_builder import _engine, _forward_capture, _graph

    for learnable in (False, True):
        engine = _engine(_graph([], [(START_ID, END_ID)]), learnable=learnable)
        state = engine.state
        commit_prepared(engine.log, state, prepare_node_started(state, START_ID))
        capture = None if learnable else _forward_capture(state.flow.nodes[START_ID], {})
        with pytest.raises(ValueError, match="capture"):
            prepare_node_succeeded(state, START_ID, {}, forward_capture=capture)


def _trace_snapshot(trace):
    return (
        trace.end_id, trace.root_input_names, trace.root_start_source, trace.terminal,
        trace._root_writer, trace._causal_route_source,
        trace.occurrences, dict(trace.sources), dict(trace.current_writers),
        dict(trace.started), trace.next_sequence, trace.boundaries, dict(trace.pending_connections),
        {source: (handler.__self__.id, handler.__self__.execution_scope,
                  handler.__func__.__qualname__)
         for source, handler in trace.backward_handlers.items()},
    )


def _seed_call_flow(sources=None, extra_nodes=(), *, extra_first=False):
    from yian.compile.model import START_ID, END_ID, CompiledFlow, Edge
    from yian.nodes.call import CallNode
    from yian.nodes.contracts import Parameter
    from yian.nodes.end import EndNode
    from yian.nodes.start import StartNode
    from tests.engine._graph_builder import _graph

    sources = {"x": "${input.seed}"} if sources is None else sources
    parameters = [Parameter(name) for name in sources]
    child = CompiledFlow({
        START_ID: StartNode(START_ID, params=parameters),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }, [Edge("order", START_ID, END_ID, ordering=True)], wiring={
        START_ID: {name: "${input." + name + "}" for name in sources}, END_ID: {},
    })
    node = CallNode("spawn", flow_id="child", child=child)
    node.params = parameters
    entries = [(START_ID, item.id) for item in extra_nodes]
    entries.insert(len(entries) if extra_first else 0, (START_ID, "spawn"))
    return _graph([node, *extra_nodes], [*entries, ("spawn", END_ID)], wiring={"spawn": sources})


def _nested_growth_flow(kind):
    from yian.compose import load_flow

    over = "    over: '${[input.n, input.n]}'\n" if kind == "map" else ""
    seed = "${item}" if kind == "map" else "${input.n}"
    return load_flow(f"""
id: nested-seeds
name: nested-seeds
input: {{n: int}}
defs:
  leaf:
    input: {{n: int}}
    nodes:
      work:
        kind: code
        input: {{n: '${{input.n}}'}}
        code: tests.engine._compose_codefns:loop_countdown
        output: {{n: int}}
    output: ${{work.output}}
  body:
    input: {{n: int}}
    nodes:
      invoke:
        kind: call
        call: leaf
        input: {{n: '${{input.n}}'}}
    output: ${{invoke.output}}
  repeated:
    input: {{n: int}}
    nodes:
      repeat:
        kind: loop
        call: body
        times: 2
        input: {{n: '${{input.n}}'}}
    output: ${{repeat.output}}
nodes:
  outer:
    kind: {kind}
    call: repeated
{over}    input: {{n: '{seed}'}}
output: ${{outer.output}}
""").compiled


@pytest.mark.parametrize("kind", ["call", "map"])
def test_nested_growth_preserves_authored_paths_and_local_driver_zero(kind):
    from yian.compile.model import START_ID
    from yian.nodes.contracts import ExecutionPosition
    from tests.engine._graph_builder import _engine

    engine = _engine(_nested_growth_flow(kind), {"n": 9}, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    occurrences = {item.node_id: item for item in trace.occurrences}
    loop_path = ("outer", "repeat")
    invoke_path = (*loop_path, "invoke")
    for item in range(2 if kind == "map" else 1):
        prefix = f"outer#{item}" if kind == "map" else "outer"
        outer = ExecutionPosition("map_item", ("outer",), item) if kind == "map" else ExecutionPosition("call", ("outer",))
        for iteration in range(3):
            driver_id = f"{prefix}/repeat" + (f"~{iteration}" if iteration else "")
            driver = occurrences[driver_id]
            assert driver.record.forward_capture.scope.static_path == loop_path
            assert driver.record.forward_capture.scope.occurrence_path == (
                outer, ExecutionPosition("loop_driver", loop_path, iteration),
            )
        for iteration in range(2):
            body_start = occurrences[f"{prefix}/repeat#{iteration}/{START_ID}"]
            call_id = f"{prefix}/repeat#{iteration}/invoke"
            leaf_start = occurrences[f"{call_id}/{START_ID}"]
            position = (outer, ExecutionPosition("loop_body", loop_path, iteration))
            assert body_start.record.forward_capture.scope.occurrence_path == position
            assert leaf_start.record.forward_capture.scope.occurrence_path == (
                *position, ExecutionPosition("call", invoke_path),
            )
            assert leaf_start.record.forward_capture.scope.flow_path == invoke_path
            assert leaf_start.input_route.children[0].source == SourceRef(
                body_start.source.run_id, body_start.source.execution_seq, ("n",),
            )
        driver_node = trace.backward_handlers[occurrences[f"{prefix}/repeat"].source].__self__
        assert driver_node.child.nodes["invoke"].execution_scope.occurrence_path == ()
    assert all(boundary.start_source is not None and boundary.end_source is not None for boundary in trace.boundaries)


@pytest.mark.parametrize("path", [("absent",), (9, "x")])
def test_missing_seed_selector_is_an_accepted_growth_failure(path):
    from yian.compile.expand import build_grow
    from yian.nodes.contracts import SeedConnection
    from yian.runtime.log import NodeFailedRecord
    from tests.engine._graph_builder import _engine

    def invalid_selector(spawner, seed):
        fragment = build_grow(spawner, seed)
        fragment.input_connections = (SeedConnection(fragment.start_id, "x", path),)
        return fragment

    engine = _engine(_seed_call_flow(), {"seed": 7}, learnable=True, grow_builder=invalid_selector)
    list(engine.run())
    record = engine.state.trace.occurrences[-1].record
    assert isinstance(record, NodeFailedRecord)
    assert record.failure.code == "contract.growth"
    assert record.failure == engine.state.result.failure
    assert record.forward_capture.bound_inputs == {"x": 7}
    assert not engine.state.trace.pending_connections and not engine.state.trace.boundaries


@pytest.mark.parametrize("change", ["selector-digest", "missing-field", "missing-item", "mutable-path", "boundary-role"])
def test_growth_recipe_tampering_fails_replay_before_apply(change):
    import copy
    from dataclasses import replace
    from yian.compile.expand import build_grow
    from yian.compile.manifest import expansion_digest
    from yian.compile.model import START_ID
    from yian.nodes.contracts import ChildFlowBoundary, SeedConnection
    from yian.runtime.state import RunState, _replay_records, prepare_record
    from tests.engine._graph_builder import _engine

    engine = _engine(_seed_call_flow(), {"seed": 7}, learnable=True)
    original = copy.deepcopy(engine.state.flow)
    for node_id in (START_ID, "spawn"):
        engine._commit_started(node_id)
        invocation = ConversationInvocation(engine.state.run_id, engine.state.accepted_starts[node_id])
        engine._commit_outcome(node_id, engine._evaluate(node_id, invocation, emit_stream=None))
    record = tuple(engine.log)[-1].record

    def changed_builder(spawner, seed):
        fragment = build_grow(spawner, seed)
        path = (9, "x") if change == "missing-item" else ("absent",)
        fragment.input_connections = (SeedConnection(fragment.start_id, "x", path),)
        if change == "mutable-path":
            object.__setattr__(fragment.input_connections[0], "seed_path", ["x"])
        elif change == "boundary-role":
            fragment.child_boundaries = (ChildFlowBoundary(fragment.end_id, fragment.start_id),)
        return fragment

    if change in ("missing-field", "missing-item"):
        changed = changed_builder(engine.state.flow.nodes["spawn"], {"x": 7})
        record = replace(record, topology_digest=expansion_digest(changed))
    replay = _replay_records(RunState(copy.deepcopy(original)), tuple(engine.log)[:-1], grow_builder=build_grow)
    before = (_trace_snapshot(replay.trace), tuple(replay.flow.nodes), replay.next_sequence)
    with pytest.raises((TypeError, ValueError)):
        prepare_record(replay, record, grow_builder=changed_builder)
    assert (_trace_snapshot(replay.trace), tuple(replay.flow.nodes), replay.next_sequence) == before


@pytest.mark.parametrize("seed,expected,early_peer", [
    ("${peer.output | null}", 11, False),
    ("${peer.output | null}", None, False),
    ("${peer.output}", 11, True),
])
def test_reference_looking_seed_strings_use_actual_forward_binding(seed, expected, early_peer):
    from yian.compile.model import START_ID
    from yian.expr.lineage import route_feedback
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine

    peer = FuncNode("peer", lambda _: expected)
    engine = _engine(_seed_call_flow(extra_nodes=(peer,), extra_first=early_peer), {"seed": seed}, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    occurrences = {item.node_id: item for item in trace.occurrences}
    parent = occurrences["spawn"]
    child = occurrences[f"spawn/{START_ID}"]
    assert parent.record.forward_capture.bound_inputs == {"x": seed}
    capture = child.record.forward_capture
    assert capture.bound_inputs == {"x": expected}
    assert capture.evaluated_keys == frozenset({"x"})
    assert child.input_route == capture.route
    if early_peer:
        assert occurrences["peer"].outcome_seq < parent.outcome_seq < child.outcome_seq
    else:
        assert parent.outcome_seq < occurrences["peer"].outcome_seq < child.outcome_seq
    routed = route_feedback(child.input_route, {"x": "Change."}, trace.lookup_source)
    assert [(item.source, item.feedback) for item in routed.contributions] == (
        [(occurrences["peer"].source, "Change.")] if expected is not None else []
    )
    if expected is None:
        assert capture.route.children[0].kind == "literal"


def test_incomplete_child_binding_joins_only_safe_captured_fields():
    from yian.compile.model import START_ID
    from yian.runtime.log import NodeFailedRecord
    from tests.engine._graph_builder import _engine

    engine = _engine(_seed_call_flow({"safe": "${input.safe}", "x": "${input.seed}"}),
                     {"safe": 7, "seed": '${null :? "missing"}'}, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    child = trace.occurrences[-1]
    assert child.node_id == f"spawn/{START_ID}", child.record
    assert isinstance(child.record, NodeFailedRecord)
    assert child.record.failure == engine.state.result.failure
    assert child.record.forward_capture.bound_inputs is None
    assert child.input_route.names == child.record.forward_capture.route.names == ("safe",)
    root = next(item for item in trace.occurrences if item.node_id == START_ID)
    assert child.input_route.children[0].source == SourceRef(root.source.run_id, root.source.execution_seq, ("safe",))
    boundary, = trace.boundaries
    assert boundary.start_source == child.source and boundary.end_source is None
    assert not trace.pending_connections


def test_seed_connections_wait_for_append_and_survive_unfinished_start_retry(monkeypatch):
    from yian.compile.model import START_ID
    from yian.runtime.state import commit_prepared, prepare_expanded, prepare_run_resumed
    from yian.runtime.trace import capture_bindings
    from tests.engine._graph_builder import _engine

    engine = _engine(_seed_call_flow(), {"seed": 7}, learnable=True)
    engine._commit_started(START_ID)
    invocation = ConversationInvocation(engine.state.run_id, engine.state.accepted_starts[START_ID])
    engine._commit_outcome(START_ID, engine._evaluate(START_ID, invocation, emit_stream=None))
    engine._commit_started("spawn")
    invocation = ConversationInvocation(engine.state.run_id, engine.state.accepted_starts["spawn"])
    evaluated = engine._evaluate("spawn", invocation, emit_stream=None)
    prepared = prepare_expanded(
        engine.state, "spawn", evaluated.outcome, engine.grow_builder,
        forward_capture=capture_bindings(evaluated.bindings, evaluated.backward_capture),
    )
    before = (_trace_snapshot(engine.state.trace), tuple(engine.state.flow.nodes), engine.state.next_sequence)
    original_append = engine.log.append

    def reject(_):
        raise OSError("append failed")

    monkeypatch.setattr(engine.log, "append", reject)
    with pytest.raises(OSError, match="append failed"):
        commit_prepared(engine.log, engine.state, prepared)
    assert (_trace_snapshot(engine.state.trace), tuple(engine.state.flow.nodes), engine.state.next_sequence) == before
    monkeypatch.setattr(engine.log, "append", original_append)
    commit_prepared(engine.log, engine.state, prepared)
    target = f"spawn/{START_ID}"
    pending = engine.state.trace.pending_connections[target]
    engine._commit_started(target)
    abandoned = engine.state.trace.started[target]
    commit_prepared(engine.log, engine.state, prepare_run_resumed(engine.state, num_workers=0))
    assert engine.state.trace.pending_connections[target] is pending
    list(engine.run())
    child = next(item for item in engine.state.trace.occurrences if item.node_id == target)
    assert child.source.execution_seq != abandoned
    assert child.input_route.children[0].source is not None
    assert not engine.state.trace.pending_connections


@pytest.mark.parametrize("skip_first", [False, True])
def test_pruned_seed_connections_do_not_leak_into_reused_node_ids(skip_first):
    import copy
    from yian.compile.expand import call_subgraph, clone_child
    from yian.compile.model import START_ID, END_ID, Edge, Flow
    from yian.events import NodeExpanded, RunSucceeded
    from yian.nodes.base import Grow, Node, NodeKind
    from yian.nodes.call import CallNode
    from yian.nodes.case import CaseNode
    from yian.nodes.contracts import ChildFlowBoundary, Parameter, SeedConnection
    from yian.nodes.end import EndNode
    from yian.nodes.start import StartNode
    from tests.engine._graph_builder import _engine, _graph

    shared_start, shared_end = f"shared/{START_ID}", f"shared/{END_ID}"
    child = _seed_call_flow().nodes["spawn"].child
    empty_child = _seed_call_flow({}).nodes["spawn"].child

    class Prune(Node):
        kind = NodeKind.CODE
        is_spawner = True

        def forward(self, inputs):
            return Grow({}, prune=frozenset({shared_start, shared_end}))

    first = CallNode("first", flow_id="child", child=copy.deepcopy(child))
    again = CallNode("again", flow_id="child", child=copy.deepcopy(child))
    first.params = again.params = [Parameter("x")]

    def builder(spawner, seed):
        if spawner.id == "prune":
            return call_subgraph(empty_child, "prune", {})
        if spawner.id == "first" and skip_first:
            finish = EndNode.record("finish", output_names=[])
            finish.commit_as = "first"
            return Flow({
                "entry": StartNode("entry", params=[]), "choice": CaseNode("choice", []),
                shared_start: StartNode(shared_start, params=[Parameter("x")]),
                shared_end: EndNode.record(shared_end, output_names=[]), "finish": finish,
            }, [
                Edge("enter", "entry", "choice", ordering=True),
                Edge("chosen", "choice", "finish", source_handle="default"),
                Edge("skipped", "choice", shared_start, source_handle="other"),
                Edge("child", shared_start, shared_end, ordering=True),
            ], wiring={"entry": {}, "choice": {}, shared_start: {"x": seed["x"]},
                       shared_end: {}, "finish": {}}, start_id="entry", end_id="finish",
                input_connections=(SeedConnection(shared_start, "x", ("x",)),),
                child_boundaries=(ChildFlowBoundary(shared_start, shared_end),))
        fragment = clone_child(spawner.child, "shared", seed)
        fragment.nodes[fragment.end_id].commit_as = spawner.id
        return fragment

    seen = {}

    def observe(event):
        if isinstance(event, NodeExpanded):
            seen[event.node_id] = dict(engine.state.trace.pending_connections)

    flow = _graph([first, Prune("prune"), again], [
        (START_ID, "first"), ("first", "prune"), ("prune", "again"), ("again", END_ID),
    ], wiring={"first": {"x": "${input.first}"}, "prune": {}, "again": {"x": "${input.second}"}})
    engine = _engine(flow, {"first": 7, "second": 11}, learnable=True, grow_builder=builder, on_event=observe)
    list(engine.run())
    assert isinstance(engine.state.result, RunSucceeded)
    trace = engine.state.trace
    assert shared_start in seen["first"] and shared_start not in seen["prune"]
    assert seen["again"][shared_start].owner != seen["first"][shared_start].owner
    reused = [item for item in trace.occurrences if item.node_id == shared_start]
    assert len(reused) == (1 if skip_first else 2)
    root = next(item for item in trace.occurrences if item.node_id == START_ID)
    assert reused[-1].input_route.children[0].source == SourceRef(
        root.source.run_id, root.source.execution_seq, ("second",),
    )
    boundaries = [item for item in trace.boundaries if item.start_id == shared_start]
    assert len(boundaries) == 2 and boundaries[0].owner != boundaries[1].owner
    assert boundaries[0].start_source == (None if skip_first else reused[0].source)
    assert boundaries[1].start_source == reused[-1].source
    assert not trace.pending_connections


def test_call_seed_routes_survive_materialized_values():
    from yian.compile.model import START_ID, END_ID
    from yian.compose import load_flow
    from yian.expr.lineage import route_feedback
    from yian.nodes.contracts import ExecutionPosition
    from tests.engine._graph_builder import _engine

    flow = load_flow("""
id: seed-call
name: seed-call
input: {n: int}
defs:
  child:
    input: {n: int}
    nodes:
      work:
        kind: code
        input: {n: '${input.n}'}
        code: tests.engine._compose_codefns:loop_countdown
        output: {n: int}
    output: ${work.output}
nodes:
  spawn:
    kind: call
    call: child
    input: {n: '${input.n}'}
output: ${spawn.output}
""").compiled
    engine = _engine(flow, {"n": 9}, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    occurrences = {item.node_id: item for item in trace.occurrences}
    child = occurrences[f"spawn/{START_ID}"]
    raw_capture = child.record.forward_capture
    assert raw_capture.bound_inputs == {"n": 9}
    assert raw_capture.route.children[0].kind == "literal"
    routed = route_feedback(child.input_route, {"n": "Use another seed."}, trace.lookup_source)
    assert [(item.source, item.feedback) for item in routed.contributions] == [
        (occurrences[START_ID].source, {"n": "Use another seed."}),
    ]
    assert raw_capture.scope.occurrence_path == (ExecutionPosition("call", ("spawn",)),)
    boundary, = trace.boundaries
    assert boundary.owner == occurrences["spawn"].source
    assert (boundary.start_id, boundary.end_id) == (f"spawn/{START_ID}", f"spawn/{END_ID}")
    assert (boundary.start_source, boundary.end_source) == (
        child.source, occurrences[f"spawn/{END_ID}"].source,
    )
    assert not trace.pending_connections


@pytest.mark.parametrize("kind", ["call", "map"])
def test_null_materialized_seed_keeps_its_selected_source(kind):
    from yian.compile.model import START_ID, END_ID
    from yian.expr.lineage import route_feedback
    from yian.nodes.contracts import Parameter
    from yian.nodes.map import MapNode
    from tests.engine._graph_builder import _engine, _graph

    flow = _seed_call_flow()
    if kind == "map":
        node = MapNode("spawn", flow_id="child", child=flow.nodes["spawn"].child)
        node.params = [Parameter("x")]
        flow = _graph([node], [(START_ID, "spawn"), ("spawn", END_ID)],
                      wiring={"spawn": {"over": "${[input.seed]}", "x": "${item}"}})
    engine = _engine(flow, {"seed": None}, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    root = next(item for item in trace.occurrences if item.node_id == START_ID)
    child_id = f"spawn#0/{START_ID}" if kind == "map" else f"spawn/{START_ID}"
    child = next(item for item in trace.occurrences if item.node_id == child_id)
    assert child.record.forward_capture.bound_inputs == {"x": None}
    assert child.record.forward_capture.route.children[0].kind == "literal"
    routed = route_feedback(child.input_route, {"x": "Fill this value."}, trace.lookup_source)
    assert [(item.source, item.feedback) for item in routed.contributions] == [
        (root.source, {"seed": "Fill this value."}),
    ]


@pytest.mark.parametrize("over,inputs,expected", [
    ("${input.left}", {"left": [7, 7]}, [{"left": ["Change.", None]}, {"left": [None, "Change."]}]),
    ("${input.left + input.right * 2}", {"left": [7, 7], "right": [7]}, [
        {"left": ["Change.", None]}, {"left": [None, "Change."]},
        {"right": ["Change."]}, {"right": ["Change."]},
    ]),
    ("${input.left}", {"left": []}, []),
])
def test_map_seed_routes_preserve_equal_item_positions(over, inputs, expected):
    from yian.compile.model import START_ID, END_ID
    from yian.expr.lineage import route_feedback
    from yian.nodes.contracts import ExecutionPosition, Parameter
    from yian.nodes.map import MapNode
    from tests.engine._graph_builder import _engine, _graph
    from tests.engine.test_expand import _child_flow

    node = MapNode("spawn", flow_id="child", child=_child_flow())
    node.params = [Parameter("x")]
    flow = _graph([node], [(START_ID, "spawn"), ("spawn", END_ID)],
                  wiring={"spawn": {"over": over, "x": "${item}"}})
    engine = _engine(flow, inputs, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    occurrences = {item.node_id: item for item in trace.occurrences}
    for index, feedback in enumerate(expected):
        child = occurrences[f"spawn#{index}/{START_ID}"]
        assert child.record.forward_capture.bound_inputs == {"x": 7}
        assert child.record.forward_capture.route.children[0].kind == "literal"
        routed = route_feedback(child.input_route, {"x": "Change."}, trace.lookup_source)
        assert [(item.source, item.feedback) for item in routed.contributions] == [
            (occurrences[START_ID].source, feedback),
        ]
        assert child.record.forward_capture.scope.occurrence_path == (
            ExecutionPosition("map_item", ("spawn",), index),
        )
    assert len(trace.boundaries) == len(expected)
    for node_id in (f"spawn/{START_ID}", f"spawn/{END_ID}"):
        assert occurrences[node_id].record.forward_capture.scope == ExecutionScope(("spawn",))
    assert not trace.pending_connections


@pytest.mark.parametrize("times", [0, 3])
def test_loop_carried_routes_survive_live_pruning(times):
    from yian.compile.model import START_ID, END_ID
    from yian.compose import load_flow
    from yian.expr.lineage import route_feedback
    from yian.nodes.contracts import ExecutionPosition
    from tests.engine._graph_builder import _engine

    loop_policy = f"times: {times}" if times else "while: '${n < 0}'\n    max: 3"
    flow = load_flow(f"""
id: seed-loop
name: seed-loop
input: {{n: int}}
defs:
  child:
    input: {{n: int}}
    nodes:
      work:
        kind: code
        input: {{n: '${{input.n}}'}}
        code: tests.engine._compose_codefns:loop_countdown
        output: {{n: int}}
    output: ${{work.output}}
nodes:
  spawn:
    kind: loop
    call: child
    {loop_policy}
    input: {{n: '${{input.n}}'}}
output: ${{spawn.output}}
""").compiled
    engine = _engine(flow, {"n": 9}, learnable=True)
    list(engine.run())
    trace = engine.state.trace
    occurrences = {item.node_id: item for item in trace.occurrences}
    drivers = [occurrences["spawn"], *(occurrences[f"spawn~{i}"] for i in range(1, times + 1))]
    assert len({item.source for item in drivers}) == times + 1
    for index, driver in enumerate(drivers):
        assert driver.record.forward_capture.scope.occurrence_path == (
            ExecutionPosition("loop_driver", ("spawn",), index),
        )
    assert trace.current_writers["spawn"].source == drivers[-1].source
    for index in range(times):
        child = occurrences[f"spawn#{index}/{START_ID}"]
        previous = occurrences[START_ID if index == 0 else f"spawn#{index - 1}/{END_ID}"]
        assert child.record.forward_capture.bound_inputs == {"n": 9 - index}
        routed = route_feedback(child.input_route, {"n": "Change."}, trace.lookup_source)
        assert [(item.source, item.feedback) for item in routed.contributions] == [
            (previous.source, {"n": "Change."}),
        ]
        assert child.record.forward_capture.scope.occurrence_path == (
            ExecutionPosition("loop_body", ("spawn",), index),
        )
        boundary = trace.boundaries[index]
        assert boundary.owner == drivers[index].source
        assert boundary.start_source == child.source
        assert boundary.end_source == occurrences[f"spawn#{index}/{END_ID}"].source
        if index < times - 1:
            assert child.node_id not in engine.state.flow.nodes
            assert child.source in trace.sources
    assert not trace.pending_connections


@pytest.mark.parametrize("kind", ["output", "route", "failure", "call", "map", "loop", "nested-call", "nested-map"])
def test_trace_replay_matches_live_accepted_prefixes(kind, monkeypatch):
    import copy
    from yian.compile.expand import build_grow
    from yian.compile.model import START_ID, END_ID
    from yian.compose import load_flow
    from yian.nodes.case import CaseNode
    from yian.runtime.state import RunState, _replay_records
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    if kind.startswith("nested-"):
        flow = _nested_growth_flow(kind.removeprefix("nested-"))
    elif kind in ("call", "map", "loop"):
        extra = "    over: [1, 1]\n" if kind == "map" else ""
        extra += "    times: 3\n" if kind == "loop" else ""
        binding = "${item}" if kind == "map" else "1"
        flow = load_flow(f"""
id: trace-growth
name: trace-growth
defs:
  child:
    input: {{n: int}}
    nodes:
      work:
        kind: code
        input: {{n: '${{input.n}}'}}
        code: tests.engine._compose_codefns:loop_countdown
        output: {{n: int}}
    output: ${{work.output}}
nodes:
  spawn:
    kind: {kind}
    call: child
{extra}    input: {{n: '{binding}'}}
output: ${{spawn.output}}
""").compiled
    elif kind == "route":
        flow = _graph([CaseNode("choice", [])], [(START_ID, "choice"), ("choice", END_ID, "default")])
    else:
        def work(_):
            if kind == "failure":
                raise RuntimeError("failed body")
            return {"x": 1}
        flow = _graph([FuncNode("work", work)], [(START_ID, "work"), ("work", END_ID)])

    snapshots = {}
    actual_nodes = {}

    def observe(_):
        snapshots[engine.state.next_sequence] = _trace_snapshot(engine.state.trace)
        actual_nodes.update(engine.state.flow.nodes)

    engine = _engine(flow, {"n": 9} if kind.startswith("nested-") else None, learnable=True, on_event=observe)
    original = copy.deepcopy(engine.state.flow)
    observe(None)
    list(engine.run())
    assert engine.state.result is not None
    if kind != "failure":
        from yian.events import RunSucceeded
        assert isinstance(engine.state.result, RunSucceeded)
    final_trace = engine.state.trace
    for source, handler in final_trace.backward_handlers.items():
        assert handler.__self__ is actual_nodes[handler.__self__.id]
    if kind == "loop":
        retired = [item for item in final_trace.occurrences if item.node_id not in engine.state.flow.nodes]
        assert retired
        for item in retired:
            assert item.source in final_trace.backward_handlers
            if type(item.record) is NodeSucceededRecord:
                assert item.source in final_trace.sources
                assert item.node_id not in final_trace.current_writers

    from yian.nodes.code import CodeNode

    def reject_rerun(*_args, **_kwargs):
        pytest.fail("replay evaluated a node body")

    monkeypatch.setattr(CodeNode, "forward", reject_rerun)
    monkeypatch.setattr(FuncNode, "forward", reject_rerun)
    records = tuple(engine.log)
    for length, expected in sorted(snapshots.items()):
        replay = _replay_records(RunState(copy.deepcopy(original)), records[:length], grow_builder=build_grow)
        assert _trace_snapshot(replay.trace) == expected


def test_retry_clears_unfinished_start_identity_and_primed_input_is_replaced():
    from yian.compile.model import START_ID, END_ID
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_run_resumed
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    node = FuncNode("read", lambda inputs: inputs, reads={"x": "${input.x}"})
    engine = _engine(_graph([node], [(START_ID, "read"), ("read", END_ID)]), {"x": 2}, learnable=True)
    trace = engine.state.trace
    assert trace.resolve("input.x").source == SourceRef(engine.state.run_id, 0, ("x",))
    commit_prepared(engine.log, engine.state, prepare_node_started(engine.state, START_ID))
    abandoned = trace.started[START_ID]
    commit_prepared(engine.log, engine.state, prepare_run_resumed(engine.state, num_workers=0))
    assert not trace.started
    list(engine.run())
    start = next(item for item in trace.occurrences if item.node_id == START_ID)
    assert start.source.execution_seq != abandoned
    assert SourceRef(engine.state.run_id, abandoned) not in trace.backward_handlers
    assert start.record.forward_capture.route.children[0].source == SourceRef(engine.state.run_id, 0, ("x",))
    assert trace.resolve("input.x").source == SourceRef(engine.state.run_id, start.source.execution_seq, ("x",))


def test_capture_canonicalizes_supported_binding_objects():
    from yian.compile.model import START_ID, END_ID
    from yian.typesys import FileRef
    from yian.runtime.log import canonical_seed
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    raw = {"values": (1, 2), "file": FileRef(uri="file:///tmp/example.txt")}
    node = FuncNode("work", lambda _: "done", reads=raw)
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learnable=True)
    list(engine.run())
    record = next(item.record for item in engine.state.trace.occurrences if item.node_id == "work")
    assert record.forward_capture.bound_inputs == canonical_seed(raw)[1]


def test_a_lookup_retains_its_entry_when_a_new_writer_commits(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from yian.compile.model import START_ID, END_ID
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from yian.runtime.trace import SourceEntry
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _forward_capture, _graph

    node = FuncNode("read", lambda inputs: inputs, reads={"x": "${input.x}"})
    engine = _engine(_graph([node], [(START_ID, "read"), ("read", END_ID)]), {"x": 10}, learnable=True)
    trace = engine.state.trace
    entry_read = threading.Event()
    continue_read = threading.Event()
    original_get = SourceEntry.value.fget

    def delayed_get(entry):
        if entry.source.execution_seq == 0:
            entry_read.set()
            assert continue_read.wait(5)
        return original_get(entry)

    monkeypatch.setattr(SourceEntry, "value", property(delayed_get))
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(trace.resolve, "input.x")
        try:
            assert entry_read.wait(5)
            commit_prepared(engine.log, engine.state, prepare_node_started(engine.state, START_ID))
            commit_prepared(engine.log, engine.state, prepare_node_succeeded(
                engine.state, START_ID, {"x": 20},
                forward_capture=_forward_capture(engine.state.flow.nodes[START_ID], {"x": 10}),
            ))
        finally:
            continue_read.set()
        resolved = pending.result(timeout=5)
    assert resolved.value == 10
    assert resolved.source == SourceRef(engine.state.run_id, 0, ("x",))
    assert trace.resolve("input.x").value == 20
    assert trace.resolve("input.x").source == SourceRef(engine.state.run_id, 1, ("x",))


@pytest.mark.parametrize("expression", [
    "${a}", "${a | b}", "${[a, b]}", "${a + b}", "${a * 2}",
    "${-2 * a}", "${n + 1}", "n=${n}", "${absent | absent}",
    "partial: ${partial()}", "score: ${score()}",
])
def test_codec_preserves_evaluated_route_variants(expression, monkeypatch):
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.expr.lineage import ResolvedValue
    from yian.expr.template import eval_binding
    from yian.runtime.contracts import ForwardCapture

    monkeypatch.setitem(TEMPLATE_FNS, "partial", lambda: {"kept": [1], "lost": object()})
    monkeypatch.setitem(TEMPLATE_FNS, "score", lambda: float("inf"))
    values = {"a": [1], "b": [2], "n": 3}

    def resolve(path):
        return ResolvedValue(values.get(path), path in values,
                             SourceRef("run", 0, (path,)) if path in values else None)

    traced = eval_binding(expression, resolve, trace=True)
    capture = ForwardCapture({"x": traced.value}, ValueRoute.record((("x", traced.route),)))
    record = NodeSucceededRecord("work", "work", canonical_typed(None, None)[0], forward_capture=capture)
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 3, record)
    assert decode_record(encode_record(envelope)) == envelope


_OPAQUE_ARGUMENTS = [
    ("1", ("",), (1,), {}, 1),
    ("1, 2", ("", ""), (1, 2), {}, 3),
    ("value=1", ("value",), (), {"value": 1}, 1),
    ("1, value=2", ("", "value"), (1,), {"value": 2}, 3),
]


@pytest.mark.parametrize("arguments,names,args,kwargs,result", _OPAQUE_ARGUMENTS)
def test_codec_preserves_opaque_argument_labels(
    arguments, names, args, kwargs, result, monkeypatch,
):
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.expr.lineage import route_feedback
    from yian.expr.template import eval_binding
    from yian.runtime.contracts import ForwardCapture

    calls = []

    def collect_args(*positional, **keywords):
        calls.append((positional, keywords))
        return sum(positional) + sum(keywords.values())

    monkeypatch.setitem(TEMPLATE_FNS, "collect_args", collect_args)
    traced = eval_binding("${collect_args(" + arguments + ")}", lambda _: None, trace=True)
    capture = ForwardCapture({"x": traced.value}, ValueRoute.record((("x", traced.route),)))
    record = NodeSucceededRecord("work", "work", canonical_typed(result, None)[0],
                                 forward_capture=capture)
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 3, record)
    decoded = decode_record(encode_record(envelope))
    assert decoded == envelope
    route = decoded.record.forward_capture.route.children[0]
    assert route.kind == "opaque" and route.names == names
    stopped = route_feedback(route, "Explain this result.", lambda _: None)
    assert stopped.contributions == ()
    assert len(stopped.stopped) == 1
    assert stopped.stopped[0].context[-1].result == result
    assert stopped.stopped[0].context[-1].expression == traced.route.expression
    assert calls == [(args, kwargs)]


@pytest.mark.parametrize("arguments,names,args,kwargs,result", _OPAQUE_ARGUMENTS)
def test_learning_retains_opaque_calls_with_arguments(
    arguments, names, args, kwargs, result, monkeypatch,
):
    from yian.compile.model import START_ID, END_ID
    from yian.events import RunSucceeded
    from yian.expr.builtins import TEMPLATE_FNS
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    calls, delivered = [], []

    def collect_args(*positional, **keywords):
        calls.append((positional, keywords))
        return sum(positional) + sum(keywords.values())

    def body(inputs):
        delivered.append(inputs)
        return inputs

    monkeypatch.setitem(TEMPLATE_FNS, "collect_args", collect_args)
    node = FuncNode("work", body, reads={"x": "${collect_args(" + arguments + ")}"})
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learnable=True)
    list(engine.run())
    assert isinstance(engine.state.result, RunSucceeded)
    capture = next(entry.record.forward_capture for entry in engine.state.trace.occurrences
                   if entry.node_id == "work")
    assert delivered == [{"x": result}] and capture.bound_inputs == {"x": result}
    assert capture.route.children[0].names == names
    assert calls == [(args, kwargs)]


@pytest.mark.parametrize("temporary", [
    pytest.param("\ud800", id="string"),
    pytest.param(["kept", "\ud800"], id="list"),
    pytest.param({"kept": 1, "lost": "\ud800"}, id="record-value"),
    pytest.param({"\ud800": 1}, id="record-key"),
    pytest.param({"kept": 1, "lost": {"\ud800": 1}}, id="nested-key"),
])
def test_learning_retains_valid_results_of_noncanonical_unicode_temporaries(temporary, monkeypatch):
    from yian.compile.model import START_ID, END_ID
    from yian.events import RunSucceeded
    from yian.expr.builtins import TEMPLATE_FNS
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    calls, delivered = [], []

    def bad_text():
        calls.append(None)
        return temporary

    def body(inputs):
        delivered.append(inputs)
        return inputs

    monkeypatch.setitem(TEMPLATE_FNS, "bad_text", bad_text)
    node = FuncNode("work", body, reads={"x": "${bad_text() == bad_text()}"})
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learnable=True)
    list(engine.run())
    assert isinstance(engine.state.result, RunSucceeded)
    envelope = next(item for item in engine.log
                    if isinstance(item.record, NodeSucceededRecord) and item.record.node_id == "work")
    assert decode_record(encode_record(envelope)) == envelope
    capture = envelope.record.forward_capture
    assert delivered == [{"x": True}] and capture.bound_inputs == {"x": True}
    assert all(route.value_json is None for route in capture.route.children[0].children)
    assert calls == [None, None]


@pytest.mark.parametrize("temporary,has_snapshot", [
    pytest.param({"safe": [1], "bad": "\ud800", "lost": object()}, True, id="partial-value"),
    pytest.param({"safe": [1], "\ud800": object()}, False, id="partial-key"),
])
def test_learning_retains_noncanonical_partial_temporary_comparisons(
    temporary, has_snapshot, monkeypatch,
):
    from yian.compile.model import START_ID, END_ID
    from yian.events import RunSucceeded
    from yian.expr.builtins import TEMPLATE_FNS
    from yian.expr.lineage import route_feedback
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    calls = []

    def partial_text():
        calls.append(None)
        return temporary

    monkeypatch.setitem(TEMPLATE_FNS, "partial_text", partial_text)
    node = FuncNode("work", lambda inputs: inputs, reads={"x": "${partial_text() != null}"})
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learnable=True)
    list(engine.run())
    assert isinstance(engine.state.result, RunSucceeded)
    envelope = next(item for item in engine.log
                    if isinstance(item.record, NodeSucceededRecord) and item.record.node_id == "work")
    assert decode_record(encode_record(envelope)) == envelope
    capture = envelope.record.forward_capture
    assert capture.bound_inputs == {"x": True}
    opaque = capture.route.children[0].children[0]
    assert opaque.kind == "opaque" and opaque.result_type == "builtins.dict"
    if has_snapshot:
        fields = dict(zip(opaque.snapshot.names, opaque.snapshot.children))
        assert json.loads(fields["safe"].value_json) == [1]
        assert fields["bad"].result_type == "builtins.str"
        assert fields["lost"].result_type == "builtins.object"
    else:
        assert opaque.snapshot is None
    routed = route_feedback(capture.route, "Explain.", lambda _: None)
    assert routed.contributions == () and len(routed.stopped) == 1
    assert routed.stopped[0].context[-1].result is None
    assert calls == [None]


@pytest.mark.parametrize("names", [[], ["", "", ""], [1, "value"]])
def test_codec_rejects_misaligned_or_nonstring_opaque_argument_labels(names):
    from yian.runtime.contracts import ForwardCapture

    route = ValueRoute("opaque", value_json="3", names=("", "value"),
                       children=(ValueRoute.literal(1), ValueRoute.literal(2)))
    capture = ForwardCapture({"x": 3}, ValueRoute.record((("x", route),)))
    record = NodeSucceededRecord("work", "work", canonical_typed(None, None)[0],
                                 forward_capture=capture)
    wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 3, record)))
    wire["payload"]["forward_capture"]["route"]["children"][0]["names"] = names
    with pytest.raises((TypeError, ValueError)):
        decode_record(json.dumps(wire))


def test_forward_capture_rejects_duplicate_record_names():
    from yian.runtime.contracts import ForwardCapture

    route = ValueRoute.record((("x", ValueRoute.literal(1)), ("x", ValueRoute.literal(2))))
    with pytest.raises(ValueError, match="unique"):
        ForwardCapture({"x": 2}, route)


@pytest.mark.parametrize("field", ["children", "path", "scope"])
def test_codec_revalidates_forged_immutable_capture_fields(field):
    from yian.runtime.contracts import ForwardCapture

    source = SourceRef("run", 0)
    child = ValueRoute("source", source=source)
    scope = ExecutionScope(("work",))
    capture = ForwardCapture({"x": 1}, ValueRoute.record((("x", child),)), scope=scope)
    record = NodeSucceededRecord("work", "work", canonical_typed(None, None)[0], forward_capture=capture)
    if field == "children":
        object.__setattr__(capture.route, "children", [child])
    elif field == "path":
        object.__setattr__(source, "path", [])
    else:
        object.__setattr__(scope, "occurrence_path", [])
    with pytest.raises((TypeError, ValueError)):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 3, record))


def test_scope_free_evidence_rejects_recalled_sections():
    from yian.memory import MemorySection

    with pytest.raises(ValueError, match="scope.memories"):
        EvaluationEvidence(None, {}, None, None, (MemorySection("outer", "recalled"),))
