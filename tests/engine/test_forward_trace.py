"""Accepted forward captures and producer identities survive reducer replay."""

from dataclasses import FrozenInstanceError
import json

import pytest

from agent_composer.expr.lineage import SourceRef, ValueRoute
from agent_composer.memory import EvaluationEvidence
from agent_composer.nodes.contracts import ExecutionScope
from agent_composer.runtime.log import (
    SCHEMA_VERSION, NodeSucceededRecord, RecordEnvelope, canonical_typed,
    decode_record, encode_record,
)


def test_forward_capture_round_trips_without_memory_scope():
    from agent_composer.runtime.contracts import ForwardCapture

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
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 3, record)
    decoded = decode_record(encode_record(envelope))
    assert decoded == envelope
    assert decoded.record.forward_capture.bound_inputs == {"value": [1, None]}
    decoded.record.forward_capture.bound_inputs["value"].append(8)
    decoded.record.forward_capture.node_data["saved"].append(9)
    assert capture.bound_inputs == {"value": [1, None]}
    assert capture.node_data == {"saved": [2]}
    with pytest.raises(FrozenInstanceError):
        capture.scope = None
    assert SCHEMA_VERSION == 5
    old = json.loads(encode_record(envelope))
    old["schema"] = 4
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
    from agent_composer.runtime.contracts import ForwardCapture

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
    from agent_composer.runtime.contracts import ForwardCapture

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
    from agent_composer.nodes.base import Node, Output
    from agent_composer.compile.model import START_ID, END_ID
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
    engine = _engine(flow, learn=True)
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
    from agent_composer.runtime.contracts import ForwardCapture
    from agent_composer.runtime.log import canonical_seed

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
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.nodes.base import Node, Output, Route, Grow
    from agent_composer.nodes.contracts import Parameter
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
                     learn=True, grow_builder=lambda *_: None)
    list(engine.run())
    accepted = engine.state.trace.occurrences[-1]
    assert accepted.node_id == "work"
    record = accepted.record
    assert record.failure == engine.state.result.failure
    assert record.forward_capture.bound_inputs == (None if stage == "binding" else {"x": 1})


def test_trace_is_applied_only_after_successful_append(monkeypatch):
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from tests.engine._graph_builder import _engine, _forward_capture, _graph

    engine = _engine(_graph([], [(START_ID, END_ID)]), learn=True)
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
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.events import NodeSucceeded
    from agent_composer.expr.builtins import TEMPLATE_FNS
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
    engine = _engine(flow, learn=True, num_workers=2, on_event=observe)
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
    from agent_composer.compile.model import START_ID, END_ID
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
    engine = _engine(flow, learn=True, num_workers=2, stop_event=stop)
    list(engine.run())
    assert body_finished == [True]
    trace = engine.state.trace
    assert "slow" in trace.started
    assert [entry.node_id for entry in trace.occurrences] == [START_ID, "bad"]
    assert "slow" not in trace.current_writers
    assert all(handler.__self__.id != "slow" for handler in trace.backward_handlers.values())


def test_learn_false_omits_backward_capture():
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.runtime.log import NodeSucceededRecord
    from tests.engine._graph_builder import _engine, _graph

    flow = _graph([], [(START_ID, END_ID)])
    engine = _engine(flow, learn=False)
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
    from agent_composer.compile.model import START_ID, END_ID
    from tests.engine._graph_builder import _engine, _graph

    engine = _engine(_graph([], [(START_ID, END_ID)]), {"x": None, "items": [1]}, learn=True)
    resolved = engine.state.trace.resolve(path)
    assert resolved.value == value
    assert resolved.present is present
    assert resolved.source == (None if source_path is None else SourceRef(engine.state.run_id, 0, source_path))
    head, *rest = path.strip().split(".")
    assert resolved.value == engine.state.pool.resolve(head, rest)


def test_reducer_requires_capture_exactly_when_learning():
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from tests.engine._graph_builder import _engine, _forward_capture, _graph

    for learn in (False, True):
        engine = _engine(_graph([], [(START_ID, END_ID)]), learn=learn)
        state = engine.state
        commit_prepared(engine.log, state, prepare_node_started(state, START_ID))
        capture = None if learn else _forward_capture(state.flow.nodes[START_ID], {})
        with pytest.raises(ValueError, match="capture"):
            prepare_node_succeeded(state, START_ID, {}, forward_capture=capture)


def _trace_snapshot(trace):
    return (
        trace.occurrences, dict(trace.sources), dict(trace.current_writers),
        dict(trace.started), trace.next_sequence,
        {source: (handler.__self__.id, handler.__self__.execution_scope,
                  handler.__func__.__qualname__)
         for source, handler in trace.backward_handlers.items()},
    )


@pytest.mark.parametrize("kind", ["output", "route", "failure", "call", "map", "loop"])
def test_trace_replay_matches_live_accepted_prefixes(kind, monkeypatch):
    import copy
    from agent_composer.compile.expand import build_grow
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.compose import load_flow
    from agent_composer.nodes.case import CaseNode
    from agent_composer.runtime.state import RunState, _replay_records
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    if kind in ("call", "map", "loop"):
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

    engine = _engine(flow, learn=True, on_event=observe)
    original = copy.deepcopy(engine.state.flow)
    observe(None)
    list(engine.run())
    assert engine.state.result is not None
    if kind != "failure":
        from agent_composer.events import RunSucceeded
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

    from agent_composer.nodes.code import CodeNode

    def reject_rerun(*_args, **_kwargs):
        pytest.fail("replay evaluated a node body")

    monkeypatch.setattr(CodeNode, "forward", reject_rerun)
    monkeypatch.setattr(FuncNode, "forward", reject_rerun)
    records = tuple(engine.log)
    for length, expected in sorted(snapshots.items()):
        replay = _replay_records(RunState(copy.deepcopy(original)), records[:length], grow_builder=build_grow)
        assert _trace_snapshot(replay.trace) == expected


def test_retry_clears_unfinished_start_identity_and_primed_input_is_replaced():
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.runtime.state import commit_prepared, prepare_node_started, prepare_run_resumed
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    node = FuncNode("read", lambda inputs: inputs, reads={"x": "${input.x}"})
    engine = _engine(_graph([node], [(START_ID, "read"), ("read", END_ID)]), {"x": 2}, learn=True)
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
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.typesys import FileRef
    from agent_composer.runtime.log import canonical_seed
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    raw = {"values": (1, 2), "file": FileRef(uri="file:///tmp/example.txt")}
    node = FuncNode("work", lambda _: "done", reads=raw)
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learn=True)
    list(engine.run())
    record = next(item.record for item in engine.state.trace.occurrences if item.node_id == "work")
    assert record.forward_capture.bound_inputs == canonical_seed(raw)[1]


def test_a_lookup_retains_its_entry_when_a_new_writer_commits(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from agent_composer.runtime.trace import SourceEntry
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _forward_capture, _graph

    node = FuncNode("read", lambda inputs: inputs, reads={"x": "${input.x}"})
    engine = _engine(_graph([node], [(START_ID, "read"), ("read", END_ID)]), {"x": 10}, learn=True)
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
    from agent_composer.expr.builtins import TEMPLATE_FNS
    from agent_composer.expr.lineage import ResolvedValue
    from agent_composer.expr.template import eval_binding
    from agent_composer.runtime.contracts import ForwardCapture

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
    from agent_composer.expr.builtins import TEMPLATE_FNS
    from agent_composer.expr.lineage import route_feedback
    from agent_composer.expr.template import eval_binding
    from agent_composer.runtime.contracts import ForwardCapture

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
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.events import RunSucceeded
    from agent_composer.expr.builtins import TEMPLATE_FNS
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
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learn=True)
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
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.events import RunSucceeded
    from agent_composer.expr.builtins import TEMPLATE_FNS
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
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learn=True)
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
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.events import RunSucceeded
    from agent_composer.expr.builtins import TEMPLATE_FNS
    from agent_composer.expr.lineage import route_feedback
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    calls = []

    def partial_text():
        calls.append(None)
        return temporary

    monkeypatch.setitem(TEMPLATE_FNS, "partial_text", partial_text)
    node = FuncNode("work", lambda inputs: inputs, reads={"x": "${partial_text() != null}"})
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), learn=True)
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
    from agent_composer.runtime.contracts import ForwardCapture

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
    from agent_composer.runtime.contracts import ForwardCapture

    route = ValueRoute.record((("x", ValueRoute.literal(1)), ("x", ValueRoute.literal(2))))
    with pytest.raises(ValueError, match="unique"):
        ForwardCapture({"x": 2}, route)


@pytest.mark.parametrize("field", ["children", "path", "scope"])
def test_codec_revalidates_forged_immutable_capture_fields(field):
    from agent_composer.runtime.contracts import ForwardCapture

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
    from agent_composer.memory import MemoryFrameRef, MemoryOwner, MemorySection

    frame = MemoryFrameRef("root", ("root",), MemoryOwner.FLOW)
    with pytest.raises(ValueError, match="scope.frame_stack"):
        EvaluationEvidence(None, {}, None, None, (MemorySection(frame, "recalled"),))
