import pytest

from yian.compile.expand import clone_child
from yian.compile.model import CompiledFlow, Edge, FlowOutput, Flow, START_ID, END_ID
from yian.compose.helpers.types import read_type
from yian.diagnostics import OriginRef
from yian.memory import (
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryScope,
    MemorySetupError,
)
from yian.nodes.base import NodeKind, Output
from yian.nodes.contracts import Assertion, Binding, Parameter, ValueContract
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.runtime.eval_node import EvaluationFailure
from yian.typesys.pool import VariablePool
from tests.engine._fakes import FuncNode, drive, stamp_reads, derive_wiring


def _parameter(name: str, type_: str = "str") -> Parameter:
    return Parameter(
        name=name,
        contract=ValueContract(read_type(type_, {})),
        required=True,
        coerce_strings=True,
    )


def _child(outputs, *, tail_reads=None, inputs=("topic",)) -> CompiledFlow:
    # A child shaped START_ID -> head (reads ${input.topic}) -> tail (reads ${head.output.r}) -> END_ID.
    head = stamp_reads(FuncNode("head", lambda i: {"r": i["topic"]}),
                       {"topic": "${input.topic}"})
    tail = stamp_reads(FuncNode("tail", lambda i: i["r"]),
                       tail_reads or {"r": "${head.output.r}"})
    parameters = [_parameter(name) for name in inputs]
    start = StartNode(START_ID, params=parameters)
    end = EndNode.record(END_ID, output_names=[o.name for o in outputs])
    nodes = {START_ID: start, "head": head, "tail": tail, END_ID: end}
    edges = [
        Edge(id="__start__->head#0", from_=START_ID, to="head", input_group="topic"),
        Edge(id="head->tail#0", from_="head", to="tail", input_group="r"),
    ]
    # the END_ID producer edges (one per declared output), as the loader mints them.
    for i, o in enumerate(outputs):
        producer = o.from_[2:-1].split(".")[0]  # ${<producer>.output...}
        edges.append(Edge(id=f"{producer}->{END_ID}#{i}", from_=producer, to=END_ID,
                          input_group=o.name))
    wiring = derive_wiring({"head": head, "tail": tail})
    wiring[START_ID] = {
        parameter.name: f"${{input.{parameter.name}}}" for parameter in parameters
    }
    wiring[END_ID] = {o.name: o.from_ for o in outputs}
    flow = CompiledFlow.from_parts(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring)
    return flow


def _single():  # one declared output
    return _child([FlowOutput(name="out", binding="${tail.output}")])


def test_clone_child_namespaces_ids():
    cloned = clone_child(_single(), callsite="each", record={"topic": "ACME"})
    assert isinstance(cloned, Flow)
    assert "each/head" in cloned.nodes and "each/tail" in cloned.nodes
    # the child START_ID/END_ID are cloned too (namespaced)
    assert "each/__start__" in cloned.nodes and "each/__end__" in cloned.nodes
    # the re-namespaced wiring uses the node-first shape:
    # internal ${head.output.r} re-namespaced to ${each/head.output.r}
    assert cloned.wiring["each/tail"]["r"] == Binding("${each/head.output.r}")
    # an internal edge is re-keyed under the callsite
    assert any(e.id == "each/head->tail#0" for e in cloned.edges)
    # start_id = the namespaced child START_ID (the sole seed point), NOT each/head
    assert cloned.start_id == "each/__start__"


def test_clone_child_preserves_node_origin_while_namespacing_runtime_ids():
    """Cloning changes execution identity without erasing authored provenance."""
    child = _single()
    origin = OriginRef("def/child/node/head")
    child.nodes["head"].origin = origin

    cloned = clone_child(child, callsite="each", record={"topic": "ACME"})

    assert cloned.nodes["each/head"].origin == origin
    assert cloned.nodes["each/head"].id == "each/head"
    assert child.nodes["head"].id == "head"


def test_clone_child_prefixes_existing_generated_origin_context():
    """A clone keeps nested ancestry after the enclosing spawner ancestry."""
    child = _single()
    outer = OriginRef("root/nodes/call")
    nested = OriginRef("child/nodes/map")
    child.nodes["head"].origin_context = (nested,)

    cloned = clone_child(
        child,
        callsite="each",
        record={"topic": "ACME"},
        origin_context=(outer,),
    )

    assert cloned.nodes["each/head"].origin_context == (outer, nested)
    assert child.nodes["head"].origin_context == (nested,)


def test_clone_child_rejects_a_dynamic_static_memory_template_atomically() -> None:
    child = _single()
    occurrence = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("call",),
        0,
    )
    child.nodes["head"].memory_scope = MemoryScope(
        ("call", "head"),
        occurrence_path=(occurrence,),
    )

    with pytest.raises(MemorySetupError, match="static memory template"):
        clone_child(child, callsite="call", record={"topic": "ACME"})

    assert child.nodes["head"].memory_scope.occurrence_path == (occurrence,)


def test_clone_child_revalidates_forged_occurrence_before_propagation() -> None:
    child = _single()
    for node_id, node in child.nodes.items():
        node.memory_scope = MemoryScope(("call", node_id))
    occurrence = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("map",),
        0,
    )
    # ``False == 0``; an equality-only guard would silently propagate this.
    object.__setattr__(occurrence, "index", False)

    with pytest.raises(TypeError, match="index must be an integer"):
        clone_child(
            child,
            callsite="call",
            record={"topic": "ACME"},
            occurrence_path=(occurrence,),
        )

    assert all(
        node.memory_scope.occurrence_path == () for node in child.nodes.values()
    )


def test_clone_child_revalidates_forged_static_scope_without_mutating_source() -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical memory key."""

    child = _single()
    scope = MemoryScope(("call", "head"), ("outer",))
    object.__setattr__(scope, "memories", (StringSubclass("outer"),))
    child.nodes["head"].memory_scope = scope

    with pytest.raises(MemorySetupError, match="noncanonical static memory scope"):
        clone_child(child, callsite="call", record={"topic": "ACME"})

    assert child.nodes["head"].memory_scope is scope


def test_clone_child_rewrites_one_binding_layer_and_preserves_its_origin():
    """Child namespacing rewrites semantic source text without replacing provenance."""
    child = _single()
    origin = OriginRef("child-tail-r")
    child.wiring["tail"]["r"] = Binding("${head.output.r}", origin)

    cloned = clone_child(child, callsite="each", record={"topic": "ACME"})

    rewritten = cloned.wiring["each/tail"]["r"]
    assert rewritten == Binding("${each/head.output.r}", origin)
    assert rewritten.origin is origin
    assert rewritten is not child.wiring["tail"]["r"]


def test_clone_child_does_not_rewrite_a_nested_binding_payload():
    """Only the outer IR carrier is interpreted; an inner carrier is user data."""
    child = _single()
    inner = Binding("${head.output.r}", OriginRef("inner"))
    child.wiring["tail"]["r"] = Binding(inner, OriginRef("outer"))

    cloned = clone_child(child, callsite="each", record={"topic": "ACME"})

    rewritten = cloned.wiring["each/tail"]["r"]
    assert isinstance(rewritten, Binding)
    assert rewritten.source is inner


def test_clone_child_seeds_start_with_call_args_as_edges():
    # a ${...} record value: the child START_ID's wiring carries it + a forward-ref edge is minted
    # (no baking into each/head).
    cloned = clone_child(_single(), callsite="each", record={"topic": "${upstream.output}"})
    assert cloned.wiring["each/__start__"]["topic"] == Binding("${upstream.output}")
    assert any(e.from_ == "upstream" and e.to == "each/__start__" and e.input_group == "topic"
               for e in cloned.edges)
    # each/head's ${input.topic} is re-pointed to the namespaced START_ID — NOT baked to a literal.
    cloned2 = clone_child(_single(), callsite="each", record={"topic": "ACME"})
    # a literal call-arg lands on the child START_ID's wiring as a constant seed (no edge).
    assert cloned2.wiring["each/__start__"]["topic"] == Binding("ACME")
    assert not any(e.to == "each/__start__" for e in cloned2.edges)
    # each/head reads the namespaced START_ID via the node-first shape.
    assert cloned2.wiring["each/head"]["topic"] == Binding(
        "${each/__start__.output.topic}"
    )


def test_clone_child_alias_target_is_child_end():
    cloned = clone_child(_single(), callsite="each", record={"topic": "ACME"})
    assert cloned.end_id == "each/__end__"
    end = cloned.nodes["each/__end__"]
    assert isinstance(end, EndNode) and end.kind == NodeKind.END


def test_clone_child_end_carries_producer_edges():
    # the cloned child END_ID keeps its producer->END_ID edges (re-keyed under the callsite), so the
    # alias filler (the child END_ID) fires only after its producers settle.
    child = _child([FlowOutput(name="report", binding="${tail.output}"),
                    FlowOutput(name="n", binding="${head.output.r}")])
    cloned = clone_child(child, callsite="each", record={"topic": "ACME"})
    into_end = {(e.from_, e.to, e.input_group) for e in cloned.edges if e.to == "each/__end__"}
    assert into_end == {("each/tail", "each/__end__", "report"),
                        ("each/head", "each/__end__", "n")}
    # the cloned child END_ID's wiring is re-namespaced to the node-first shape.
    assert cloned.wiring["each/__end__"] == {
        "report": Binding("${each/tail.output}"),
        "n": Binding("${each/head.output.r}"),
    }


def test_clone_child_lowered_post_asserts_survive_clone():
    # A def's flow-level `asserts:` are LOWERED onto its END at build: each `${producer.output}`
    # post-assert becomes a synth END param (`__assert_<pid>`) fed by a `${pid.output:-null}`
    # wiring + a producer->END edge, and the post-assert is rewritten node-local (`${__assert_<pid>}`).
    # clone_child deep-copies the END, re-namespaces its wiring, and re-keys its edges — so the whole
    # structural dependency SURVIVES the clone: no re-homing is needed or possible without
    # deleting the synth edge that gates the assert on its producer.
    child = _child([FlowOutput(name="out", binding="${tail.output}")])
    end = child.nodes[END_ID]
    # Lower `${tail.output} != ''` onto END exactly as boundary synthesis does.
    origin = OriginRef("flow/asserts/0")
    end.post_asserts = [Assertion("${__assert_tail} != ''", "${tail.output} != ''", origin)]
    end.params.append(Parameter(name="__assert_tail"))
    child.wiring[END_ID]["__assert_tail"] = Binding("${tail.output:-null}")
    child.edges.append(Edge(id=f"tail->{END_ID}#assert", from_="tail", to=END_ID,
                            input_group="__assert_tail", optional=True))

    cloned = clone_child(child, callsite="each", record={"topic": "ACME"})
    ce = cloned.nodes["each/__end__"]
    # the lowered post-assert is node-local (it names a synth PARAM, not a node) -> UNCHANGED.
    assert ce.post_asserts == [
        Assertion("${__assert_tail} != ''", "${tail.output} != ''", origin)
    ]
    assert ce.post_asserts[0].origin == origin
    # the synth wiring feeding the param is re-namespaced under the callsite.
    assert cloned.wiring["each/__end__"]["__assert_tail"] == Binding(
        "${each/tail.output:-null}"
    )
    # the producer->END synth edge is re-keyed under the callsite (gates the assert on its producer).
    assert any(e.from_ == "each/tail" and e.to == "each/__end__"
               and e.input_group == "__assert_tail" for e in cloned.edges)


def test_spawner_assert_append_preserves_child_and_spawner_origins():
    """Spawner lowering appends the semantic carrier without replacing child provenance."""
    child = _single()
    child_end = child.nodes[END_ID]
    child_origin = OriginRef("child/asserts/0")
    spawner_origin = OriginRef("parent/nodes/call/asserts/0")
    child_end.post_asserts = [
        Assertion("${output} != null", "child output exists", child_origin)
    ]

    cloned = clone_child(
        child,
        callsite="each",
        record={"topic": "ACME"},
        post_asserts=[
            Assertion("${output} != null", "call output exists", spawner_origin)
        ],
    )
    cloned_end = cloned.nodes[cloned.end_id]

    assert [a.display for a in cloned_end.post_asserts] == [
        "child output exists",
        "call output exists",
    ]
    assert [a.origin for a in cloned_end.post_asserts] == [
        child_origin,
        spawner_origin,
    ]


# --- firing contract: eval_node drives the node's `run` wrapper, which fires an END's post-assert
# RECORD-scoped. This is the production shape after ref lowering: a cross-node ref
# (`${<callsite>/X.output}`) becomes a synth PARAM (`__assert_tail`) fed by a `${.../X.output:-null}`
# wiring + a producer->END edge, so bind_params binds the value INTO the record — the assert reads
# `${__assert_tail}` node-locally. There is no pool-fallback (it was retired); an unbound ref would
# be absent from the record, not silently read from the pool.
def _run_end(end, pool):
    # END binds its declared params (including the synth `__assert_tail`) from its wiring, exactly
    # like a real flow. We derive the wiring from the node's stamped reads so both `out` and the
    # synth param resolve from the pool INTO the record; the post-assert then fires record-scoped.
    return drive(end, pool)


def _tail_assert_end():
    # An END whose post-assert reads a SYNTH PARAM bound into the record from `${each/tail.output}`,
    # modeled on the lowered shape in test_clone_child_lowered_post_asserts_survive_clone above.
    end = EndNode.record("each/__end__", output_names=["out"])
    end.post_asserts = [
        Assertion("${__assert_tail} != ''", "${each/tail.output} != ''")
    ]
    end.params.append(Parameter(name="__assert_tail"))
    return stamp_reads(
        end,
        {"out": "${each/tail.output}", "__assert_tail": "${each/tail.output:-null}"},
    )


def test_child_end_satisfied_synth_param_post_assert_succeeds():
    pool = VariablePool()
    pool.set("each/tail", "ok")                 # binds `__assert_tail` = "ok" into the record
    result = _run_end(_tail_assert_end(), pool)
    assert isinstance(result, Output)


def test_child_end_violated_synth_param_post_assert_fails():
    pool = VariablePool()
    pool.set("each/tail", "")                    # binds `__assert_tail` = "" -> violates the assert
    result = _run_end(_tail_assert_end(), pool)
    assert isinstance(result, EvaluationFailure)
    assert "post-assert" in result.failure.message
