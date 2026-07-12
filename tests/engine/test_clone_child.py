from agent_composer.compile.expand import clone_child
from agent_composer.compile.model import CompiledFlow, Edge, FlowOutput, Flow, START_ID, END_ID
from agent_composer.compose.validate.asserts import AssertSet
from agent_composer.compose.helpers.types import InputDecl, read_type
from agent_composer.events import NodeFailed, NodeSucceeded
from agent_composer.nodes.base import NodeKind, Output
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import FuncNode, drive, stamp_reads, derive_wiring


def _decl(name: str, type_: str = "str") -> InputDecl:
    return InputDecl(name, type_, None, True, read_type(type_, {}))


def _child(outputs, *, tail_reads=None, asserts=None, inputs=("topic",)) -> CompiledFlow:
    # A child shaped START_ID -> head (reads ${input.topic}) -> tail (reads ${head.output.r}) -> END_ID.
    head = stamp_reads(FuncNode("head", lambda i: {"r": i["topic"]}),
                       {"topic": "${input.topic}"})
    tail = stamp_reads(FuncNode("tail", lambda i: i["r"]),
                       tail_reads or {"r": "${head.output.r}"})
    decls = [_decl(n) for n in inputs]
    start = StartNode(START_ID, input_decls=decls)
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
    wiring[START_ID] = {d.name: f"${{input.{d.name}}}" for d in decls}
    wiring[END_ID] = {o.name: o.from_ for o in outputs}
    flow = CompiledFlow.from_parts(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring)
    flow.child_asserts = asserts  # clone_child reads the carried AssertSet off the cloned child
    return flow


def _single():  # one declared output
    return _child([FlowOutput(name="out", from_="${tail.output}")])


def test_clone_child_namespaces_ids():
    cloned = clone_child(_single(), callsite="each", record={"topic": "ACME"})
    assert isinstance(cloned, Flow)
    assert "each/head" in cloned.nodes and "each/tail" in cloned.nodes
    # the child START_ID/END_ID are cloned too (namespaced)
    assert "each/__start__" in cloned.nodes and "each/__end__" in cloned.nodes
    # the re-namespaced wiring uses the node-first shape:
    # internal ${head.output.r} re-namespaced to ${each/head.output.r}
    assert cloned.wiring["each/tail"]["r"] == "${each/head.output.r}"
    # an internal edge is re-keyed under the callsite
    assert any(e.id == "each/head->tail#0" for e in cloned.edges)
    # start_id = the namespaced child START_ID (the sole seed point), NOT each/head
    assert cloned.start_id == "each/__start__"


def test_clone_child_seeds_start_with_call_args_as_edges():
    # a ${...} record value: the child START_ID's wiring carries it + a forward-ref edge is minted
    # (no baking into each/head).
    cloned = clone_child(_single(), callsite="each", record={"topic": "${upstream.output}"})
    assert cloned.wiring["each/__start__"]["topic"] == "${upstream.output}"
    assert any(e.from_ == "upstream" and e.to == "each/__start__" and e.input_group == "topic"
               for e in cloned.edges)
    # each/head's ${input.topic} is re-pointed to the namespaced START_ID — NOT baked to a literal.
    cloned2 = clone_child(_single(), callsite="each", record={"topic": "ACME"})
    # a literal call-arg lands on the child START_ID's wiring as a constant seed (no edge).
    assert cloned2.wiring["each/__start__"]["topic"] == "ACME"
    assert not any(e.to == "each/__start__" for e in cloned2.edges)
    # each/head reads the namespaced START_ID via the node-first shape.
    assert cloned2.wiring["each/head"]["topic"] == "${each/__start__.output.topic}"


def test_clone_child_alias_target_is_child_end():
    cloned = clone_child(_single(), callsite="each", record={"topic": "ACME"})
    assert cloned.end_id == "each/__end__"
    end = cloned.nodes["each/__end__"]
    assert isinstance(end, EndNode) and end.kind == NodeKind.END


def test_clone_child_end_carries_producer_edges():
    # the cloned child END_ID keeps its producer->END_ID edges (re-keyed under the callsite), so the
    # alias filler (the child END_ID) fires only after its producers settle.
    child = _child([FlowOutput(name="report", from_="${tail.output}"),
                    FlowOutput(name="n", from_="${head.output.r}")])
    cloned = clone_child(child, callsite="each", record={"topic": "ACME"})
    into_end = {(e.from_, e.to, e.input_group) for e in cloned.edges if e.to == "each/__end__"}
    assert into_end == {("each/tail", "each/__end__", "report"),
                        ("each/head", "each/__end__", "n")}
    # the cloned child END_ID's wiring is re-namespaced to the node-first shape.
    assert cloned.wiring["each/__end__"] == {
        "report": "${each/tail.output}",
        "n": "${each/head.output.r}",
    }


def test_clone_child_rehomes_post_asserts_onto_child_end():
    child = _child([FlowOutput(name="out", from_="${tail.output}")],
                   asserts=AssertSet(boundary=["${input.topic} != ''"],
                                     post=["${tail.output} != ''"]))
    cloned = clone_child(child, callsite="each", record={"topic": "ACME"})
    # boundary asserts are NOT carried on the Flow — the engine reads them off the spawner node
    # via `iter_boundary_records` and evaluates them eagerly in `_apply_grow`.
    # post-assert re-namespaced under the callsite via the node-first shape.
    assert cloned.nodes["each/__end__"].post_asserts == ["${each/tail.output} != ''"]


# --- firing contract: eval_node fires an END_ID node's re-homed post-asserts POOL-scoped --
# An ${<callsite>/X.output} post-assert re-homed onto the child END_ID must fire POOL-scoped (not
# the record-scoped generic path, which would resolve the ref to None and false-fail).
def _run_end(end, pool):
    flow = type("F", (), {"wiring": {end.id: {}}})()  # END_ID takes its inputs as bound record
    return list(drive(end, pool, flow))


def test_child_end_satisfied_pool_scoped_post_assert_succeeds():
    end = EndNode.record("each/__end__", output_names=["out"])
    end.post_asserts = ["${each/tail.output} != ''"]
    pool = VariablePool()
    pool.set("each/tail", "ok")                 # the re-homed ref resolves POOL-scoped
    events = _run_end(stamp_reads(end, {"out": "${each/tail.output}"}), pool)
    assert any(isinstance(e, NodeSucceeded) for e in events)
    assert not any(isinstance(e, NodeFailed) for e in events)


def test_child_end_violated_pool_scoped_post_assert_fails():
    end = EndNode.record("each/__end__", output_names=["out"])
    end.post_asserts = ["${each/tail.output} != ''"]
    pool = VariablePool()
    pool.set("each/tail", "")                    # violates the post-assert (empty string)
    events = _run_end(stamp_reads(end, {"out": "${each/tail.output}"}), pool)
    failed = [e for e in events if isinstance(e, NodeFailed)]
    assert failed and "post-assert" in failed[0].error
