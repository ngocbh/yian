from agent_composer.compile.expand import clone_child
from agent_composer.compile.model import CompiledFlow, Edge, FlowOutput, Flow, START_ID, END_ID
from agent_composer.compose.helpers.types import InputDecl, read_type
from agent_composer.nodes.base import NodeKind, Output, ParamDecl
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.eval_node import EvaluationFailure
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import FuncNode, drive, stamp_reads, derive_wiring


def _decl(name: str, type_: str = "str") -> InputDecl:
    return InputDecl(name, type_, None, True, read_type(type_, {}))


def _child(outputs, *, tail_reads=None, inputs=("topic",)) -> CompiledFlow:
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


def test_clone_child_lowered_post_asserts_survive_clone():
    # A def's flow-level `asserts:` are LOWERED onto its END at build: each `${producer.output}`
    # post-assert becomes a synth END param (`__assert_<pid>`) fed by a `${pid.output:-null}`
    # wiring + a producer->END edge, and the post-assert is rewritten node-local (`${__assert_<pid>}`).
    # clone_child deep-copies the END, re-namespaces its wiring, and re-keys its edges — so the whole
    # structural dependency SURVIVES the clone (the C2 fix: no re-homing needed, and none possible
    # without deleting the synth edge that gates the assert on its producer).
    child = _child([FlowOutput(name="out", from_="${tail.output}")])
    end = child.nodes[END_ID]
    # Lower `${tail.output} != ''` onto END exactly as boundary synthesis does.
    end.post_asserts = ["${__assert_tail} != ''"]
    end.assert_texts = {"${__assert_tail} != ''": "${tail.output} != ''"}
    end.params.append(ParamDecl(name="__assert_tail"))
    child.wiring[END_ID]["__assert_tail"] = "${tail.output:-null}"
    child.edges.append(Edge(id=f"tail->{END_ID}#assert", from_="tail", to=END_ID,
                            input_group="__assert_tail", optional=True))

    cloned = clone_child(child, callsite="each", record={"topic": "ACME"})
    ce = cloned.nodes["each/__end__"]
    # the lowered post-assert is node-local (it names a synth PARAM, not a node) -> UNCHANGED.
    assert ce.post_asserts == ["${__assert_tail} != ''"]
    # the author's ORIGINAL text rides along so a failure message shows `${tail.output} != ''`.
    assert ce.assert_texts == {"${__assert_tail} != ''": "${tail.output} != ''"}
    # the synth wiring feeding the param is re-namespaced under the callsite.
    assert cloned.wiring["each/__end__"]["__assert_tail"] == "${each/tail.output:-null}"
    # the producer->END synth edge is re-keyed under the callsite (gates the assert on its producer).
    assert any(e.from_ == "each/tail" and e.to == "each/__end__"
               and e.input_group == "__assert_tail" for e in cloned.edges)


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
    end.post_asserts = ["${__assert_tail} != ''"]
    end.assert_texts = {"${__assert_tail} != ''": "${each/tail.output} != ''"}
    end.params.append(ParamDecl(name="__assert_tail"))
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
    assert "post-assert" in result.error
