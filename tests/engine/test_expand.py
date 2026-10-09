"""Runtime graph expansion: the live CALL/MAP Grow path through the engine.

Covers the live CALL Grow path through the engine's generic `_apply_grow` (clone the child
namespaced, substitute the spawner's value via the `commit_as` filler, fire its out-edges) plus
the runtime bounds (MAX_TOTAL_NODES / MAX_REF_DEPTH). The `eval_node` -> `NodeExpanded` routing
(spawner Grow grows the graph; a non-spawner Grow fails loud) is pinned in `test_eval_node_grow.py`.
"""

import pytest

from yian.compile.expand import build_grow
from yian.nodes.contracts import Binding
from tests.engine._fakes import stamp_reads


# --- _rens_internal — re-namespace a binding source under a callsite (no baking) ---------- #


def test_rens_internal_input_and_output_heads():
    from yian.compile.expand import _rens_internal

    # ${input.k} -> the namespaced child START_ID's output object; ${x.output} node-first.
    assert _rens_internal(Binding("${input.k}"), "cs") == Binding(
        "${cs/__start__.output.k}"
    )
    assert _rens_internal(Binding("${x.output.field}"), "cs") == Binding(
        "${cs/x.output.field}"
    )
    # system.X and non-string pass through untouched.
    assert _rens_internal(Binding("${system.now}"), "cs") == Binding("${system.now}")
    assert _rens_internal(Binding(5), "cs") == Binding(5)

    with pytest.raises(TypeError, match="must contain Binding"):
        _rens_internal("${input.k}", "cs")  # type: ignore[arg-type]


def test_rens_internal_computed_whole_span_renamespaces_all_leaves():
    # A whole-span computed expression re-namespaces every reference leaf.
    from yian.compile.expand import _rens_internal

    assert (
        _rens_internal(Binding("${x.output - y.output}"), "cs")
        == Binding("${cs/x.output - cs/y.output}")
    )
    # embedded text + two separate spans: literal text preserved, both spans re-namespaced.
    assert (
        _rens_internal(Binding("pre ${x.output} mid ${input.k} post"), "cs")
        == Binding("pre ${cs/x.output} mid ${cs/__start__.output.k} post")
    )


# --- live CALL Grow path — clone namespaced, substitute spawner, fire out-edge ----------- #

from yian.events import RunSucceeded
from yian.compile.model import CompiledFlow, Edge, FlowOutput, START_ID, END_ID
from yian.runtime.common import NodeState
from yian.compose.helpers.types import read_type
from yian.nodes.code import CodeNode
from yian.nodes.contracts import Parameter, ValueContract
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from tests.engine._fakes import FuncNode, derive_wiring
from tests.engine._graph_builder import _engine


def _parameter(name: str, type_: str = "str") -> Parameter:
    return Parameter(
        name=name,
        contract=ValueContract(read_type(type_, {})),
        required=True,
        coerce_strings=True,
    )


def _with_boundary(
    nodes: dict,
    edges: list,
    outputs: list,
    parameters=(),
    node_wiring=None,
) -> tuple[dict, list, dict]:
    # inject the real START_ID/END_ID boundary NODES so a manual `from_parts` flow runs. The
    # child START_ID carries params matching the call-arg names the splice seeds;
    # END_ID is RECORD-mode from the declared outputs; the manual `(.., END_ID)` terminal Edge is
    # replaced by the producer edge (input_group = output name) so the engine reads store[END_ID].
    from yian.compose.edges.producers import _binding_producers

    params = list(parameters)
    nodes = dict(nodes)
    nodes.setdefault(START_ID, StartNode(START_ID, params=params))
    nodes[END_ID] = EndNode.record(END_ID, output_names=[o.name for o in outputs])
    edges = [e for e in edges if e.to != END_ID]
    counts: dict = {}
    for o in outputs:
        for producer in _binding_producers(o.from_):
            i = counts.get(producer, 0)
            counts[producer] = i + 1
            edges.append(Edge(id=f"{producer}->{END_ID}#{i}", from_=producer, to=END_ID,
                              input_group=o.name))
    wiring = dict(node_wiring) if node_wiring is not None else derive_wiring(nodes)
    wiring[START_ID] = {parameter.name: f"${{input.{parameter.name}}}" for parameter in params}
    wiring[END_ID] = {o.name: o.from_ for o in outputs}
    return nodes, edges, wiring


def _child_flow():
    # head(reads ${input.x}) -> tail(reads ${head.output}); single output ${tail.output}.
    head = CodeNode("head", code='return inputs["x"] + 1')
    head.params = [Parameter(name="x")]
    tail = CodeNode("tail", code='return inputs["r"] * 10')
    tail.params = [Parameter(name="r")]
    nodes = {"head": head, "tail": tail}
    edges = [Edge("__start__->head#0", START_ID, "head", input_group="x"),
             Edge("head->tail#0", "head", "tail", input_group="r"),
             Edge("tail->__end__#0", "tail", END_ID)]
    outputs = [FlowOutput("out", "${tail.output}")]
    nodes, edges, wiring = _with_boundary(
        nodes,
        edges,
        outputs,
        parameters=[_parameter("x", "int")],
        node_wiring={
            "head": {"x": "${input.x}"},
            "tail": {"r": "${head.output}"},
        },
    )
    return CompiledFlow.from_parts(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring)


def _parent_with_spawner(child, record):
    # spawner "sp" is a real CALL returning a seed-first Grow; successor "after" reads ${sp.output}.
    # The call-args come from stamped `${input.<k>}` reads for each key in `record`, so the test
    # seeds them via `run_inputs=record`. The parent declares matching parameters.
    from yian.nodes.call.node import CallNode
    sp = CallNode("sp", flow_id="child", child=child,
                  child_inputs=child.nodes[child.start_id].params)
    stamp_reads(sp, {k: f"${{input.{k}}}" for k in record})
    after = stamp_reads(FuncNode("after", lambda i: i["v"]), {"v": "${sp.output}"})
    nodes = {"sp": sp, "after": after}
    edges = [Edge("__start__->sp", START_ID, "sp"),
             Edge("sp->after#0", "sp", "after", input_group="v"),
             Edge("after->__end__#0", "after", END_ID)]
    outputs = [FlowOutput("out", "${after.output}")]
    parameters = [_parameter(key, "int") for key in record]
    nodes, edges, wiring = _with_boundary(
        nodes, edges, outputs, parameters=parameters
    )
    return CompiledFlow.from_parts(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring)


# (7 + 1) * 10 == 80 — a concrete child value flowing through the spawner substitution. The child
# has a single output "out", so the committed spawner value unwraps to the bare scalar 80.
EXPECTED_CHILD_VALUE = 80


@pytest.mark.parametrize("num_workers", [0, 4])
def test_call_grow_clones_namespaced_and_substitutes_spawner(num_workers):
    parent = _parent_with_spawner(_child_flow(), {"x": 7})
    eng = _engine(
        parent,
        run_inputs={"x": 7},
        num_workers=num_workers,
        grow_builder=build_grow,
    )
    events = list(eng.run())
    assert isinstance(events[-1], RunSucceeded)
    assert events[-1].output == EXPECTED_CHILD_VALUE            # "after" saw the child's value under "sp"
    assert eng.state.node_state["sp"] == NodeState.EXPANDED
    assert any(nid.startswith("sp/") for nid in eng.state.flow.nodes)
    assert eng.state.pool.get("sp") == EXPECTED_CHILD_VALUE
    assert eng.state.flow.nodes["sp/__end__"].commit_as == "sp"


# --- runtime bounds (MAX_TOTAL_NODES + MAX_REF_DEPTH) + CALL-inside-MAP nesting proof ------ #

from yian.compose import load_flow


# A CALL-inside-MAP: parent MAP "each" over the elements; the child "mid" contains a CALL
# to a CODE leaf. Uniform recursion namespaces the MAP element as `each#i` and the nested
# CALL clone under it as `each#i/inner/...` (the CALL site is the spawner id; only the
# MAP layer adds the `#i` element suffix — `ns`/`map_callsite`).
_CALL_INSIDE_MAP = """
id: parent
name: parent
input:
  xs: list[str]
defs:
  leaf:
    input:
      v: str
    nodes:
      e:
        kind: code
        input:
          answer: ${input.v}
        output: str
        code: tests.seeds.fns:confirm_action
    output: ${e.output}
  mid:
    input:
      v: str
    nodes:
      inner:
        kind: call
        call: leaf
        input:
          v: ${input.v}
    output: ${inner.output}
nodes:
  each:
    kind: map
    over: ${input.xs}
    call: mid
    input:
      v: ${item}
output: ${each.output}
"""


@pytest.mark.parametrize("num_workers", [0, 4])
def test_call_inside_map_expands_and_runs_uniformly(num_workers):
    # Parent MAP "each" over 2 elements; the child contains a CALL to a leaf. Uniform
    # recursion composes the namespacing as each#0/inner/..., each#1/inner/... (the MAP
    # layer adds `#i`; the nested CALL site is the spawner id `each#i/inner`).
    loaded = load_flow(_CALL_INSIDE_MAP)
    eng = _engine(
        loaded.compiled,
        run_inputs={"xs": ["a", "b"]},
        num_workers=num_workers,
        grow_builder=build_grow,
    )
    events = list(eng.run())
    assert isinstance(events[-1], RunSucceeded)
    assert events[-1].output == ["a", "b"]            # both elements ran through the nested CALL
    # the doubly-namespaced clones exist in the live graph (uniform recursion)
    assert any(nid.startswith("each#0/inner/") for nid in eng.state.flow.nodes)
    assert any(nid.startswith("each#1/inner/") for nid in eng.state.flow.nodes)


def _call_chain_flow(depth: int):
    """A static CALL -> CALL -> ... -> CODE leaf chain `depth` defs deep (a genuine chain,
    NOT a wide fan-out): `step0` calls `step1` calls ... `step{depth-1}` calls the `leaf`.
    Drives the depth bound independently of the node budget."""
    lines = [
        "id: chain",
        "name: chain",
        "input:",
        "  v: str",
        "defs:",
        "  leaf:",
        "    input:",
        "      v: str",
        "    nodes:",
        "      e:",
        "        kind: code",
        "        input:",
        "          answer: ${input.v}",
        "        output: str",
        "        code: tests.seeds.fns:confirm_action",
        "    output: ${e.output}",
    ]
    for k in range(depth):
        callee = "leaf" if k == depth - 1 else f"step{k + 1}"
        lines += [
            f"  step{k}:",
            "    input:",
            "      v: str",
            "    nodes:",
            "      call_next:",
            "        kind: call",
            f"        call: {callee}",
            "        input:",
            "          v: ${input.v}",
            "    output: ${call_next.output}",
        ]
    lines += [
        "nodes:",
        "  root_call:",
        "    kind: call",
        "    call: step0",
        "    input:",
        "      v: ${input.v}",
        "output: ${root_call.output}",
    ]
    return load_flow("\n".join(lines)).compiled


@pytest.mark.parametrize("num_workers", [0, 4])
def test_deep_call_chain_trips_max_ref_depth(num_workers):
    from yian.events import RunFailed
    from yian.runtime.state import MAX_REF_DEPTH

    # A CALL chain deeper than MAX_REF_DEPTH (a genuine chain,
    # NOT a wide fan-out) -> the depth bound fires before MAX_TOTAL_NODES. The RuntimeError
    # funnels to RunFailed via the dispatcher-thread wrap (NOT an uncaught raise out of run());
    # the pooled path funnels identically.
    parent = _call_chain_flow(depth=MAX_REF_DEPTH + 2)
    events = list(
        _engine(
            parent,
            run_inputs={"v": "go"},
            num_workers=num_workers,
            grow_builder=build_grow,
        ).run()
    )
    assert isinstance(events[-1], RunFailed)            # status=="failed", not pytest.raises
    assert "MAX_REF_DEPTH" in events[-1].failure.message


def _wide_child(n: int) -> CompiledFlow:
    # a single child flow whose node count alone exceeds the budget: one declared-output
    # root `n0` + (n-1) extra root pads, all reading ${input.x}. One clone busts the budget
    # right after add_subgraph (before any node runs), so the dangling pads never execute.
    nodes = {}
    node_wiring = {}
    edges = []
    for k in range(n):
        nid = f"n{k}"
        node = CodeNode(nid, code="tests.engine._compose_codefns:echo_x")
        node.params = [Parameter(name="x")]
        nodes[nid] = node
        node_wiring[nid] = {"x": "${input.x}"}
        edges.append(Edge(f"__start__->{nid}#0", START_ID, nid, input_group="x"))
    edges.append(Edge("n0->__end__#0", "n0", END_ID))
    outputs = [FlowOutput("out", "${n0.output}")]
    nodes, edges, wiring = _with_boundary(
        nodes,
        edges,
        outputs,
        parameters=[_parameter("x", "int")],
        node_wiring=node_wiring,
    )
    return CompiledFlow.from_parts(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring)


@pytest.mark.parametrize("num_workers", [0, 4])
def test_over_budget_expansion_trips_max_total_nodes(num_workers):
    from yian.events import RunFailed
    from yian.runtime.state import MAX_TOTAL_NODES

    parent = _parent_with_spawner(_wide_child(MAX_TOTAL_NODES + 10), {"x": 0})  # one clone busts it
    events = list(
        _engine(
            parent,
            run_inputs={"x": 0},
            num_workers=num_workers,
            grow_builder=build_grow,
        ).run()
    )
    assert isinstance(events[-1], RunFailed)            # status=="failed" via the dispatcher wrap
    assert "node budget" in events[-1].failure.message
