"""Runtime graph expansion: the live CALL/MAP Grow path through the engine.

Covers the live CALL Grow path through the engine's generic `_apply_grow` (clone the child
namespaced, substitute the spawner's value via the `commit_as` filler, fire its out-edges) plus
the runtime bounds (MAX_TOTAL_NODES / MAX_REF_DEPTH). The `eval_node` -> `NodeExpanded` routing
(spawner Grow grows the graph; a non-spawner Grow fails loud) is pinned in `test_eval_node_grow.py`.
"""

from tests.engine._fakes import stamp_reads


# --- _rens_internal — re-namespace a binding source under a callsite (no baking) ---------- #


def test_rens_internal_input_and_output_heads():
    from agent_composer.compile.expand import _rens_internal

    # ${input.k} -> the namespaced child START_ID's output object; ${x.output} node-first.
    assert _rens_internal("${input.k}", "cs") == "${cs/__start__.output.k}"
    assert _rens_internal("${x.output.field}", "cs") == "${cs/x.output.field}"
    # system.X and non-string pass through untouched.
    assert _rens_internal("${system.now}", "cs") == "${system.now}"
    assert _rens_internal(5, "cs") == 5


def test_rens_internal_computed_whole_span_renamespaces_all_leaves():
    # A whole-span computed expression `${x.output - y.output}` — the OLD flat-regex
    # rewriter read the interior as one bogus path and left it UNCHANGED; the parse-based
    # rewriter re-namespaces every reference leaf.
    from agent_composer.compile.expand import _rens_internal

    assert (
        _rens_internal("${x.output - y.output}", "cs")
        == "${cs/x.output - cs/y.output}"
    )
    # embedded text + two separate spans: literal text preserved, both spans re-namespaced.
    assert (
        _rens_internal("pre ${x.output} mid ${input.k} post", "cs")
        == "pre ${cs/x.output} mid ${cs/__start__.output.k} post"
    )


# --- live CALL Grow path — clone namespaced, substitute spawner, fire out-edge ----------- #

import pytest
from agent_composer.events import RunSucceeded
from agent_composer.compile.model import CompiledFlow, Edge, FlowOutput, NodeState, START_ID, END_ID
from agent_composer.compose.helpers.types import InputDecl, read_type
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.engine import FlowEngine
from tests.engine._fakes import FuncNode, derive_wiring


def _idecl(name: str, type_: str = "str") -> InputDecl:
    return InputDecl(name, type_, None, True, read_type(type_, {}))


def _with_boundary(nodes: dict, edges: list, outputs: list, input_decls=()) -> tuple[dict, list, dict]:
    # inject the real START_ID/END_ID boundary NODES so a manual `from_parts` flow runs. The
    # child START_ID carries `input_decls` (so its params == the call-arg names the splice seeds);
    # END_ID is RECORD-mode from the declared outputs; the manual `(.., END_ID)` terminal Edge is
    # replaced by the producer edge (input_group = output name) so the engine reads store[END_ID].
    from agent_composer.compose.edges.producers import _binding_producers

    decls = list(input_decls)
    nodes = dict(nodes)
    nodes.setdefault(START_ID, StartNode(START_ID, input_decls=decls))
    nodes[END_ID] = EndNode.record(END_ID, output_names=[o.name for o in outputs])
    edges = [e for e in edges if e.to != END_ID]
    counts: dict = {}
    for o in outputs:
        for producer in _binding_producers(o.from_):
            i = counts.get(producer, 0)
            counts[producer] = i + 1
            edges.append(Edge(id=f"{producer}->{END_ID}#{i}", from_=producer, to=END_ID,
                              input_group=o.name))
    wiring = derive_wiring(nodes)
    wiring[START_ID] = {d.name: f"${{input.{d.name}}}" for d in decls}
    wiring[END_ID] = {o.name: o.from_ for o in outputs}
    return nodes, edges, wiring


def _child_flow():
    # head(reads ${input.x}) -> tail(reads ${head.output}); single output ${tail.output}.
    head = stamp_reads(FuncNode("head", lambda i: i["x"] + 1), {"x": "${input.x}"})
    tail = stamp_reads(FuncNode("tail", lambda i: i["r"] * 10), {"r": "${head.output}"})
    nodes = {"head": head, "tail": tail}
    edges = [Edge("__start__->head#0", START_ID, "head", input_group="x"),
             Edge("head->tail#0", "head", "tail", input_group="r"),
             Edge("tail->__end__#0", "tail", END_ID)]
    outputs = [FlowOutput("out", "${tail.output}")]
    nodes, edges, wiring = _with_boundary(nodes, edges, outputs, input_decls=[_idecl("x", "int")])
    return CompiledFlow.from_parts(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring)


def _parent_with_spawner(child, record):
    # spawner "sp" is a real CALL returning Grow(child subgraph); successor "after" reads ${sp.output}.
    # The call-args come from stamped `${input.<k>}` reads for each key in `record`, so the test
    # seeds them via `run_inputs=record`. The parent declares matching input decls.
    from agent_composer.nodes.call.node import CallNode
    sp = CallNode("sp", flow_id="child", child=child,
                  child_inputs=child.nodes[child.start_id].params)
    stamp_reads(sp, {k: f"${{input.{k}}}" for k in record})
    after = stamp_reads(FuncNode("after", lambda i: i["v"]), {"v": "${sp.output}"})
    nodes = {"sp": sp, "after": after}
    edges = [Edge("__start__->sp", START_ID, "sp"),
             Edge("sp->after#0", "sp", "after", input_group="v"),
             Edge("after->__end__#0", "after", END_ID)]
    outputs = [FlowOutput("out", "${after.output}")]
    decls = [_idecl(k, "int") for k in record]
    nodes, edges, wiring = _with_boundary(nodes, edges, outputs, input_decls=decls)
    return CompiledFlow.from_parts(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring)


# (7 + 1) * 10 == 80 — a concrete child value flowing through the spawner substitution. The child
# has a single output "out", so the committed spawner value unwraps to the bare scalar 80.
EXPECTED_CHILD_VALUE = 80


@pytest.mark.parametrize("num_workers", [0, 4])
def test_call_grow_clones_namespaced_and_substitutes_spawner(num_workers):
    parent = _parent_with_spawner(_child_flow(), {"x": 7})
    eng = FlowEngine(parent, run_inputs={"x": 7}, num_workers=num_workers)
    events = list(eng.run())
    assert isinstance(events[-1], RunSucceeded)
    assert events[-1].output == EXPECTED_CHILD_VALUE            # "after" saw the child's value under "sp"
    assert eng.sm.node_state["sp"] == NodeState.EXPANDED
    assert any(nid.startswith("sp/") for nid in eng.flow.nodes)          # child cloned under "sp/"
    assert eng.pool.get("sp") == EXPECTED_CHILD_VALUE                    # substituted under the spawner id
    assert eng.flow.nodes["sp/__end__"].commit_as == "sp"               # the redirect filler is the child END_ID


# --- runtime bounds (MAX_TOTAL_NODES + MAX_REF_DEPTH) + REF-inside-MAP nesting proof ------- #

from agent_composer.compose import load_flow
from agent_composer.typesys.pool import VariablePool


# A REF-inside-MAP: parent MAP "each" over the elements; the child "mid" is itself a REF
# -> a CODE leaf. Uniform recursion namespaces the MAP element as `each#i` and the nested
# REF clone under it as `each#i/inner/...` (the REF callsite is the spawner id; only the
# MAP layer adds the `#i` element suffix — `ns`/`map_callsite`).
_REF_INSIDE_MAP = """
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
def test_ref_inside_map_expands_and_runs_uniformly(num_workers):
    # parent MAP "each" over 2 elements; the child is itself a REF -> a leaf. Uniform
    # recursion composes the namespacing as each#0/inner/..., each#1/inner/... (the MAP
    # layer adds `#i`; the nested REF callsite is the spawner id `each#i/inner`).
    loaded = load_flow(_REF_INSIDE_MAP)
    pool = VariablePool()
    pool.set(START_ID, {"xs": ["a", "b"]})
    eng = FlowEngine(loaded.compiled, pool, num_workers=num_workers)
    events = list(eng.run())
    assert isinstance(events[-1], RunSucceeded)
    assert events[-1].output == ["a", "b"]            # both elements ran through the nested REF
    # the doubly-namespaced clones exist in the live graph (uniform recursion)
    assert any(nid.startswith("each#0/inner/") for nid in eng.flow.nodes)
    assert any(nid.startswith("each#1/inner/") for nid in eng.flow.nodes)


def _ref_chain_flow(depth: int):
    """A static REF -> REF -> ... -> CODE leaf chain `depth` defs deep (a genuine chain,
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
def test_deep_ref_chain_trips_max_ref_depth(num_workers):
    from agent_composer.events import RunFailed
    from agent_composer.runtime.engine import FlowEngine, MAX_REF_DEPTH

    # a REF that calls a REF that calls a REF ... deeper than MAX_REF_DEPTH (a genuine chain,
    # NOT a wide fan-out) -> the depth bound fires before MAX_TOTAL_NODES. The RuntimeError
    # funnels to RunFailed via the dispatcher-thread wrap (NOT an uncaught raise out of run());
    # the pooled path funnels identically.
    parent = _ref_chain_flow(depth=MAX_REF_DEPTH + 2)
    pool = VariablePool()
    pool.set(START_ID, {"v": "go"})
    events = list(FlowEngine(parent, pool, num_workers=num_workers).run())
    assert isinstance(events[-1], RunFailed)            # status=="failed", not pytest.raises
    assert "MAX_REF_DEPTH" in events[-1].error


def _wide_child(n: int) -> CompiledFlow:
    # a single child flow whose node count alone exceeds the budget: one declared-output
    # root `n0` + (n-1) extra root pads, all reading ${input.x}. One clone busts the budget
    # right after add_subgraph (before any node runs), so the dangling pads never execute.
    nodes = {}
    edges = []
    for k in range(n):
        nid = f"n{k}"
        nodes[nid] = stamp_reads(FuncNode(nid, lambda i: i["x"]), {"x": "${input.x}"})
        edges.append(Edge(f"__start__->{nid}#0", START_ID, nid, input_group="x"))
    edges.append(Edge("n0->__end__#0", "n0", END_ID))
    outputs = [FlowOutput("out", "${n0.output}")]
    nodes, edges, wiring = _with_boundary(nodes, edges, outputs, input_decls=[_idecl("x", "int")])
    return CompiledFlow.from_parts(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring)


@pytest.mark.parametrize("num_workers", [0, 4])
def test_over_budget_expansion_trips_max_total_nodes(num_workers):
    from agent_composer.events import RunFailed
    from agent_composer.runtime.engine import FlowEngine, MAX_TOTAL_NODES

    parent = _parent_with_spawner(_wide_child(MAX_TOTAL_NODES + 10), {"x": 0})  # one clone busts it
    events = list(FlowEngine(parent, run_inputs={"x": 0}, num_workers=num_workers).run())
    assert isinstance(events[-1], RunFailed)            # status=="failed" via the dispatcher wrap
    assert "node budget" in events[-1].error
