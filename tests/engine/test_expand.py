"""Runtime graph expansion: the live CALL/MAP Grow path through the engine.

Covers the live CALL Grow path through the engine's generic `_apply_grow` (clone the child
namespaced, substitute the spawner's value via the `commit_as` filler, fire its out-edges) plus
the runtime node budget (MAX_TOTAL_NODES). The `eval_node` -> `NodeExpanded` routing
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
    # END_ID takes its params from the declared outputs; the manual `(.., END_ID)` terminal Edge is
    # replaced by the producer edge (input_group = output name) so the engine reads store[END_ID].
    from yian.compose.edges.producers import _binding_producers

    params = list(parameters)
    nodes = dict(nodes)
    nodes.setdefault(START_ID, StartNode(START_ID, params=params))
    # These Python-API tests use a single output as a value; two or more form a record.
    if len(outputs) <= 1:
        nodes[END_ID] = EndNode.value(END_ID, output_name=outputs[0].name if outputs else None)
    else:
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


# --- runtime node budget (MAX_TOTAL_NODES), deep static nesting, CALL-inside-MAP nesting ---- #

from yian.compose import load_flow


# A CALL-inside-MAP: parent MAP "each" over the elements; the child "mid" contains a CALL
# to a CODE leaf. Uniform recursion namespaces the MAP element as `each#i` and the nested
# CALL clone under it as `each#i/inner/...` (the CALL site is the spawner id; only the
# MAP layer adds the `#i` element suffix — `ns`/`map_callsite`).
_CALL_INSIDE_MAP = """
id: parent
name: parent
input:
  xs: !list(str)
defs:
  leaf:
    input:
      v: !str
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
      v: !str
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


def _mixed_chain_source(levels: int) -> str:
    """A static chain `levels` CALL/MAP levels deep ending in a CODE leaf.

    The root CALLs `step1`; odd steps MAP over a one-element list and even steps CALL,
    so CALL and MAP alternate. Each level is a distinct def, so the chain is a genuine
    nesting (not a wide fan-out) and the loader accepts it as finite."""
    lines = [
        "id: chain",
        "name: chain",
        "input:",
        "  v: !str",
        "defs:",
        "  leaf:",
        "    input:",
        "      v: !str",
        "    nodes:",
        "      e:",
        "        kind: code",
        "        input:",
        "          answer: ${input.v}",
        "        output: str",
        "        code: tests.seeds.fns:confirm_action",
        "    output: ${e.output}",
    ]
    for k in range(1, levels):
        callee = "leaf" if k == levels - 1 else f"step{k + 1}"
        lines += [f"  step{k}:", "    input:", "      v: !str", "    nodes:"]
        if k % 2:
            lines += [
                "      next:",
                "        kind: map",
                # A whole-span list expression; a list literal would keep "${input.v}" as text.
                "        over: ${[input.v]}",
                f"        call: {callee}",
                "        input:",
                "          v: ${item}",
            ]
        else:
            lines += [
                "      next:",
                "        kind: call",
                f"        call: {callee}",
                "        input:",
                "          v: ${input.v}",
            ]
        lines += ["    output: ${next.output}"]
    lines += [
        "nodes:",
        "  top:",
        "    kind: call",
        "    call: step1",
        "    input:",
        "      v: ${input.v}",
        "output: ${top.output}",
    ]
    return "\n".join(lines)


_DEEP_LEVELS = 8
# 4 MAP levels (odd steps 1, 3, 5, 7) wrap the leaf's string in 4 one-element lists.
_DEEP_OUTPUT = [[[["go"]]]]


@pytest.mark.parametrize("num_workers", [0, 4])
def test_deep_static_call_map_chain_loads_and_succeeds(num_workers):
    # Nesting depth is static (the loader rejects recursion), so an 8-level chain, well
    # under the call nesting limit, runs to completion; the growth core still stamps each
    # level's depth.
    loaded = load_flow(_mixed_chain_source(_DEEP_LEVELS))
    eng = _engine(
        loaded.compiled,
        run_inputs={"v": "go"},
        num_workers=num_workers,
        grow_builder=build_grow,
    )
    events = list(eng.run())
    assert isinstance(events[-1], RunSucceeded), events[-1]
    assert events[-1].output == _DEEP_OUTPUT
    assert max(eng.state.depth.values()) == _DEEP_LEVELS


def test_deep_static_call_map_chain_resumes_after_its_last_expansion(tmp_path):
    # Record replay re-validates every recorded growth (depth included): cut the deep run's
    # log right after its last `node_expanded` record, so resume must replay all 8
    # expansions and then run the deepest leaf fresh.
    import json

    from yian.events import NodeStarted
    from yian.run import resume_flow, run_flow

    source = _mixed_chain_source(_DEEP_LEVELS)
    path = tmp_path / "deep.jsonl"
    first = run_flow(load_flow(source), {"v": "go"}, log=path, num_workers=0)
    assert first.status == "succeeded", first.failure
    lines = path.read_text().splitlines(keepends=True)
    expanded = [i for i, line in enumerate(lines) if json.loads(line)["kind"] == "node_expanded"]
    assert len(expanded) == _DEEP_LEVELS
    path.write_text("".join(lines[: expanded[-1] + 1]))

    resumed = resume_flow(load_flow(source), log=path)

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == _DEEP_OUTPUT
    started = {e.node_id for e in resumed.events if isinstance(e, NodeStarted)}
    assert any(node_id.endswith("/e") for node_id in started)  # the leaf ran after replay


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
    from yian.defaults import Defaults

    budget = Defaults().max_total_nodes
    parent = _parent_with_spawner(_wide_child(budget + 10), {"x": 0})  # one clone busts it
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
