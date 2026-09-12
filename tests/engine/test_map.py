"""MAP child-run seam — end-to-end, Ollama-free.

A `map` node (`kind: map` with `over:`) runs its baked child once per element of
`over:`, binding `${item}` per element, collecting each child's single value into
`list[U]`. `parallel` overlaps the element runs but preserves order. These drive real
CODE-only children via `run_flow`.
"""

import pytest

from agent_composer.compile.expand import build_grow
from agent_composer.compile.model import START_ID
from agent_composer.compose import LoadError, load_flow
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.run import run_flow
from tests.engine._graph_builder import _engine

# topic -> the topic (echo), so MAP results == the over list.
_ECHO_CHILD = """
id: echo-one
name: echo_one
input:
  topic: str
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.engine._compose_codefns:echo
output: ${emit.output}
"""

_BOOM_CHILD = """
id: boom-child
name: boom_child
input:
  topic: str
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.engine._compose_codefns:boom
output: ${emit.output}
"""


def _map_flow(child_id, parallel):
    par = "\n    parallel: true" if parallel else ""
    return f"""
id: map-parent
name: map_parent
input:
  topics: list[str]
uses:
  {child_id}: {child_id}
nodes:
  each:
    kind: map
    call: {child_id}
    over: ${{input.topics}}{par}
    input:
      topic: ${{item}}
output: ${{each.output}}
"""


def _resolver(**children):
    loaded = {fid: load_flow(text) for fid, text in children.items()}

    def resolve(flow_id, version=None):
        try:
            return loaded[flow_id]
        except KeyError as exc:
            raise LoadDiagnostics().error(f"unknown child {flow_id!r}") from exc

    return resolve


def test_map_runs_child_per_element_collecting_results():
    flow = load_flow(_map_flow("echo-one", parallel=False),
                     child_resolver=_resolver(**{"echo-one": _ECHO_CHILD}))
    result = run_flow(flow, {"topics": ["ACME", "BETA", "GOOG"]})
    assert result.status == "succeeded"
    assert result.output == ["ACME", "BETA", "GOOG"]


_OPAQUE_OUTPUT_MAP = """
id: opaque-output-map
name: opaque_output_map
input:
  values: list[int]
defs:
  increment:
    input:
      value: int
    nodes: {}
    output: ${input.value + 1}
nodes:
  each:
    kind: map
    call: increment
    over: ${input.values}
    input:
      value: ${item}
output: ${each.output}
"""


def test_map_distinguishes_declared_opaque_output_from_no_output():
    """A declared but unproven child output maps without a false list contract."""
    loaded = load_flow(_OPAQUE_OUTPUT_MAP)

    assert loaded.compiled.nodes["each"].output_contract is None

    result = run_flow(loaded, {"values": [1, 4]})

    assert result.status == "succeeded", result.failure
    assert result.output == [2, 5]


def test_map_empty_over_yields_empty_list():
    flow = load_flow(_map_flow("echo-one", parallel=False),
                     child_resolver=_resolver(**{"echo-one": _ECHO_CHILD}))
    result = run_flow(flow, {"topics": []})
    assert result.status == "succeeded"
    assert result.output == []


def test_map_parallel_preserves_element_order():
    flow = load_flow(_map_flow("echo-one", parallel=True),
                     child_resolver=_resolver(**{"echo-one": _ECHO_CHILD}))
    result = run_flow(flow, {"topics": ["A", "B", "C", "D"]})
    assert result.status == "succeeded"
    assert result.output == ["A", "B", "C", "D"]  # order preserved under parallel


def test_map_child_failure_fails_the_run():
    flow = load_flow(_map_flow("boom-child", parallel=False),
                     child_resolver=_resolver(**{"boom-child": _BOOM_CHILD}))
    result = run_flow(flow, {"topics": ["ACME"]})
    assert result.status != "succeeded"
    assert result.failure is not None
    assert "boom" in result.failure.message
    mapped = flow.compiled.nodes["each"]
    assert result.failure.node_id == "each#0/emit"
    assert result.failure.trace == (
        mapped.origin,
        mapped.child.nodes["emit"].code_origin,
    )


# `over:` with a `|` coalesce resolves at run through the same template layer it
# validated through at load (no load-vs-run asymmetry — the old bare-path resolver
# would have mis-split "inputs.topics | inputs.alt" and crashed).
_MAP_OVER_COALESCE = """
id: map-coalesce-over
name: map_coalesce_over
input:
  topics: Optional[list[str]]
  alt: list[str]
uses:
  echo-one: echo-one
nodes:
  each:
    kind: map
    call: echo-one
    over: ${input.topics | input.alt}
    input:
      topic: ${item}
output: ${each.output}
"""


def test_map_over_with_coalesce_resolves_at_run():
    # A run GROWS its compiled flow in place (the runtime expansion overlay), so each independent
    # run loads a fresh flow (the established per-run pattern; one loaded.compiled per run).
    # omitted topics -> coalesce falls through to alt.
    flow = load_flow(_MAP_OVER_COALESCE, child_resolver=_resolver(**{"echo-one": _ECHO_CHILD}))
    result = run_flow(flow, {"alt": ["X", "Y"]})
    assert result.status == "succeeded"
    assert result.output == ["X", "Y"]
    # present topics -> the first branch wins.
    flow2 = load_flow(_MAP_OVER_COALESCE, child_resolver=_resolver(**{"echo-one": _ECHO_CHILD}))
    result2 = run_flow(flow2, {"topics": ["A"], "alt": ["Z"]})
    assert result2.status == "succeeded"
    assert result2.output == ["A"]


# --- MapNode.run -> seed; compile builder reconstructs the ordered END-list fan-in --- #
def test_map_run_returns_grow():
    from agent_composer.nodes.base import Grow
    from agent_composer.nodes.ids import map_callsite, ns
    from agent_composer.nodes.map import MapNode

    from tests.engine.test_expand import _child_flow

    n = MapNode("m", flow_id="c", child=_child_flow(), child_inputs=[])
    out = n.run({"over": ["A", "B"]}, bind_item=lambda el, index: {"x": el})  # no system cap
    assert isinstance(out, Grow)
    assert out.seed == [{"x": "A"}, {"x": "B"}]
    subgraph = build_grow(n, out.seed)
    from agent_composer.compile.model import START_ID
    assert subgraph.start_id == ns("m", START_ID)
    for i in range(2):
        assert ns(map_callsite("m", i), _child_flow().start_id) in subgraph.nodes


def test_map_empty_over_returns_grow_with_lone_list_end():
    from agent_composer.compile.model import END_ID, START_ID
    from agent_composer.nodes.base import Grow
    from agent_composer.nodes.ids import ns
    from agent_composer.nodes.map import MapNode

    from tests.engine.test_expand import _child_flow

    n = MapNode("m", flow_id="c", child=_child_flow(), child_inputs=[])
    out = n.run({"over": []}, bind_item=lambda el, index: {})
    assert isinstance(out, Grow) and out.seed == []
    subgraph = build_grow(n, out.seed)
    assert set(subgraph.nodes) == {ns("m", START_ID), ns("m", END_ID)}
    assert subgraph.start_id == ns("m", START_ID)


@pytest.mark.parametrize("num_workers", [0, 4])
def test_map_parallel_via_num_workers_preserves_order(num_workers):
    flow = load_flow(_map_flow("echo-one", parallel=False),
                     child_resolver=_resolver(**{"echo-one": _ECHO_CHILD}))
    eng = _engine(flow.compiled, run_inputs={"topics": ["A", "B", "C", "D"]},
                     num_workers=num_workers, grow_builder=build_grow)
    assert list(eng.run())[-1].output == ["A", "B", "C", "D"]   # over-order join (the END_ID-list invariant)


@pytest.mark.parametrize("num_workers", [0, 4])
def test_map_n_zero_synthesizes_end_list_emitting_empty(num_workers):
    flow = load_flow(_map_flow("echo-one", parallel=False),
                     child_resolver=_resolver(**{"echo-one": _ECHO_CHILD}))
    eng = _engine(
        flow.compiled,
        run_inputs={"topics": []},
        num_workers=num_workers,
        grow_builder=build_grow,
    )
    assert list(eng.run())[-1].output == []     # N=0 -> synthetic map#/__start__ -> list END -> []


@pytest.mark.parametrize("num_workers", [0, 4])
def test_map_synthesizes_one_end_list_node(num_workers):
    # the MAP fan-in is ONE EndNode.list_ aggregator (replacing COLLECTOR): N=2 inputs e0/e1
    # wired to each element's child END_ID, aliased to the spawner.
    from agent_composer.nodes.base import NodeKind

    flow = load_flow(_map_flow("echo-one", parallel=False),
                     child_resolver=_resolver(**{"echo-one": _ECHO_CHILD}))
    eng = _engine(
        flow.compiled,
        run_inputs={"topics": ["A", "B"]},
        num_workers=num_workers,
        grow_builder=build_grow,
    )
    assert list(eng.run())[-1].output == ["A", "B"]
    map_end_id = "each/__end__"                          # ns(spawner, END_ID)
    assert eng.state.flow.nodes[map_end_id].kind == NodeKind.END
    into_end = {
        (e.from_, e.input_group)
        for e in eng.state.flow.edges
        if e.to == map_end_id
    }
    assert into_end == {("each#0/__end__", "e0"), ("each#1/__end__", "e1")}
    assert eng.state.flow.nodes[map_end_id].commit_as == "each"
