"""Unit tests for the CompiledFlow topology model."""

from agent_composer.compile.model import END_ID, START_ID, Edge, Flow, CompiledFlow
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from tests.engine._fakes import FuncNode


def _nodes(*ids):
    return {i: FuncNode(i, lambda p: {}) for i in ids}


def _with_boundary(nodes: dict) -> dict:
    # inject the real START_ID/END_ID boundary NODES (so `from_parts` roots at START_ID + END_ID
    # is the terminal node, the post-flip model).
    nodes = dict(nodes)
    nodes.setdefault(START_ID, StartNode(START_ID, input_decls=[]))
    nodes.setdefault(END_ID, EndNode.record(END_ID, output_names=[]))
    return nodes


def test_adjacency_and_root():
    nodes = _with_boundary(_nodes("a", "b", "c"))
    edges = [
        Edge("e0", START_ID, "a"),
        Edge("e1", "a", "b"),
        Edge("e2", "a", "c"),
        Edge("e3", "b", END_ID),
    ]
    g = CompiledFlow.from_parts(nodes, edges)
    # the single root is the synthesized START_ID; `a` is its out-edge target.
    assert g.start_id == START_ID
    assert {e.to for e in g.outgoing(START_ID)} == {"a"}
    assert {e.to for e in g.outgoing("a")} == {"b", "c"}
    assert [e.from_ for e in g.incoming("b")] == ["a"]
    assert g.terminal_id == END_ID


def test_diamond_incoming():
    nodes = _nodes("a", "b", "c", "d")
    edges = [
        Edge("e0", START_ID, "a"),
        Edge("e1", "a", "b"),
        Edge("e2", "a", "c"),
        Edge("e3", "b", "d"),
        Edge("e4", "c", "d"),
        Edge("e5", "d", END_ID),
    ]
    g = CompiledFlow.from_parts(nodes, edges)
    assert {e.from_ for e in g.incoming("d")} == {"b", "c"}


def test_edge_input_group():
    tagged = Edge(id="e0", from_="a", to="b", input_group="x")
    assert tagged.input_group == "x"
    untagged = Edge(id="e1", from_="a", to="b")
    assert untagged.input_group is None
    legacy = Edge("e2", START_ID, "a")
    assert legacy.input_group is None


def test_compiled_flow_wiring_field_threads_and_defaults():
    # CompiledFlow carries flow-owned wiring (dict[node_id][param] -> source). Default {}.
    nodes = _nodes("a")
    edges = [Edge("e0", START_ID, "a"), Edge("e1", "a", END_ID)]
    g = CompiledFlow.from_parts(nodes, edges, wiring={"a": {"x": "${input.x}"}})
    assert g.wiring == {"a": {"x": "${input.x}"}}
    assert CompiledFlow.from_parts(nodes, edges).wiring == {}


def test_flow_base_roundtrips_fields_with_boundary_defaults():
    # A bare Flow round-trips nodes/edges/wiring; start_id/end_id default to the reserved boundary ids.
    nodes = _nodes("a")
    edges = [Edge("e0", START_ID, "a"), Edge("e1", "a", END_ID)]
    wiring = {"a": {"x": "${input.x}"}}
    f = Flow(nodes, edges, wiring=wiring)
    assert f.nodes is nodes
    assert f.edges is edges
    assert f.wiring == wiring
    assert f.start_id == START_ID
    assert f.end_id == END_ID
    # Spliced flows pass their own namespaced boundary ids.
    spliced = Flow(nodes, edges, start_id="call#1/__start__", end_id="call#1/__end__")
    assert spliced.start_id == "call#1/__start__"
    assert spliced.end_id == "call#1/__end__"
    assert spliced.wiring == {}  # wiring defaults to {}


def test_compiled_flow_is_flow_and_keeps_boundary_and_terminal():
    from agent_composer.compose import load_flow

    compiled = load_flow(_OPTIONAL_EDGE_FLOW).compiled
    assert isinstance(compiled, Flow)
    assert compiled.start_id == START_ID
    assert compiled.end_id == END_ID
    # terminal_id resolves to END_ID while the synthesized END node is present, None once absent.
    assert compiled.terminal_id == END_ID
    compiled.nodes.pop(END_ID)
    assert compiled.terminal_id is None


def test_edge_optional_defaults_false_and_roundtrips():
    assert Edge(id="a->b#0", from_="a", to="b", input_group="x").optional is False
    e = Edge(id="a->b#0", from_="a", to="b", input_group="x", optional=True)
    assert e.optional is True


_OPTIONAL_EDGE_FLOW = """
id: opt
name: opt
input:
  seed: float
nodes:
  a:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  hard:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      v: ${a.output}
    output: str
  soft:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      v: ${a.output:-null}
    output: str
output: ${hard.output}
"""


def test_data_edge_optional_reflects_binding_escape():
    from agent_composer.compose import load_flow

    flow = load_flow(_OPTIONAL_EDGE_FLOW).compiled
    by_to = {e.to: e for e in flow.edges if e.input_group == "v"}
    assert by_to["hard"].optional is False  # plain ref -> required
    assert by_to["soft"].optional is True   # `:-null` escape -> optional


_CASE_ESCAPE_FLOW = """
id: ce
name: ce
input:
  seed: float
nodes:
  a:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "${a.output:-0} >= 1"
        then: hot
    else: cold
  hot:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      seed: ${input.seed}
    output: str
  cold:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      seed: ${input.seed}
    output: str
output: ${gate.output}
"""


def test_case_condition_escape_marks_optional():
    from agent_composer.compose import load_flow

    flow = load_flow(_CASE_ESCAPE_FLOW).compiled
    a_to_gate = [e for e in flow.edges if e.from_ == "a" and e.to == "gate"]
    assert len(a_to_gate) == 1
    assert a_to_gate[0].optional is True  # `${a.output:-0}` in when: -> optional gate input


# --------------------------------------------------------------------------- #
# item-head edge-skip: a MAP/LOOP body binding `${item[.field]}` is body-local and
# yields NO data edge (there is no node named `item`), while `${producer.output}` in the
# SAME body DOES create a producer->consumer edge. Pins `_ref_producer`/`_binding_producers`
# skipping the `item` head against a silent regression of the edge-DAG builder.
# --------------------------------------------------------------------------- #


def test_item_head_ref_yields_no_edge_but_producer_ref_does():
    from agent_composer.compose.edges.data import infer_data_edges
    from agent_composer.compose.parse.nodes import parse_nodes
    from agent_composer.compose.source import SourceCursor

    # A CODE `producer` and a stand-in `mapbody` consumer whose wiring binds both an
    # `${item.foo}` (body-local) and a `${producer.output}` (a real producer ref). We drive
    # `infer_data_edges` with the hand-built `flow_wiring` for the consumer (the leaf/MAP
    # branch reads `flow_wiring[node_id]` verbatim), so this exercises the exact producer
    # ref-walk the map/loop body inputs go through.
    descriptors = parse_nodes(
        SourceCursor.virtual(
            {
                "producer": {
                    "kind": "code",
                    "code": "tests.engine._compose_codefns:score",
                    "input": {"seed": "${input.seed}"},
                    "output": "float",
                },
                "mapbody": {
                    "kind": "code",
                    "code": "tests.engine._compose_codefns:cautious",
                    "input": {"v": "${producer.output}"},
                    "output": "str",
                },
            }
        )
    )
    flow_wiring = {
        # the map body's per-element sink: `item`-headed refs are body-local (no edge),
        # `producer.output` is a genuine producer ref (an edge).
        "mapbody": {"a": "${item}", "b": "${item.foo}", "c": "${producer.output}"},
    }
    edges = infer_data_edges(descriptors, flow_wiring)
    froms = {(e.from_, e.to) for e in edges}
    assert ("producer", "mapbody") in froms      # the real producer ref makes an edge
    assert not any(e.from_ == "item" for e in edges)  # `${item[.foo]}` makes NO edge
    assert "item" not in {e.to for e in edges}


# --------------------------------------------------------------------------- #
# `binding_co_skips` -> edge `optional=` end-to-end (the co-skip contract, "Option A").
# A binding whose fallback is a REF (a bare ref, `.field` walk, or a `:-<ref>` default) may
# be absent without failing -> co-skips -> REQUIRED edge. A binding that pins a concrete
# value or REQUIRES presence (`:-"literal"`, `:-null`, `:?`) does NOT co-skip -> OPTIONAL
# edge. Pins `emit_for`'s `optional = not binding_co_skips(source)` for each stance.
# --------------------------------------------------------------------------- #

_COSKIP_STANCE_FLOW = """
id: coskip_stance
name: coskip_stance
input:
  seed: float
nodes:
  a:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  bare:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      v: ${a.output}
    output: str
  ref_default:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      v: ${a.output:-a.output}
    output: str
  lit_default:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      v: ${a.output:-"fallback"}
    output: str
  required:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      v: ${a.output:?"required"}
    output: str
output: ${bare.output}
"""


def test_binding_coskip_sets_edge_optional_per_stance():
    from agent_composer.compose import load_flow

    flow = load_flow(_COSKIP_STANCE_FLOW).compiled
    by_to = {e.to: e for e in flow.edges if e.input_group == "v"}
    # co-skipping bindings (ref fallback) -> REQUIRED (optional False):
    assert by_to["bare"].optional is False          # a bare ref
    assert by_to["ref_default"].optional is False   # `:-<ref>` default (ref fallback)
    # non-co-skipping bindings (concrete/required) -> OPTIONAL (optional True):
    assert by_to["lit_default"].optional is True    # `:-"literal"` default
    assert by_to["required"].optional is True       # `:?` required
