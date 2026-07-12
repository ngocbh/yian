"""Unit tests for infer_data_edges — data-edge inference + input_group.

A flow-level pass: for every SINK binding in the flow — each built node's
`inputs` (the InputBindings from the binding pass), a mapped `call`'s `over:`, and a `case` node's
`on:`/`when:` refs — collect its reference paths. Binding/template sinks use `expr_refs_of`
(over `${...}` spans); a `case` `on:`/`when:` is a whole expression, so its refs come from
`condition_refs` (parses the whole condition, so the bare spelling is seen too). Each
`${<id>[.output.…]}` ref becomes a data `Edge(from_=<id>, to=<consumer>, input_group=<sink key>)`.

A coalesce `${a | b}` sink yields TWO edges that SHARE one `input_group` (the sink
input name). An `${input.X}` read mints a `START_ID->reader` input-producer DATA edge;
`${system.X}`/`${item}` mint none. NO `->END_ID` synthesis here (that is a later step).
"""

from pathlib import Path

import pytest

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.compose.build import build_leaf_node, infer_data_edges
from agent_composer.compose.parse.file import (
    AgentDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    parse_file,
)
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.parse.nodes import parse_nodes

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _seed(name: str):
    """(descriptors, flow_wiring) for a seed — leaf-kind wiring built, case/ref/map skipped.
    infer_data_edges now derives leaf edges from the flow-owned wiring."""
    f = parse_file((_SEEDS / name).read_text())
    descriptors = parse_nodes(NodeReader.spanless(f.nodes))
    flow_wiring = {
        nid: build_leaf_node(desc, {})[1]
        for nid, desc in descriptors.items()
        if isinstance(desc, (AgentDescriptor, CodeDescriptor))
    }
    return descriptors, flow_wiring


def _edge_tuples(edges):
    return {(e.from_, e.to, e.input_group) for e in edges}


# ---------- seed 01: plain data edges, input_group = sink field name ----------


def test_seed01_data_edges_with_input_group():
    descriptors, built = _seed("01-structured-agent.yaml")
    edges = infer_data_edges(descriptors, built)
    tuples = _edge_tuples(edges)
    # verdict reads score.rating + score.rationale (the input_group = the sink field).
    assert ("score", "verdict", "rating") in tuples
    assert ("score", "verdict", "rationale") in tuples
    # score reads only ${input.topic} -> its only incoming edge is the START_ID input-producer
    # edge (no ${X.output} producer feeds it).
    assert not any(e.to == "score" and e.from_ != START_ID for e in edges)
    assert ("__start__", "score", "topic") in tuples   # the input-producer edge


def test_infer_mints_start_input_edges_never_end_edges():
    # infer_data_edges mints START_ID->reader input-producer edges, but NEVER an
    # edge to/from END_ID (the END_ID producer + START_ID-root edges are synthesize_boundary).
    descriptors, built = _seed("01-structured-agent.yaml")
    edges = infer_data_edges(descriptors, built)
    assert any(e.from_ == START_ID for e in edges)      # input-producer edges exist
    for e in edges:
        assert e.to != END_ID and e.from_ != END_ID


def test_edge_id_is_synthesized():
    descriptors, built = _seed("01-structured-agent.yaml")
    edges = infer_data_edges(descriptors, built)
    # ids are unique + follow the from->to#i convention.
    ids = [e.id for e in edges]
    assert len(ids) == len(set(ids))
    for e in edges:
        assert e.id.startswith(f"{e.from_}->{e.to}#")


# ---------- coalesce sink: two edges SHARING one input_group ----------


def test_coalesce_sink_shares_input_group():
    # A node input bound to ${pro.output | con.output} -> two edges, same group.
    descriptors = parse_nodes(
        NodeReader.spanless(
            {
                "pro": {"kind": "agent", "outputs": "str", "prompt": "x"},
                "con": {"kind": "agent", "outputs": "str", "prompt": "x"},
                "merge": {
                    "kind": "code",
                    "inputs": {"claim": "${pro.output | con.output}"},
                    "outputs": "str",
                    "code": "m:f",
                },
            }
        )
    )
    wiring = {nid: build_leaf_node(d, {})[1] for nid, d in descriptors.items()}
    edges = infer_data_edges(descriptors, wiring)
    coalesce_edges = [e for e in edges if e.to == "merge"]
    assert len(coalesce_edges) == 2
    froms = {e.from_ for e in coalesce_edges}
    assert froms == {"pro", "con"}
    # BOTH edges share ONE input_group (the sink input name).
    groups = {e.input_group for e in coalesce_edges}
    assert groups == {"claim"}


# ---------- mapped call (over:) produces a data edge ----------


def test_map_over_ref_produces_data_edge():
    descriptors = parse_nodes(
        NodeReader.spanless(
            {
                "bundle": {"kind": "code", "outputs": "list[str]", "code": "m:f"},
                "each": {
                    "kind": "map",
                    "call": "child",
                    "over": "${bundle.output}",
                    "inputs": {"topic": "${item}"},
                },
            }
        )
    )
    # The MAP wiring is built by build_call_node (needs a resolver); supply it directly —
    # the `over` source rides the reserved "over" key, over-then-inputs order.
    wiring = {
        "bundle": build_leaf_node(descriptors["bundle"], {})[1],
        "each": {"over": "${bundle.output}", "topic": "${item}"},
    }
    edges = infer_data_edges(descriptors, wiring)
    tuples = _edge_tuples(edges)
    assert ("bundle", "each", "over") in tuples
    # ${item} is a body-local scope, not a pool ref -> no edge.
    assert not any(e.from_ == "item" for e in edges)


# ---------- case on:/when: refs produce data edges (score -> gate) ----------


def test_case_on_ref_produces_data_edge():
    # seed 06: route is a case with on: ${classify.output} -> classify -> route edge.
    descriptors, built = _seed("06-case-on.yaml")
    edges = infer_data_edges(descriptors, built)
    assert any(e.from_ == "classify" and e.to == "route" for e in edges)


def test_case_searched_when_ref_produces_data_edge():
    # seed 02: gate's when: "${score.output} >= 0.5" -> score -> gate edge.
    descriptors, built = _seed("02-case.yaml")
    edges = infer_data_edges(descriptors, built)
    assert any(e.from_ == "score" and e.to == "gate" for e in edges)


def test_case_multiref_when_produces_data_edge():
    # seed 10: size's when reads ${score.output} (and ${input.weight}) -> one edge.
    descriptors, built = _seed("10-asserts-arithmetic.yaml")
    edges = infer_data_edges(descriptors, built)
    case_edges = [e for e in edges if e.to == "size"]
    assert any(e.from_ == "score" for e in case_edges)
    # ${input.weight} is a flow input, not an ${X.output} -> no data edge.
    assert not any(e.from_ == "weight" for e in case_edges)


@pytest.mark.parametrize(
    "when",
    [
        "score.output >= 0.5",        # bare (canonical)
        "${score.output} >= 0.5",     # mixed
        "${score.output >= 0.5}",     # whole-span
    ],
)
def test_case_when_produces_data_edge_in_every_spelling(when):
    # A searched `when:` is a condition, not a template: its refs must be extracted with
    # condition_refs (which parses the whole expression), so the score->gate data edge is
    # inferred even for the bare spelling (which has no ${...} span for the template walker).
    text = (
        "id: f\nname: f\ninput:\n  x: float\nnodes:\n"
        "  score:\n    kind: code\n    input:\n      x: ${input.x}\n"
        "    output: float\n    code: m:fn\n"
        "  gate:\n    kind: case\n    cases:\n"
        f'      - when: "{when}"\n        then: hi\n    else: lo\n'
        '  hi:\n    kind: agent\n    prompt: "hi"\n    output: str\n'
        '  lo:\n    kind: agent\n    prompt: "lo"\n    output: str\n'
        "output: ${hi.output | lo.output}\n"
    )
    f = parse_file(text)
    descriptors = parse_nodes(NodeReader.spanless(f.nodes))
    built = {
        nid: build_leaf_node(desc, {})[1]
        for nid, desc in descriptors.items()
        if isinstance(desc, (AgentDescriptor, CodeDescriptor))
    }
    edges = infer_data_edges(descriptors, built)
    assert any(e.from_ == "score" and e.to == "gate" for e in edges)


def test_no_end_edges_for_case_seed():
    # input-readers get START_ID input-producer edges, but no END_ID edge is minted here.
    descriptors, built = _seed("02-case.yaml")
    edges = infer_data_edges(descriptors, built)
    for e in edges:
        assert e.to != END_ID and e.from_ != END_ID
