"""Unit tests for infer_data_edges — data-edge inference + input_group.

A flow-level pass: for every SINK binding in the flow — each built node's
`inputs` (the InputBindings from the binding pass) and a mapped `call`'s `over:` — collect its
reference paths via `expr_refs_of` (over `${...}` spans). Each `${<id>[.output.…]}` ref becomes
a data `Edge(from_=<id>, to=<consumer>, input_group=<sink key>)`.

A `case` node is SKIPPED here (the pass `continue`s on it): a case owns its lowering in
`CaseDescriptor.build`, which emits its `__rN`/`__on` data edges directly. So `infer_data_edges`
mints no edge to/from a case node (the case data-edge coverage is `tests/engine/test_case.py`).

A coalesce `${a | b}` sink yields TWO edges that SHARE one `input_group` (the sink
input name). An `${input.X}` read mints a `START_ID->reader` input-producer DATA edge;
`${system.X}`/`${item}` mint none. Boundary assembly, not this pass, synthesizes `->END_ID` edges.
"""

from pathlib import Path

from yian.compile.model import END_ID, START_ID
from yian.compose.nodes.base import BuildContext
from yian.compose.edges.data import infer_data_edges
from yian.compose.parse.file import (
    AgentDescriptor,
    CodeDescriptor,
    parse_file,
)
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceCursor, SourceDocument

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _wiring(desc):
    """The flow-owned wiring of a built leaf descriptor (unwrapping BuildResult)."""
    return desc.build(BuildContext({})).wiring


def _seed(name: str):
    """Return descriptors and leaf wiring for a seed, excluding case/call/map builds."""
    path = _SEEDS / name
    document = SourceDocument.parse(path.read_text(), label=str(path))
    parse_file(document)
    descriptors = parse_nodes(document.root.child("nodes"))
    flow_wiring = {
        nid: _wiring(desc)
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
        SourceCursor.virtual(
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
    wiring = {nid: _wiring(d) for nid, d in descriptors.items()}
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
        SourceCursor.virtual(
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
    # `CallDescriptor.build` creates MAP wiring and needs a resolver; supply it directly —
    # the `over` source rides the reserved "over" key, over-then-inputs order.
    wiring = {
        "bundle": _wiring(descriptors["bundle"]),
        "each": {"over": "${bundle.output}", "topic": "${item}"},
    }
    edges = infer_data_edges(descriptors, wiring)
    tuples = _edge_tuples(edges)
    assert ("bundle", "each", "over") in tuples
    # ${item} is a body-local scope, not a pool ref -> no edge.
    assert not any(e.from_ == "item" for e in edges)


# ---------- a case node's data edges are NO LONGER inferred here ----------
# The case owns its lowering in `CaseDescriptor.build`, which emits the reconciled
# `__rN`/`__on` data edges directly (its `BuildResult.edges`). `infer_data_edges` therefore
# `continue`s on a case descriptor — it emits NO edge to/from a case node. (The case data-edge
# coverage lives in `tests/engine/test_case.py`.)


def test_case_on_ref_produces_no_infer_edge():
    # seed 06: route is a case with on: ${classify.output}. infer_data_edges emits no edge
    # into route (the classify->route edge is the case build's, not inferred here).
    descriptors, built = _seed("06-case-on.yaml")
    edges = infer_data_edges(descriptors, built)
    assert not any(e.to == "route" for e in edges)


def test_case_searched_when_ref_produces_no_infer_edge():
    # seed 02: gate is a case; infer_data_edges emits no edge into gate.
    descriptors, built = _seed("02-case.yaml")
    edges = infer_data_edges(descriptors, built)
    assert not any(e.to == "gate" for e in edges)


def test_case_multiref_when_produces_no_infer_edge():
    # seed 10: size is a case; infer_data_edges emits no edge into size.
    descriptors, built = _seed("10-asserts-arithmetic.yaml")
    edges = infer_data_edges(descriptors, built)
    assert not any(e.to == "size" for e in edges)


def test_case_when_produces_no_infer_edge_in_every_spelling():
    # A searched `when:` in any spelling yields no infer_data_edges edge into the case node
    # (the case build owns the score->gate edge). One flow per spelling.
    for when in ("score.output >= 0.5", "${score.output >= 0.5}"):
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
        document = SourceDocument.parse(text)
        parse_file(document)
        descriptors = parse_nodes(document.root.child("nodes"))
        built = {
            nid: _wiring(desc)
            for nid, desc in descriptors.items()
            if isinstance(desc, (AgentDescriptor, CodeDescriptor))
        }
        edges = infer_data_edges(descriptors, built)
        assert not any(e.to == "gate" for e in edges), when


def test_no_end_edges_for_case_seed():
    # input-readers get START_ID input-producer edges, but no END_ID edge is minted here.
    descriptors, built = _seed("02-case.yaml")
    edges = infer_data_edges(descriptors, built)
    for e in edges:
        assert e.to != END_ID and e.from_ != END_ID
