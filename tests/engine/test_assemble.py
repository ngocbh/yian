"""Unit tests for the DAG assembly — the synthesized START_ID/END_ID boundary.

Runs AFTER data edges + case control edges. Roots are body nodes with
NO incoming edge of ANY kind (data OR control), so a `case` branch target (its only
incoming edge is the `gate->target` control edge) is correctly demoted from root —
otherwise the engine enqueues both branches unconditionally. Terminals come from the
flow `outputs:` bindings: `synthesize_boundary_graph` mints a `P -> __end__` edge per
producer per output (a coalesce flow-output -> one edge per producer), keyed by output name.
"""

from pathlib import Path

import pytest

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.typesys.types import read_typedefs
from agent_composer.typesys.values import Type, ValueKind
from agent_composer.compose import LoadError
from agent_composer.compose.helpers.types import InputDecl
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.build.boundary import synthesize_boundary_graph
from agent_composer.compose.edges.data import infer_data_edges
from agent_composer.compose.validate.wiring_parity import check_wiring_parity
from agent_composer.compose.build.pipeline import _assemble, _flow_outputs
from agent_composer.compose.parse.file import (
    AgentDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    WaitDescriptor,
    parse_file,
)
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.parse.nodes import parse_nodes

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _assemble_resolved_input_decls(input_decls):
    """Assemble an empty flow through the already-resolved input seam."""
    def inert_resolver(name):
        raise AssertionError(f"empty flow unexpectedly resolved {name!r}")

    return _assemble(
        input_decls=input_decls,
        nodes_reader=NodeReader.spanless({}),
        outputs_section=None,
        asserts_section=[],
        registry={},
        resolver=inert_resolver,
        n_lines={},
        s_lines={},
    )


def test_resolved_input_decls_preserve_structural_declaration_identity():
    element_type = Type(
        kind=ValueKind.OBJECT,
        fields={"label": Type(kind=ValueKind.STRING, nullable=True)},
        required=frozenset(),
    )
    structural_type = Type(
        kind=ValueKind.LIST_OBJECT,
        element=element_type,
    )
    declaration = InputDecl(
        name="items",
        type_str="list[object]",
        default=None,
        required=True,
        type=structural_type,
    )

    flow = _assemble_resolved_input_decls([declaration])
    start = flow.compiled.nodes[START_ID]

    assert flow.input[0] is declaration
    assert flow.input[0].type is structural_type
    assert start.input_decls[0] is declaration
    assert start.input_decls[0].type is structural_type


def test_resolved_input_decls_reject_both_input_representations():
    with pytest.raises(ValueError, match="exactly one of inputs_section or input_decls"):
        _assemble(
            inputs_section={},
            input_decls=[],
            nodes_reader=NodeReader.spanless({}),
            outputs_section=None,
            asserts_section=[],
            registry={},
            resolver=lambda name: None,
            n_lines={},
            s_lines={},
        )


def test_resolved_input_decls_reject_missing_input_representation():
    with pytest.raises(ValueError, match="exactly one of inputs_section or input_decls"):
        _assemble(
            nodes_reader=NodeReader.spanless({}),
            outputs_section=None,
            asserts_section=[],
            registry={},
            resolver=lambda name: None,
            n_lines={},
            s_lines={},
        )


def test_resolved_input_decls_accept_zero_args_and_wire_empty_start():
    flow = _assemble_resolved_input_decls([])
    start = flow.compiled.nodes[START_ID]

    assert flow.input == []
    assert start.input_decls == []
    assert start.params == []
    assert flow.compiled.wiring[START_ID] == {}

    raw_flow = _assemble(
        inputs_section={},
        nodes_reader=NodeReader.spanless({}),
        outputs_section=None,
        asserts_section=[],
        registry={},
        resolver=lambda name: None,
        n_lines={},
        s_lines={},
    )
    assert raw_flow.input == []
    assert raw_flow.compiled.wiring[START_ID] == {}


def _end_edges(f, node_ids, edges):
    """The END_ID producer edges synthesize_boundary_graph mints for the flow `outputs:` (the old
    synthesize_terminals' `P -> __end__` edges, now keyed by output name)."""
    _, all_edges, _ = synthesize_boundary_graph(
        [], _flow_outputs(f.outputs), node_ids, list(edges)
    )
    return [e for e in all_edges if e.to == END_ID]


def _all_edges(seed: str):
    """Data edges reconciled with the case builds' control edges.

    Returns (node_ids, edges) — `node_ids` is the runtime node-id set (a `case` builds to a
    CaseNode under the SAME id), `edges` is every real-node edge. Leaf wiring is built for
    the data-edge inference; a `case`'s own `build` emits its reconciled data+control edges
    (its `BuildResult.edges`), appended after the leaf data edges — mirrors the loader.
    """
    _p = (_SEEDS / seed) if (_SEEDS / seed).exists() else (_SEEDS / "_future" / seed)
    f = parse_file(_p.read_text())
    registry = read_typedefs(f.typedefs)
    descriptors = parse_nodes(NodeReader.spanless(f.nodes))
    flow_wiring = {
        nid: d.build(BuildContext(registry)).wiring
        for nid, d in descriptors.items()
        if isinstance(d, (AgentDescriptor, CodeDescriptor))
    }
    data_edges = infer_data_edges(descriptors, flow_wiring)
    case_edges = []
    for nid, d in descriptors.items():
        if isinstance(d, CaseDescriptor):
            case_edges.extend(d.build(BuildContext(registry)).edges)
    edges = data_edges + case_edges
    node_ids = set(descriptors)  # a case's node id == its built CaseNode id
    return node_ids, edges, f


def _bare_roots(node_ids, edges):
    """The set of body nodes `synthesize_boundary_graph` marks as bare roots.

    A bare root is a `START_ID -> nid` edge with no `input_group` (a data-producer edge
    carries `input_group`; a control edge carries `source_handle`). Filters the boundary
    edges for `from_ == START_ID and input_group is None and to != END_ID`.
    """
    _, all_edges, _ = synthesize_boundary_graph([], [], node_ids, list(edges))
    return {
        e.to
        for e in all_edges
        if e.from_ == START_ID and e.to != END_ID and e.input_group is None
    }


# --------------------------------------------------------------------------- #
# 10a — roots / input-producer edges: an ${input.X} reader gets a START_ID->reader
# DATA edge — NOT a bare root; a case branch target stays control-gated. infer_data_edges now
# mints the input-producer edges, so the boundary synthesis (no-incoming) yields no bare body root.
# --------------------------------------------------------------------------- #


def test_seed01_score_is_input_producer_not_bare_root():
    node_ids, edges, _ = _all_edges("01-structured-agent.yaml")
    # `score` reads ${input.topic} -> a START_ID->score DATA edge (input_group set); it is NOT a
    # bare root (boundary synthesis, which excludes any node with incoming, yields it no bare root edge).
    assert any(e.from_ == START_ID and e.to == "score" and e.input_group is not None for e in edges)
    assert "score" not in _bare_roots(node_ids, edges)


def test_seed02_case_targets_not_bare_roots():
    # positive/cautious are control-gated branch targets (gate->positive/cautious); neither is a
    # BARE root. (They also read ${input.topic} -> a START_ID data edge, but the control veto
    # still skip-floods the untaken branch.) `score` reads ${input.topic} too -> START_ID data edge.
    node_ids, edges, _ = _all_edges("02-case.yaml")
    bare_roots = _bare_roots(node_ids, edges)
    assert "positive" not in bare_roots and "cautious" not in bare_roots
    assert "gate" not in bare_roots  # gate has the score->gate data edge


def test_seed02_branch_target_has_control_edge():
    node_ids, edges, _ = _all_edges("02-case.yaml")
    control = [e for e in edges if e.to == "positive" and e.source_handle is not None]
    assert len(control) == 1 and control[0].from_ == "gate"
    assert control[0].source_handle == "positive"


def test_seed06_notes_not_bare_roots():
    node_ids, edges, _ = _all_edges("06-case-on.yaml")
    bare_roots = _bare_roots(node_ids, edges)
    for note in ("pro_note", "con_note", "choppy_note"):
        assert note not in bare_roots


# --------------------------------------------------------------------------- #
# 10b — terminals from the flow outputs: bindings
# --------------------------------------------------------------------------- #


def test_seed01_multi_output_terminals():
    node_ids, edges, f = _all_edges("01-structured-agent.yaml")
    ends = _end_edges(f, node_ids, edges)
    producers = {e.from_ for e in ends}
    # output: {rating: ${score.output.rating}, verdict: ${verdict.output}}
    assert producers == {"score", "verdict"}
    for e in ends:
        assert e.to == END_ID


def test_seed02_coalesce_both_producers_terminal():
    node_ids, edges, f = _all_edges("02-case.yaml")
    ends = _end_edges(f, node_ids, edges)
    producers = {e.from_ for e in ends}
    # outputs: ${positive.output | cautious.output} -> BOTH get a __end__ edge.
    assert producers == {"positive", "cautious"}
    for e in ends:
        assert e.to == END_ID


def test_seed18_multi_output_with_coalesce():
    node_ids, edges, f = _all_edges("18-research-pipeline.yaml")
    ends = _end_edges(f, node_ids, edges)
    producers = {e.from_ for e in ends}
    # stance/confidence -> synth; note -> pro_note | con_note | neutral_note.
    assert producers == {"synth", "pro_note", "con_note", "neutral_note"}


def test_seed05_multi_output_terminals():
    node_ids, edges, f = _all_edges("05-call-map.yaml")
    ends = _end_edges(f, node_ids, edges)
    producers = {e.from_ for e in ends}
    # report -> compare; briefs -> research_each.
    assert producers == {"compare", "research_each"}


def test_terminal_edge_ids_unique():
    node_ids, edges, f = _all_edges("18-research-pipeline.yaml")
    ends = _end_edges(f, node_ids, edges)
    ids = [e.id for e in ends]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# the assembled edge set is acceptable to CompiledFlow.from_parts
# --------------------------------------------------------------------------- #


def test_assembled_edges_have_start_and_end():
    node_ids, edges, f = _all_edges("02-case.yaml")
    _, boundary_edges, _ = synthesize_boundary_graph(
        [], _flow_outputs(f.outputs), node_ids, list(edges)
    )
    full = edges + boundary_edges
    assert any(e.from_ == START_ID for e in full)
    assert any(e.to == END_ID for e in full)


# --------------------------------------------------------------------------- #
# flow.wiring / params key parity
# --------------------------------------------------------------------------- #


def _leaf(name="n", **inputs):
    result = CodeDescriptor(id=name, code="m:f", inputs=inputs).build(BuildContext({}))
    return result.node, result.wiring


def test_wiring_parity_accepts_matched_keys():
    node, wiring = _leaf(x="${input.x}", y="${input.y}")
    check_wiring_parity({"n": node}, {"n": wiring})  # params == wiring keys -> no raise


def test_wiring_parity_accepts_wait_reserved_until():
    result = WaitDescriptor(id="w", until="${input.t}").build(BuildContext({}))
    node, wiring = result.node, result.wiring
    check_wiring_parity({"w": node}, {"w": wiring})  # {"until"} reserved key accepted


def test_wiring_parity_rejects_orphan_source():
    node, wiring = _leaf(x="${input.x}")
    with pytest.raises(LoadError, match="key parity"):
        check_wiring_parity({"n": node}, {"n": {**wiring, "orphan": "${input.y}"}})


def test_wiring_parity_rejects_missing_source():
    node, _ = _leaf(x="${input.x}")  # node declares param x...
    with pytest.raises(LoadError, match="key parity"):
        check_wiring_parity({"n": node}, {"n": {}})  # ...but the wiring has no entry for it


def test_wiring_parity_rejects_node_absent_from_wiring():
    node, _ = _leaf(x="${input.x}")
    with pytest.raises(LoadError, match="key parity"):
        check_wiring_parity({"n": node}, {})  # whole node missing -> silent {} bind guarded
