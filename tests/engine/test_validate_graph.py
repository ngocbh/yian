"""Unit tests for graph validation — cycle + CASE handle alignment.

`reject_cycles` wraps the shared `compile.validation._reject_cycles` (now `(edges,
node_ids)`) on the synthesized edges; a cyclic inferred graph (`errors/e02`) is a
loud `LoadError` naming the stuck nodes. `check_case_handles` ports the legacy
handle-alignment rule to the desugared `CaseNode`s + their control edges (every
case handle has an outgoing edge; every edge handle is a case handle or `default`).
"""

from pathlib import Path

import pytest

from agent_composer.compile.model import Edge
from agent_composer.nodes.case import DEFAULT_HANDLE
from agent_composer.compose.build import build_leaf_node, infer_data_edges
from agent_composer.compose.cases import desugar_case, reconcile_case_edges
from agent_composer.compose.errors import LoadError
from agent_composer.compose.parser import (
    AgentDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    parse_file,
)
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.stages.parse import parse_nodes
from agent_composer.compose.validate import check_case_handles, reject_cycles

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _build(seed: str):
    f = parse_file((_SEEDS / seed).read_text())
    descriptors = parse_nodes(NodeReader.spanless(f.nodes))
    built = {
        nid: build_leaf_node(d, {})
        for nid, d in descriptors.items()
        if isinstance(d, (AgentDescriptor, CodeDescriptor))
    }  # {nid: (node, wiring)}
    leaf = {nid: n for nid, (n, _) in built.items()}
    flow_wiring = {nid: w for nid, (_, w) in built.items()}
    step8 = infer_data_edges(descriptors, flow_wiring)
    desugars = {
        nid: desugar_case(d, {})
        for nid, d in descriptors.items()
        if isinstance(d, CaseDescriptor)
    }
    edges = reconcile_case_edges(step8, desugars)
    nodes = dict(leaf)
    for nid, d in desugars.items():
        nodes[nid] = d.node
    return set(descriptors), edges, nodes


# --------------------------------------------------------------------------- #
# cycle detection on the inferred graph
# --------------------------------------------------------------------------- #


def test_e02_cycle_is_loud():
    node_ids, edges, _ = _build("errors/e02-cycle.yaml")
    with pytest.raises(LoadError) as exc:
        reject_cycles(edges, node_ids)
    msg = str(exc.value)
    assert "cycle" in msg
    assert "a" in msg and "b" in msg  # the stuck nodes are named


def test_e02_cycle_carries_dependency_legend():
    """The cycle error's `.notes` spell out each loop-internal dependency edge."""
    node_ids, edges, _ = _build("errors/e02-cycle.yaml")
    with pytest.raises(LoadError) as exc:
        reject_cycles(edges, node_ids)
    notes = exc.value.notes or []
    # Every note reads "<consumer> depends on <producer> (...)" and both stuck nodes appear
    # on both sides of the loop.
    assert notes, "a cycle must carry a dependency legend"
    assert all("depends on" in n for n in notes)
    joined = " ".join(notes)
    assert "a depends on b" in joined and "b depends on a" in joined


def test_acyclic_seed_passes_cycle_check():
    node_ids, edges, _ = _build("01-structured-agent.yaml")
    reject_cycles(edges, node_ids)  # no raise


def test_case_seed_passes_cycle_check():
    node_ids, edges, _ = _build("02-case.yaml")
    reject_cycles(edges, node_ids)  # control edges don't introduce a cycle


# --------------------------------------------------------------------------- #
# CASE handle alignment over the desugared cases + control edges
# --------------------------------------------------------------------------- #


def test_seed02_handles_align():
    node_ids, edges, nodes = _build("02-case.yaml")
    check_case_handles(nodes, edges)  # no raise


def test_seed06_handles_align():
    node_ids, edges, nodes = _build("06-case-on.yaml")
    check_case_handles(nodes, edges)  # no raise


def test_handle_with_no_matching_case_rejected():
    node_ids, edges, nodes = _build("02-case.yaml")
    # add a stray control edge with a handle that is not a case of `gate`.
    edges = edges + [Edge(id="gate->x#9", from_="gate", to="positive", source_handle="ghost")]
    with pytest.raises(LoadError) as exc:
        check_case_handles(nodes, edges)
    assert "ghost" in str(exc.value)


def test_case_without_outgoing_edge_rejected():
    node_ids, edges, nodes = _build("02-case.yaml")
    # drop the gate->positive control edge -> the `positive` case has no outgoing edge.
    edges = [e for e in edges if not (e.from_ == "gate" and e.source_handle == "positive")]
    with pytest.raises(LoadError) as exc:
        check_case_handles(nodes, edges)
    assert "positive" in str(exc.value)


def test_default_handle_always_allowed():
    node_ids, edges, nodes = _build("02-case.yaml")
    # the else: -> default handle is on a control edge; it must not trip the check.
    default_edges = [
        e for e in edges if e.from_ == "gate" and e.source_handle == DEFAULT_HANDLE
    ]
    assert default_edges  # seed 02 has an else:
    check_case_handles(nodes, edges)  # no raise
