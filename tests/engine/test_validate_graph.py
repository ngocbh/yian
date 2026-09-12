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
from agent_composer.compile.validation import types_compatible
from agent_composer.nodes.case import DEFAULT_HANDLE
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.edges.data import infer_data_edges
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.parse.file import parse_file
from agent_composer.compose.validate.cycles import reject_cycles
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.source import SourceDocument
from agent_composer.compose.validate.handles import check_case_handles
from agent_composer.typesys import Type, ValueKind

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _build(seed: str):
    """(node_ids, reconciled_edges, nodes) for a seed — every descriptor built kind-blind.

    Mirrors the loader's conductor loop: build each descriptor via `desc.build(BuildContext(...))`,
    collect its `BuildResult` (node + wiring + edges). A leaf's `.edges` is empty; a `case`'s
    `.edges` already carries its own reconciled data + control edges (its lowering is its build).
    `reconciled = infer_data_edges(leaf wiring) + all BuildResult.edges` (case edges last)."""
    path = _SEEDS / seed
    document = SourceDocument.parse(path.read_text(), label=str(path))
    parse_file(document)
    descriptors = parse_nodes(document.root.child("nodes"))
    nodes: dict = {}
    flow_wiring: dict = {}
    case_edges: list = []
    for nid, d in descriptors.items():
        result = d.build(BuildContext({}))
        nodes[nid] = result.node
        flow_wiring[nid] = result.wiring
        case_edges.extend(result.edges)
    data_edges = infer_data_edges(descriptors, flow_wiring)
    edges = data_edges + case_edges
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
    notes = exc.value.diagnostic.notes
    # Every note reads "<consumer> depends on <producer> (...)" and both stuck nodes appear
    # on both sides of the loop.
    assert notes, "a cycle must carry a dependency legend"
    assert all("depends on" in n for n in notes)
    joined = " ".join(notes)
    assert "a depends on b" in joined and "b depends on a" in joined


def test_acyclic_seed_passes_cycle_check():
    node_ids, edges, _ = _build("01-structured-agent.yaml")
    reject_cycles(edges, node_ids)  # no raise


# --------------------------------------------------------------------------- #
# recursive type compatibility
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source,sink,compatible", [
    ("int[2, 4]", "int[1, 12]", True),
    ("int[1, 12]", "int[2, 4]", False),
    ("int[1, 12]", "int[1, 12]", True),
    ("int[-2, 4]", "int[0, 12]", False),
    ("int[2, 13]", "int[1, 12]", False),
    ("int[4, 4]", "int[1, 12]", True),
    ("int", "int[1, 12]", False),
    ("int[1, 12]", "int", True),
    ("float[0.25, 0.75]", "float[0, 1]", True),
    ("float[0, 1]", "float[0.25, 0.75]", False),
    ("float", "float[0, 1]", False),
    ("float[0, 1]", "float", True),
    ("int[1, 2]", "float[0.5, 2.5]", True),
    ("int[0, 2]", "float[0.5, 2.5]", False),
    ("int", "float[0, 1]", False),
    ("int[0, 1]", "float", True),
    ("float[1, 1]", "int[1, 1]", False),
    ("Optional[int[1, 2]]", "int[0, 3]", False),
    ("int[1, 2]", "Optional[int[0, 3]]", True),
    ("Optional[int[1, 2]]", "Optional[float[0, 3]]", True),
    ("Optional[int]", "Optional[int[0, 3]]", False),
    ("None", "Optional[int[0, 3]]", True),
    ("None", "int[0, 3]", False),
])
def test_numeric_domain_compatibility_requires_interval_containment(source, sink, compatible):
    from agent_composer.typesys.types import type_for

    assert types_compatible(type_for(source, {}), type_for(sink, {})) is compatible


@pytest.mark.parametrize("source,sink,compatible", [
    ("list[int[2, 4]]", "list[float[1, 12]]", True),
    ("list[int]", "list[int[1, 12]]", False),
    ("list[int[0, 4]]", "list[float[1, 12]]", False),
])
def test_numeric_domain_compatibility_recurses_through_records_and_lists(source, sink, compatible):
    from agent_composer.typesys.types import type_for

    left = Type(ValueKind.OBJECT, fields={"values": type_for(source, {})},
                required=frozenset({"values"}))
    right = Type(ValueKind.OBJECT, fields={"values": type_for(sink, {})},
                 required=frozenset({"values"}))
    assert types_compatible(left, right) is compatible


def test_list_element_compatibility_preserves_nullable_direction() -> None:
    """A source list may narrow but cannot widen the sink's element contract."""
    strict_element = Type.scalar(ValueKind.INTEGER)
    nullable_element = Type(kind=ValueKind.INTEGER, nullable=True)
    strict = Type(kind=ValueKind.LIST_INTEGER, element=strict_element)
    nullable = Type(kind=ValueKind.LIST_INTEGER, element=nullable_element)

    assert types_compatible(strict, nullable) is True
    assert types_compatible(nullable, nullable) is True
    assert types_compatible(nullable, strict) is False


def test_none_source_is_compatible_exactly_with_none_or_nullable_sinks() -> None:
    """Static compatibility mirrors runtime acceptance of the singleton None value."""
    none = Type.scalar(ValueKind.NONE)
    strict_string = Type.scalar(ValueKind.STRING)
    nullable_string = Type(kind=ValueKind.STRING, nullable=True)

    assert types_compatible(none, none) is True
    assert types_compatible(none, nullable_string) is True
    assert types_compatible(none, strict_string) is False

    source_record = Type(
        kind=ValueKind.OBJECT,
        fields={"note": none},
        required=frozenset({"note"}),
    )
    sink_record = Type(
        kind=ValueKind.OBJECT,
        fields={"note": nullable_string},
        required=frozenset({"note"}),
    )
    assert types_compatible(source_record, sink_record) is True


def test_bare_list_compatibility_uses_its_storage_element_contract() -> None:
    """Bare specialized lists stay strict while bare LIST_ANY stays unconstrained."""
    strict_element = Type.scalar(ValueKind.INTEGER)
    nullable_element = Type(kind=ValueKind.INTEGER, nullable=True)
    bare_integer = Type(kind=ValueKind.LIST_INTEGER)
    strict_integer = Type(
        kind=ValueKind.LIST_INTEGER,
        element=strict_element,
    )
    nullable_integer = Type(
        kind=ValueKind.LIST_INTEGER,
        element=nullable_element,
    )
    bare_any = Type(kind=ValueKind.LIST_ANY)
    refined_any = Type(kind=ValueKind.LIST_ANY, element=strict_element)

    assert types_compatible(bare_integer, nullable_integer) is True
    assert types_compatible(strict_integer, bare_integer) is True
    assert types_compatible(nullable_integer, bare_integer) is False
    assert types_compatible(refined_any, bare_any) is True
    assert types_compatible(bare_any, refined_any) is False


def test_list_element_compatibility_recurses_before_storage_kind_rejection() -> None:
    """A list of None values may fill a list whose elements are nullable."""
    none = Type.scalar(ValueKind.NONE)
    nullable_string = Type(kind=ValueKind.STRING, nullable=True)
    source = Type(kind=ValueKind.LIST_ANY, element=none)
    sink = Type(kind=ValueKind.LIST_STRING, element=nullable_string)

    assert types_compatible(source, sink) is True


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
