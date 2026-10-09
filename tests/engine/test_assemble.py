"""Unit tests for the DAG assembly — the synthesized START_ID/END_ID boundary.

Runs AFTER data edges + case control edges. Roots are body nodes with
NO incoming edge of ANY kind (data OR control), so a `case` branch target (its only
incoming edge is the `gate->target` control edge) is correctly demoted from root —
otherwise the engine enqueues both branches unconditionally. Terminals come from the
flow `output:` bindings: `synthesize_boundary_graph` mints a `P -> __end__` edge per
producer per output (a coalesce flow-output -> one edge per producer), keyed by output name.
"""

from pathlib import Path

import pytest

from yian.compile.model import END_ID, START_ID
from yian.diagnostics import SourceRange
from yian.typesys.types import read_typedefs
from yian.typesys.values import Type, ValueKind
from yian.compose import LoadError
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.base import BuildContext, BuildResult
from yian.compose.build.boundary import synthesize_boundary_graph
from yian.compose.edges.data import infer_data_edges
from yian.compose.validate.wiring_parity import check_wiring_parity
from yian.compose.build.pipeline import (
    _DescriptorBuildCache,
    _assemble,
    _flow_outputs,
)
from yian.compose.parse.file import (
    AgentDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    WaitDescriptor,
    parse_file,
)
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceCursor, SourceDocument
from yian.compose.source.catalog import _SourceCatalogBuilder
from yian.nodes.contracts import Parameter, ValueContract

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _range_diagnostics() -> LoadDiagnostics:
    """Return source text large enough for synthetic descriptor ranges."""
    return LoadDiagnostics(label="flow.yaml", text=("x" * 80 + "\n") * 60)


class _FakeBuildDescriptor:
    """A kind-neutral descriptor whose callback controls one build attempt."""

    def __init__(self, node_id, calls, callback=None, *, range=None):
        self.node_id = node_id
        self.calls = calls
        self.callback = callback
        self.range = range

    def build(self, ctx):
        self.calls.append((self.node_id, ctx))
        if self.callback is not None:
            self.callback()
        return BuildResult(node=object(), wiring={"built": self.node_id})


class _InertCallableScope:
    """Callable-scope double for assembly tests with no composite nodes."""

    def __init__(self):
        self.catalog_builder = _SourceCatalogBuilder()

    def resolve(self, flow_id, version=None):
        raise AssertionError(f"empty flow unexpectedly resolved {flow_id!r}")

    def register_private(self, flow_id, owner, label, builder):
        raise AssertionError(f"empty flow unexpectedly registered {label!r}")


def test_descriptor_build_cache_reuses_forward_builds_in_source_order():
    calls = []
    cache = None
    descriptors = {
        "a": _FakeBuildDescriptor(
            "a", calls, lambda: cache.build_one("b", "a.input.x")
        ),
        "b": _FakeBuildDescriptor("b", calls),
    }
    ctx = BuildContext(registry={})
    cache = _DescriptorBuildCache(descriptors, ctx)

    built = cache.build_in_source_order()

    assert list(built) == ["a", "b"]
    assert [node_id for node_id, _ in calls] == ["a", "b"]
    assert all(actual_ctx is ctx for _, actual_ctx in calls)
    assert built["b"] is cache.build_one("b")
    assert [node_id for node_id, _ in calls] == ["a", "b"]


def test_descriptor_build_cache_reports_only_mutual_cycle_with_closing_range():
    calls = []
    closing_range = SourceRange(31, 9, 31, 27)
    cache = None
    descriptors = {
        "outside": _FakeBuildDescriptor(
            "outside", calls, lambda: cache.build_one("a", "outside.input.seed")
        ),
        "a": _FakeBuildDescriptor(
            "a", calls, lambda: cache.build_one("b", "a.input.x")
        ),
        "b": _FakeBuildDescriptor(
            "b",
            calls,
            lambda: cache.build_one("a", "b.input.y", closing_range),
        ),
    }
    cache = _DescriptorBuildCache(
        descriptors,
        BuildContext(registry={}, diagnostics=_range_diagnostics()),
    )

    with pytest.raises(LoadError) as caught:
        cache.build_in_source_order()

    assert str(caught.value).endswith("a.input.x -> b.input.y -> a")
    assert "outside.input.seed" not in str(caught.value)
    assert caught.value.diagnostic.frames[0].range == closing_range
    assert not isinstance(caught.value, RecursionError)


def test_descriptor_build_cycle_falls_back_to_first_cycle_edge_range():
    calls = []
    first_edge_range = SourceRange(17, 7, 17, 21)
    cache = None
    descriptors = {
        "a": _FakeBuildDescriptor(
            "a",
            calls,
            lambda: cache.build_one("b", "a.input.x", first_edge_range),
        ),
        "b": _FakeBuildDescriptor(
            "b", calls, lambda: cache.build_one("a", "b.input.y")
        ),
    }
    cache = _DescriptorBuildCache(
        descriptors,
        BuildContext(registry={}, diagnostics=_range_diagnostics()),
    )

    with pytest.raises(LoadError) as caught:
        cache.build_in_source_order()

    assert str(caught.value).endswith("a.input.x -> b.input.y -> a")
    assert caught.value.diagnostic.frames[0].range == first_edge_range


def test_descriptor_build_cycle_uses_node_ids_when_an_edge_has_no_label():
    calls = []
    cache = None
    descriptors = {
        "a": _FakeBuildDescriptor(
            "a", calls, lambda: cache.build_one("b", "a.input.x")
        ),
        "b": _FakeBuildDescriptor(
            "b", calls, lambda: cache.build_one("a")
        ),
    }
    cache = _DescriptorBuildCache(descriptors, BuildContext(registry={}))

    with pytest.raises(LoadError) as caught:
        cache.build_in_source_order()

    assert str(caught.value).endswith("a -> b -> a")


def test_descriptor_build_cache_does_not_cache_unlocated_failure():
    calls = []
    cache = None

    def fail():
        raise LoadDiagnostics().error("broken descriptor")

    descriptors = {
        "outer": _FakeBuildDescriptor(
            "outer", calls, lambda: cache.build_one("broken")
        ),
        "broken": _FakeBuildDescriptor("broken", calls, fail),
    }
    cache = _DescriptorBuildCache(descriptors, BuildContext(registry={}))

    for _ in range(2):
        with pytest.raises(LoadError, match="broken descriptor") as caught:
            cache.build_one("outer")
        assert caught.value.diagnostic.frames == ()

    assert [node_id for node_id, _ in calls] == [
        "outer",
        "broken",
        "outer",
        "broken",
    ]


def test_descriptor_build_cache_attaches_descriptor_range_to_unlocated_failure():
    calls = []
    source_range = SourceRange(27, 7, 27, 31)

    def fail():
        raise LoadDiagnostics().error("broken descriptor")

    descriptor = _FakeBuildDescriptor(
        "broken",
        calls,
        fail,
        range=source_range,
    )
    cache = _DescriptorBuildCache(
        {"broken": descriptor},
        BuildContext(registry={}, diagnostics=_range_diagnostics()),
    )

    with pytest.raises(LoadError, match="broken descriptor") as caught:
        cache.build_one("broken")

    assert caught.value.diagnostic.frames[0].range == source_range


def _assemble_resolved_parameters(parameters):
    """Assemble an empty flow through the already-resolved input seam."""
    return _assemble(
        input_parameters=parameters,
        nodes_cursor=SourceCursor.virtual({}),
        outputs_section=None,
        asserts_section=[],
        registry={},
        scope=_InertCallableScope(),
        owner=("test",),
    )


def test_resolved_parameters_preserve_structural_contract_identity():
    element_type = Type(
        kind=ValueKind.OBJECT,
        fields={"label": Type(kind=ValueKind.STRING, nullable=True)},
        required=frozenset(),
    )
    structural_type = Type(
        kind=ValueKind.LIST_OBJECT,
        element=element_type,
    )
    parameter = Parameter(
        name="items",
        contract=ValueContract(structural_type),
        required=True,
        coerce_strings=True,
    )

    flow = _assemble_resolved_parameters([parameter])
    start = flow.compiled.nodes[START_ID]

    assert flow.input is start.params
    assert flow.input[0] is parameter
    assert flow.input[0].contract is not None
    assert flow.input[0].contract.type is structural_type


def test_resolved_parameters_reject_both_input_representations():
    with pytest.raises(ValueError, match="exactly one of inputs_section or input_parameters"):
        _assemble(
            inputs_section={},
            input_parameters=[],
            nodes_cursor=SourceCursor.virtual({}),
            outputs_section=None,
            asserts_section=[],
            registry={},
            scope=_InertCallableScope(),
            owner=("test",),
        )


def test_resolved_parameters_reject_missing_input_representation():
    with pytest.raises(ValueError, match="exactly one of inputs_section or input_parameters"):
        _assemble(
            nodes_cursor=SourceCursor.virtual({}),
            outputs_section=None,
            asserts_section=[],
            registry={},
            scope=_InertCallableScope(),
            owner=("test",),
        )


def test_resolved_parameters_accept_zero_args_and_wire_empty_start():
    flow = _assemble_resolved_parameters([])
    start = flow.compiled.nodes[START_ID]

    assert flow.input is start.params
    assert start.params == []
    assert flow.compiled.wiring[START_ID] == {}

    raw_flow = _assemble(
        inputs_section={},
        nodes_cursor=SourceCursor.virtual({}),
        outputs_section=None,
        asserts_section=[],
        registry={},
        scope=_InertCallableScope(),
        owner=("test",),
    )
    assert raw_flow.input == []
    assert raw_flow.compiled.wiring[START_ID] == {}


def _end_edges(f, node_ids, edges):
    """Return END_ID producer edges synthesized from flow `output:` bindings."""
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
    document = SourceDocument.parse(_p.read_text(), label=str(_p))
    f = parse_file(document)
    registry = read_typedefs(f.typedefs)
    descriptors = parse_nodes(document.root.child("nodes"))
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
# Roots / input-producer edges: an ${input.X} reader gets a START_ID->reader
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
# Terminals from flow `output:` bindings
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
    # output: ${positive.output | cautious.output} -> both get a __end__ edge.
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
    return CodeDescriptor(id=name, code="m:f", inputs=inputs).build(BuildContext({}))


def test_wiring_parity_accepts_matched_keys():
    result = _leaf(x="${input.x}", y="${input.y}")
    check_wiring_parity({"n": result}, {"n": result.wiring})  # no raise


def test_wiring_parity_accepts_wait_reserved_until():
    result = WaitDescriptor(id="w", until="${input.t}").build(BuildContext({}))
    check_wiring_parity({"w": result}, {"w": result.wiring})  # reserved key accepted


def test_wiring_parity_rejects_orphan_source():
    result = _leaf(x="${input.x}")
    with pytest.raises(LoadError, match="key parity"):
        check_wiring_parity(
            {"n": result},
            {"n": {**result.wiring, "orphan": "${input.y}"}},
        )


def test_wiring_parity_rejects_missing_source():
    result = _leaf(x="${input.x}")  # node declares param x...
    with pytest.raises(LoadError, match="key parity"):
        check_wiring_parity({"n": result}, {"n": {}})  # wiring has no entry


def test_wiring_parity_rejects_node_absent_from_wiring():
    result = _leaf(x="${input.x}")
    with pytest.raises(LoadError, match="key parity"):
        check_wiring_parity({"n": result}, {})  # whole node missing
