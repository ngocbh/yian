"""Cross-layer compatibility tests for provenance-bearing semantic IR values."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_composer.compile.model import END_ID, FlowOutput
from agent_composer.compose.build.boundary import synthesize_boundary_graph
from agent_composer.compose.build.inline_inputs import infer_inline_binding_type
from agent_composer.compose.edges.data import infer_data_edges
from agent_composer.compose.edges.producers import _binding_producers
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.types import InputDecl
from agent_composer.compose.nodes.case import _data_edges as case_data_edges
from agent_composer.compose.nodes.resolver import _output_value_type
from agent_composer.compose.validate.call_types import check_ref_map_types
from agent_composer.compose.validate.references import validate_references
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import Binding
from agent_composer.typesys import Type, ValueKind


def _origin(key: str) -> OriginRef:
    """Return one short opaque provenance key for semantic-IR tests."""
    return OriginRef(key)


def test_data_edge_inference_is_identical_for_raw_and_wrapped_sources() -> None:
    """Carrier provenance must not change producers, groups, or optionality."""
    source = "${producer.output}"
    descriptors = {"sink": object()}
    raw = infer_data_edges(descriptors, {"sink": {"value": source}})
    wrapped = infer_data_edges(
        descriptors,
        {"sink": {"value": Binding(source, _origin("sink-value"))}},
    )

    assert wrapped == raw
    assert [(edge.from_, edge.to, edge.input_group, edge.optional) for edge in raw] == [
        ("producer", "sink", "value", False)
    ]


def test_producer_scan_unwraps_exactly_one_carrier_layer() -> None:
    """A nested carrier remains an opaque payload rather than executable wiring."""
    inner = Binding("${producer.output}", _origin("inner"))

    assert _binding_producers(Binding(inner, _origin("outer"))) == []


def test_boundary_reads_wrapped_output_but_preserves_it_in_end_wiring() -> None:
    """Boundary dependencies use source semantics without discarding provenance."""
    wrapped = Binding("${producer.output}", _origin("flow-output"))
    output = FlowOutput(name="result", from_=wrapped)

    _nodes, edges, wiring = synthesize_boundary_graph(
        [],
        [output],
        {"producer"},
        [],
    )

    assert any(
        edge.from_ == "producer"
        and edge.to == END_ID
        and edge.input_group == "result"
        and edge.optional is False
        for edge in edges
    )
    assert wiring[END_ID]["result"] is wrapped


def test_reference_validation_checks_a_wrapped_flow_output() -> None:
    """A carrier cannot hide unknown node or output namespaces from validation."""
    bad = Binding("${missing.output}", _origin("bad-reference"))

    with pytest.raises(LoadError, match="unknown namespace"):
        validate_references(
            {"sink": Node("sink")},
            set(),
            {},
            None,
            {"sink": {"value": bad}},
        )
    with pytest.raises(LoadError, match="unknown namespace"):
        validate_references(
            {},
            set(),
            {},
            bad,
            {},
        )

    validate_references(
        {"sink": Node("sink")},
        set(),
        {},
        None,
        {"sink": {"value": Binding(bad, _origin("outer"))}},
    )


def test_output_type_resolution_is_identical_for_raw_and_wrapped_sources() -> None:
    """Child signatures and call/loop checks see the carrier's semantic source."""
    output_type = Type.scalar(ValueKind.STRING)
    source = "${producer.output}"
    producers = {"producer": output_type}

    assert _output_value_type(source, producers, {}) is output_type
    assert (
        _output_value_type(Binding(source, _origin("typed-output")), producers, {})
        is output_type
    )
    assert (
        _output_value_type(
            Binding(Binding(source, _origin("inner")), _origin("outer")),
            producers,
            {},
        )
        is None
    )


def test_call_type_error_is_identical_for_raw_and_wrapped_sources() -> None:
    """Nullable-call compatibility reads the source expression, not carrier repr."""
    child_type = Type.scalar(ValueKind.STRING)
    nullable_parent = Type(kind=ValueKind.STRING, nullable=True)
    child_input = InputDecl(
        name="topic",
        type_str="str",
        default=None,
        required=True,
        type=child_type,
    )
    node = SimpleNamespace(
        is_loop=False,
        child_inputs=[child_input],
        field_ranges={},
        range=None,
    )

    source = "${input.x}"

    errors: list[str] = []
    for value in (source, Binding(source, _origin("call-topic"))):
        with pytest.raises(LoadError) as caught:
            check_ref_map_types(
                {"call": node},
                {},
                {"x": nullable_parent},
                {"call": {"topic": value}},
            )
        errors.append(str(caught.value))

    assert errors[0] == errors[1]
    assert "child expects 'str', source is 'str'" in errors[0]


def test_case_edge_inference_reads_wrapped_lowered_wiring() -> None:
    """CASE-owned data dependencies cannot disappear behind a carrier."""
    edges = case_data_edges(
        "gate",
        {"__on": Binding("${producer.output}", _origin("case-on"))},
    )

    assert [
        (edge.from_, edge.to, edge.input_group, edge.optional) for edge in edges
    ] == [("producer", "gate", "__on", False)]


def test_inline_binding_type_inference_reads_one_carrier_layer() -> None:
    """Generated inline parameters retain the type of a wrapped parent reference."""
    parent_type = Type.scalar(ValueKind.INTEGER)

    inferred = infer_inline_binding_type(
        Binding("${input.count}", _origin("inline-count")),
        parent_input_types={"count": parent_type},
        producer_type=lambda _node, _label, _range: Type.scalar(ValueKind.NONE),
        binding_label="call.input.count",
    )

    assert inferred == parent_type

    with pytest.raises(LoadError, match="unsupported type 'Binding'"):
        infer_inline_binding_type(
            Binding(
                Binding("${input.count}", _origin("inner")),
                _origin("outer"),
            ),
            parent_input_types={"count": parent_type},
            producer_type=lambda _node, _label, _range: Type.scalar(ValueKind.NONE),
            binding_label="call.input.count",
        )
