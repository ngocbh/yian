"""Cross-layer compatibility tests for provenance-bearing semantic IR values."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_composer.compile.manifest import (
    expansion_digest,
    flow_fingerprint,
    flow_manifest,
)
from agent_composer.compile.model import (
    END_ID,
    START_ID,
    CompiledFlow,
    Edge,
    Flow,
    FlowOutput,
)
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
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.typesys import Type, ValueKind


def _origin(key: str) -> OriginRef:
    """Return one short opaque provenance key for semantic-IR tests."""
    return OriginRef(key)


def _manifest_fragment(value: object, *, key: str = "result") -> Flow:
    """Return a topology-only flow whose terminal wiring contains ``value``."""
    start = StartNode("entry", input_decls=[])
    end = EndNode.record("terminal", output_names=[key])
    return Flow(
        nodes={"entry": start, "terminal": end},
        edges=[Edge("entry->terminal", "entry", "terminal", input_group=key)],
        wiring={"entry": {}, "terminal": {key: value}},
        start_id="entry",
        end_id="terminal",
    )


def _manifest_compiled_output(value: object) -> CompiledFlow:
    """Return a compiled flow whose declared output metadata contains ``value``."""
    start = StartNode(START_ID, input_decls=[])
    end = EndNode.record(END_ID, output_names=["result"])
    return CompiledFlow.from_parts(
        nodes={START_ID: start, END_ID: end},
        edges=[Edge("boundary", START_ID, END_ID, ordering=True)],
        wiring={START_ID: {}, END_ID: {"result": "fixed"}},
        outputs=[FlowOutput("result", value)],
    )


def test_flow_construction_normalizes_direct_wiring_values() -> None:
    """Stored flow wiring is canonical without copying semantic payloads."""
    payload = {"literal": ["kept-by-identity"]}
    existing = Binding("${producer.output}", _origin("existing"))
    nested = Binding(existing, _origin("outer"))
    authored = {
        "sink": {
            "raw": payload,
            "existing": existing,
            "nested": nested,
        }
    }

    flow = Flow(nodes={"sink": Node("sink")}, edges=[], wiring=authored)

    assert flow.wiring is not authored
    assert flow.wiring["sink"] is not authored["sink"]
    assert flow.wiring["sink"]["raw"] == Binding(payload)
    assert flow.wiring["sink"]["raw"].source is payload
    assert flow.wiring["sink"]["existing"] is existing
    assert flow.wiring["sink"]["nested"] is nested


def test_compiled_flow_from_parts_normalizes_direct_wiring_values() -> None:
    """The public compiled-graph constructor delegates to the same boundary."""
    existing = Binding(7, _origin("count"))

    flow = CompiledFlow.from_parts(
        nodes={"sink": Node("sink")},
        edges=[],
        wiring={"sink": {"raw": 3, "existing": existing}},
    )

    assert flow.wiring["sink"]["raw"] == Binding(3)
    assert flow.wiring["sink"]["existing"] is existing


@pytest.mark.parametrize("constructor", ["flow", "compiled"])
@pytest.mark.parametrize(
    ("wiring", "message"),
    [
        ({1: {"value": "raw"}}, "node id"),
        ({"sink": {1: "raw"}}, "parameter name"),
    ],
)
def test_flow_construction_rejects_non_string_wiring_keys(
    constructor: str,
    wiring: dict[object, dict[object, object]],
    message: str,
) -> None:
    """Both model entry points reject keys outside the compiled IR contract."""
    nodes = {"sink": Node("sink")}

    with pytest.raises(TypeError, match=message):
        if constructor == "flow":
            Flow(nodes=nodes, edges=[], wiring=wiring)  # type: ignore[arg-type]
        else:
            CompiledFlow.from_parts(  # type: ignore[arg-type]
                nodes=nodes,
                edges=[],
                wiring=wiring,
            )


def test_add_subgraph_normalizes_wiring_and_preserves_existing_binding() -> None:
    """Programmatic growth enters the same canonical stored representation."""
    flow = CompiledFlow.from_parts(nodes={"base": Node("base")}, edges=[])
    existing = Binding("${base.output}", _origin("child-value"))

    flow.add_subgraph(
        nodes={"child": Node("child")},
        edges=[Edge("base->child", "base", "child", input_group="value")],
        wiring={"child": {"raw": 4, "existing": existing}},
    )

    assert flow.wiring["child"]["raw"] == Binding(4)
    assert flow.wiring["child"]["existing"] is existing


@pytest.mark.parametrize(
    ("wiring", "message"),
    [
        ({1: {"value": "raw"}}, "node id"),
        ({"child": {1: "raw"}}, "parameter name"),
    ],
)
def test_add_subgraph_rejects_bad_wiring_before_mutating_topology(
    wiring: dict[object, dict[object, object]],
    message: str,
) -> None:
    """An invalid splice cannot leave nodes, edges, or wiring partially installed."""
    base = Node("base")
    flow = CompiledFlow.from_parts(
        nodes={"base": base},
        edges=[],
        wiring={"base": {}},
    )
    before_nodes = dict(flow.nodes)
    before_edges = list(flow.edges)
    before_wiring = dict(flow.wiring)

    with pytest.raises(TypeError, match=message):
        flow.add_subgraph(  # type: ignore[arg-type]
            nodes={"child": Node("child")},
            edges=[Edge("base->child", "base", "child")],
            wiring=wiring,
        )

    assert flow.nodes == before_nodes
    assert flow.edges == before_edges
    assert flow.wiring == before_wiring


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


def test_wiring_identity_uses_source_and_ignores_binding_origin() -> None:
    """Manifest, fingerprint, and expansion identity exclude provenance only."""
    source = "${entry.output}"
    flows = [
        _manifest_fragment(source),
        _manifest_fragment(Binding(source, _origin("first"))),
        _manifest_fragment(Binding(source, _origin("second"))),
    ]
    changed = _manifest_fragment(Binding("changed", _origin("first")))

    manifests = [flow_manifest(flow) for flow in flows]
    fingerprints = [flow_fingerprint(flow) for flow in flows]
    digests = [expansion_digest(flow) for flow in flows]

    assert manifests[0] == manifests[1] == manifests[2]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
    assert digests[0] == digests[1] == digests[2]
    assert flow_manifest(changed) != manifests[0]
    assert flow_fingerprint(changed) != fingerprints[0]
    assert expansion_digest(changed) != digests[0]


def test_flow_output_identity_uses_source_and_ignores_binding_origin() -> None:
    """Declared-output metadata applies the same source-only identity projection."""
    source = "${__start__.output}"
    flows = [
        _manifest_compiled_output(source),
        _manifest_compiled_output(Binding(source, _origin("first"))),
        _manifest_compiled_output(Binding(source, _origin("second"))),
    ]
    changed = _manifest_compiled_output(Binding("changed", _origin("first")))

    manifests = [flow_manifest(flow) for flow in flows]
    fingerprints = [flow_fingerprint(flow) for flow in flows]

    assert manifests[0] == manifests[1] == manifests[2]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
    assert flow_manifest(changed) != manifests[0]
    assert flow_fingerprint(changed) != fingerprints[0]


@pytest.mark.parametrize("site", ["wiring", "output"])
def test_manifest_binding_policy_rejects_unknown_fields(site: str) -> None:
    """A future carrier field must receive an explicit canonical/ignored policy."""
    binding = Binding("value", _origin("site"))
    object.__setattr__(binding, "future_semantic_field", True)
    flow = (
        _manifest_fragment(binding)
        if site == "wiring"
        else _manifest_compiled_output(binding)
    )

    with pytest.raises(ValueError, match="future_semantic_field"):
        flow_manifest(flow)


@pytest.mark.parametrize("site", ["wiring", "output"])
def test_manifest_unwraps_exactly_one_binding_layer(site: str) -> None:
    """A nested carrier is unsupported literal data, not a second IR layer."""
    nested = Binding(Binding("value", _origin("inner")), _origin("outer"))
    flow = (
        _manifest_fragment(nested)
        if site == "wiring"
        else _manifest_compiled_output(nested)
    )

    with pytest.raises(TypeError, match="Binding"):
        flow_manifest(flow)


def test_wrapped_credential_wiring_keeps_redaction_and_source_identity() -> None:
    """Projecting a carrier must retain credential secrecy and ref identity."""
    first_source = '${input.primary_token:-"first-secret"}'
    second_source = '${input.backup_token:-"second-secret"}'
    raw = _manifest_fragment(first_source, key="authorization")
    wrapped = _manifest_fragment(
        Binding(first_source, _origin("credential")),
        key="authorization",
    )
    changed = _manifest_fragment(
        Binding(second_source, _origin("credential")),
        key="authorization",
    )

    raw_manifest = flow_manifest(raw)
    wrapped_manifest = flow_manifest(wrapped)
    rendered = repr(wrapped_manifest)

    assert wrapped_manifest == raw_manifest
    assert flow_fingerprint(wrapped) == flow_fingerprint(raw)
    assert "first-secret" not in rendered
    assert "credential" not in rendered
    assert flow_fingerprint(changed) != flow_fingerprint(wrapped)
