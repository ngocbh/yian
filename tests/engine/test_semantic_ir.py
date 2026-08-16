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
from agent_composer.compose.nodes.base import BuildContext, BuildResult
from agent_composer.compose.nodes.call import CallDescriptor
from agent_composer.compose.nodes.loop import LoopDescriptor
from agent_composer.compose.nodes.map import MapDescriptor
from agent_composer.compose.edges.data import infer_data_edges
from agent_composer.compose.edges.producers import _binding_producers
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.case import _data_edges as case_data_edges
from agent_composer.compose.nodes.resolver import (
    _output_value_contract,
    child_signature,
)
from agent_composer.compose.validate.call_types import check_ref_map_types
from agent_composer.compose.validate.references import validate_references
from agent_composer.compose import load_flow
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
)
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.typesys import Type, ValueKind


def _origin(key: str) -> OriginRef:
    """Return one short opaque provenance key for semantic-IR tests."""
    return OriginRef(key)


def _manifest_fragment(value: object, *, key: str = "result") -> Flow:
    """Return a topology-only flow whose terminal wiring contains ``value``."""
    start = StartNode("entry", params=[])
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
    start = StartNode(START_ID, params=[])
    end = EndNode.record(END_ID, output_names=["result"])
    return CompiledFlow.from_parts(
        nodes={START_ID: start, END_ID: end},
        edges=[Edge("boundary", START_ID, END_ID, ordering=True)],
        wiring={START_ID: {}, END_ID: {"result": "fixed"}},
        outputs=[FlowOutput("result", value)],
    )


def _manifest_assertions(assertions: list[object]) -> Flow:
    """Return a manifestable fragment whose END owns the supplied assertion storage."""
    flow = _manifest_fragment("value")
    flow.nodes["terminal"].post_asserts = assertions  # type: ignore[assignment]
    return flow


def test_node_origin_is_optional_validated_and_distinct_from_loop_attribution() -> None:
    """Opaque diagnostic provenance cannot be confused with execution `origin_id`."""
    source_origin = _origin("node/header")

    source_free = Node("source-free")
    authored = Node("authored", origin=source_origin)

    assert source_free.origin is None
    assert authored.origin is source_origin
    assert authored.origin_id is None
    with pytest.raises(TypeError, match="node origin must be an OriginRef or None"):
        Node("invalid", origin="node/header")  # type: ignore[arg-type]


def test_node_failure_trace_selects_local_origin_after_generated_context() -> None:
    """Failure ancestry remains ordered while operation precision wins locally."""
    outer = _origin("outer/call")
    node_origin = _origin("child/node")
    operation = _origin("child/input/value")
    node = Node("child", origin=node_origin)

    assert node.origin_context == ()
    assert node.failure_trace() == (node_origin,)
    assert node.failure_trace(operation) == (operation,)

    node.origin_context = (outer,)
    assert node.failure_trace() == (outer, node_origin)
    assert node.failure_trace(operation) == (outer, operation)


def test_node_failure_trace_preserves_source_free_context_and_duplicates() -> None:
    """Missing local origins omit the suffix without deduplicating outer ancestry."""
    repeated = _origin("shared/site")
    source_free = Node("source-free")

    assert source_free.failure_trace() == ()

    source_free.origin_context = (repeated, repeated)
    assert source_free.failure_trace() == (repeated, repeated)
    assert source_free.failure_trace(repeated) == (repeated, repeated, repeated)


def test_node_failure_trace_rejects_malformed_generated_origins() -> None:
    """Malformed generated metadata cannot cross into a canonical Failure."""

    class InvalidOrigin(OriginRef):
        def __post_init__(self) -> None:
            pass

    node = Node("node")
    node.origin_context = []  # type: ignore[assignment]
    with pytest.raises(TypeError, match="origin_context must be a tuple"):
        node.failure_trace()

    node.origin_context = ("outer",)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="origin_context item must be an OriginRef"):
        node.failure_trace()

    node.origin_context = (InvalidOrigin(""),)
    with pytest.raises(ValueError, match="origin key must be a non-empty string"):
        node.failure_trace()

    node.origin_context = ()
    with pytest.raises(TypeError, match="failure operation origin must be an OriginRef"):
        node.failure_trace("operation")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="origin key must be a non-empty string"):
        node.failure_trace(InvalidOrigin(""))


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


def test_flow_output_normalizes_one_binding_and_keeps_a_source_view() -> None:
    """Declared outputs own carriers while legacy reads expose only the source."""
    payload = {"literal": ["kept-by-identity"]}
    raw = FlowOutput("result", payload)
    existing = Binding("${producer.output}", _origin("declared-output"))
    wrapped = FlowOutput("result", existing)

    assert raw.binding == Binding(payload)
    assert raw.binding.source is payload
    assert raw.from_ is payload
    assert wrapped.binding is existing
    assert wrapped.from_ is existing.source

    with pytest.raises(AttributeError):
        wrapped.from_ = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        FlowOutput(name="result", from_="legacy-keyword")  # type: ignore[call-arg]


def test_loaded_compose_flow_has_only_canonical_binding_storage() -> None:
    """The YAML front end crosses into one uniform compiled representation."""
    loaded = load_flow(
        """
id: canonical-bindings
name: Canonical bindings
input:
  topic: str
nodes:
  emit:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${input.topic}
    output: str
output: ${emit.output}
"""
    )

    assert all(
        isinstance(binding, Binding)
        for node_wiring in loaded.compiled.wiring.values()
        for binding in node_wiring.values()
    )
    assert all(
        isinstance(output.binding, Binding) for output in loaded.compiled.outputs
    )


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
    output = FlowOutput(name="result", binding=wrapped)

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
    sink = BuildResult(node=Node("sink"), wiring={})

    with pytest.raises(LoadError, match="unknown namespace"):
        validate_references(
            {"sink": sink},
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
        {"sink": sink},
        set(),
        {},
        None,
        {"sink": {"value": Binding(bad, _origin("outer"))}},
    )


def test_output_contract_resolution_retains_whole_value_identity() -> None:
    """Raw and wrapped references resolve to the producer's exact carrier."""
    output_type = Type.scalar(ValueKind.STRING)
    contract = ValueContract(
        output_type,
        ContractOrigins(default=_origin("producer-output")),
    )
    source = "${producer.output}"
    producers = {"producer": contract}

    assert _output_value_contract(source, producers, {}) is contract
    assert (
        _output_value_contract(Binding(source, _origin("typed-output")), producers, {})
        is contract
    )
    assert (
        _output_value_contract(
            Binding(Binding(source, _origin("inner")), _origin("outer")),
            producers,
            {},
        )
        is None
    )
    assert _output_value_contract("${input.topic}", {}, {"topic": contract}) is contract

    computed = _output_value_contract(
        "${producer.output + 1}",
        {"producer": contract},
        {},
    )
    assert type(computed) is ValueContract
    assert computed is not contract
    assert computed.type is contract.type
    assert computed.origins == ContractOrigins()

    with pytest.raises(TypeError, match="flow input contracts.*ValueContract"):
        _output_value_contract(  # type: ignore[dict-item]
            "${input.topic}",
            {},
            {"topic": output_type},
        )


def test_output_contract_resolution_rebases_nested_origins() -> None:
    """Selecting a record field makes its origin root and strips its path prefix."""
    nested_type = Type(
        kind=ValueKind.OBJECT,
        fields={
            "answer": Type.scalar(ValueKind.INTEGER),
            "note": Type.scalar(ValueKind.STRING),
        },
        required=frozenset({"answer", "note"}),
    )
    root_type = Type(
        kind=ValueKind.OBJECT,
        fields={
            "result": nested_type,
            "ignored": Type.scalar(ValueKind.BOOLEAN),
        },
        required=frozenset({"result", "ignored"}),
    )
    contract = ValueContract(
        root_type,
        ContractOrigins(
            default=_origin("root"),
            entries=(
                (("ignored",), _origin("ignored")),
                (("result",), _origin("result")),
                (("result", "answer"), _origin("answer")),
                (("result", "note"), _origin("note")),
            ),
        ),
    )

    projected = _output_value_contract(
        "${producer.output.result}",
        {"producer": contract},
        {},
    )

    assert type(projected) is ValueContract
    assert projected.type is nested_type
    assert projected.origins == ContractOrigins(
        default=_origin("result"),
        entries=(
            (("answer",), _origin("answer")),
            (("note",), _origin("note")),
        ),
    )

    fallback_contract = ValueContract(
        root_type,
        ContractOrigins(
            default=_origin("root"),
            entries=((("result",), _origin("result")),),
        ),
    )
    leaf = _output_value_contract(
        "${producer.output.result.answer}",
        {"producer": fallback_contract},
        {},
    )
    assert leaf.origins.default == _origin("result")
    assert leaf.origins.entries == ()


def test_child_signature_prefixes_multi_output_contract_origins() -> None:
    """A record codomain retains each field contract under its output name."""
    alpha = ValueContract(
        Type(
            kind=ValueKind.OBJECT,
            fields={"leaf": Type.scalar(ValueKind.STRING)},
            required=frozenset({"leaf"}),
        ),
        ContractOrigins(
            default=_origin("alpha-root"),
            entries=((("leaf",), _origin("alpha-leaf")),),
        ),
    )
    beta = ValueContract(
        Type.scalar(ValueKind.INTEGER),
        ContractOrigins(default=_origin("beta-root")),
    )
    loaded = SimpleNamespace(
        input=[],
        compiled=SimpleNamespace(
            nodes={
                "alpha_node": Node("alpha_node", output_contract=alpha),
                "beta_node": Node("beta_node", output_contract=beta),
            },
            outputs=[
                FlowOutput("beta", "${beta_node.output}"),
                FlowOutput("alpha", "${alpha_node.output}"),
            ],
        ),
    )

    output = child_signature(loaded).output

    assert type(output) is ValueContract
    assert output.type.kind is ValueKind.OBJECT
    assert list(output.type.fields) == ["beta", "alpha"]
    assert output.type.fields == {"beta": beta.type, "alpha": alpha.type}
    assert output.origins == ContractOrigins(
        entries=(
            (("alpha",), _origin("alpha-root")),
            (("alpha", "leaf"), _origin("alpha-leaf")),
            (("beta",), _origin("beta-root")),
        )
    )


def test_composite_builders_preserve_or_deliberately_wrap_child_contract() -> None:
    """CALL/LOOP retain identity; MAP creates a default-only list contract."""
    contract = ValueContract(
        Type(
            kind=ValueKind.OBJECT,
            fields={"state": Type.scalar(ValueKind.INTEGER)},
            required=frozenset({"state"}),
        ),
        ContractOrigins(
            default=_origin("child-root"),
            entries=((("state",), _origin("child-state")),),
        ),
    )
    child = SimpleNamespace(
        input=[],
        compiled=SimpleNamespace(
            nodes={"producer": Node("producer", output_contract=contract)},
            outputs=[FlowOutput("result", "${producer.output}")],
        ),
    )
    ctx = BuildContext(registry={}, resolver=lambda _flow_id, _version: child)

    call = CallDescriptor(id="call", call="child").build(ctx).node
    loop = LoopDescriptor(
        id="loop",
        call="child",
        inputs={"state": "${input.state}"},
        times=1,
    ).build(ctx).node
    mapped = MapDescriptor(
        id="mapped",
        call="child",
        over="${input.items}",
    ).build(ctx).node

    assert call.output_contract is contract
    assert loop.output_contract is contract
    assert type(mapped.output_contract) is ValueContract
    assert mapped.output_contract is not contract
    assert mapped.output_contract.type.element is contract.type
    assert mapped.output_contract.origins == ContractOrigins(
        default=_origin("child-root")
    )


def test_call_type_error_is_identical_for_raw_and_wrapped_sources() -> None:
    """Nullable-call compatibility reads the source expression, not carrier repr."""
    child_type = Type.scalar(ValueKind.STRING)
    nullable_parent = Type(kind=ValueKind.STRING, nullable=True)
    child_input = Parameter(
        name="topic",
        contract=ValueContract(child_type),
        required=True,
        coerce_strings=True,
    )
    node = SimpleNamespace(
        is_loop=False,
        child_inputs=[child_input],
    )
    result = BuildResult(node=node, wiring={})

    source = "${input.x}"

    errors: list[str] = []
    for value in (source, Binding(source, _origin("call-topic"))):
        with pytest.raises(LoadError) as caught:
            check_ref_map_types(
                {"call": result},
                {},
                {"x": ValueContract(nullable_parent)},
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


@pytest.mark.parametrize("site", ["wiring", "output"])
def test_manifest_rejects_raw_values_that_bypass_construction(site: str) -> None:
    """Identity serialization fails loudly when mutation violates canonical IR."""
    if site == "wiring":
        flow = _manifest_fragment("value")
        flow.wiring["terminal"]["result"] = "bypass"  # type: ignore[assignment]
    else:
        flow = _manifest_compiled_output("value")
        object.__setattr__(flow.outputs[0], "binding", "bypass")

    with pytest.raises(TypeError, match="expected Binding"):
        flow_manifest(flow)


def test_assertion_identity_uses_expression_and_display_but_ignores_origin() -> None:
    """All identity surfaces retain behavior/presentation and exclude provenance only."""
    base = Assertion("${output} > 0", "output must be positive", _origin("first"))
    other_origin = Assertion(
        base.expression,
        base.display,
        _origin("second"),
    )
    changed_expression = Assertion("${output} >= 0", base.display, base.origin)
    changed_display = Assertion(base.expression, "nonnegative output", base.origin)

    same = [_manifest_assertions([base]), _manifest_assertions([other_origin])]
    changed = [
        _manifest_assertions([changed_expression]),
        _manifest_assertions([changed_display]),
    ]

    assert flow_manifest(same[0]) == flow_manifest(same[1])
    assert flow_fingerprint(same[0]) == flow_fingerprint(same[1])
    assert expansion_digest(same[0]) == expansion_digest(same[1])
    for candidate in changed:
        assert flow_manifest(candidate) != flow_manifest(same[0])
        assert flow_fingerprint(candidate) != flow_fingerprint(same[0])
        assert expansion_digest(candidate) != expansion_digest(same[0])


def test_assertion_manifest_preserves_order_and_duplicate_count() -> None:
    """An ordered contract sequence is behavioral identity, including repeated checks."""
    first = Assertion("a", "first")
    second = Assertion("b", "second")

    ordered = _manifest_assertions([first, second])
    reversed_ = _manifest_assertions([second, first])
    duplicated = _manifest_assertions([first, second, second])

    assert flow_fingerprint(ordered) != flow_fingerprint(reversed_)
    assert flow_fingerprint(ordered) != flow_fingerprint(duplicated)


def test_assertion_manifest_rejects_unknown_fields_and_raw_bypass() -> None:
    """Every carrier field needs an explicit policy and strings cannot re-enter Node IR."""
    assertion = Assertion("true", "true")
    object.__setattr__(assertion, "future_semantic_field", True)

    with pytest.raises(ValueError, match="future_semantic_field"):
        flow_manifest(_manifest_assertions([assertion]))
    with pytest.raises(TypeError, match="expected Assertion"):
        flow_manifest(_manifest_assertions(["true"]))


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
