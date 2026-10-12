"""Identity-safe traversal and rewriting of compiled opaque source origins."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from yian.compile import (
    iter_flow_origins,
    rebase_flow_origins,
    rewrite_flow_origins,
)
from yian.compile.manifest import (
    expansion_digest,
    flow_fingerprint,
    flow_manifest,
)
from yian.compile.model import (
    END_ID,
    START_ID,
    CompiledFlow,
    Edge,
    Flow,
    FlowOutput,
)
from yian.diagnostics import (
    DiagnosticFrame,
    OriginRef,
    ResolverTable,
    SourceRange,
)
from yian.memory import DeclSite, MemoryDecl
from yian.nodes.agent import AgentNode
from yian.nodes.base import Node
from yian.nodes.call import CallNode
from yian.nodes.case import Case, CaseNode
from yian.nodes.code import CodeNode
from yian.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    Parameter,
    PartialRecordContract,
    ValueContract,
)
from yian.nodes.end import EndNode
from yian.nodes.loop import LoopNode
from yian.nodes.map import MapNode
from yian.nodes.start import StartNode
from yian.typesys import Type, ValueKind


def _origin(key: str) -> OriginRef:
    """Return one opaque source identity with a readable test key."""
    return OriginRef(key)


def _contract(prefix: str) -> ValueContract:
    """Return a contract carrying both root and nested provenance."""
    return ValueContract(
        Type.scalar(ValueKind.STRING),
        ContractOrigins(
            default=_origin(prefix),
            entries=((('leaf',), _origin(f"{prefix}/leaf")),),
        ),
    )


def _parameter(
    name: str,
    prefix: str,
    *,
    contract: ValueContract | None = None,
    default: Any = "fallback",
) -> Parameter:
    """Return a parameter carrying declaration, default, and contract origins."""
    return Parameter(
        name=name,
        contract=contract or _contract(f"{prefix}/contract"),
        has_default=True,
        default=default,
        origin=_origin(prefix),
        default_origin=_origin(f"{prefix}/default"),
    )


def _call_driver(child: CompiledFlow, signature: Parameter) -> CallNode:
    """Build one CALL fixture over a baked child and cached signature."""
    return CallNode(
        "call",
        flow_id="child",
        child=child,
        child_inputs=[signature],
    )


def _map_driver(child: CompiledFlow, signature: Parameter) -> MapNode:
    """Build one MAP fixture over a baked child and cached signature."""
    return MapNode(
        "map",
        flow_id="child",
        child=child,
        child_inputs=[signature],
    )


def _loop_driver(child: CompiledFlow, signature: Parameter) -> LoopNode:
    """Build one LOOP fixture over a baked child and cached signature."""
    return LoopNode(
        "loop",
        flow_id="child",
        child=child,
        child_inputs=[signature],
        predicate_kind="times",
        times=1,
        max_iters=1,
    )


_CompositeFactory = Callable[[CompiledFlow, Parameter], Node]


def test_agent_variable_origins_rewrite_shared_carriers_once_without_walking_defaults():
    payload_origin = _origin("payload-only")
    contract = ValueContract(
        Type.scalar(ValueKind.OBJECT),
        ContractOrigins(_origin("type"), ((('leaf',), _origin("nested-type")),)),
    )
    variable = _parameter("settings", "variable", contract=contract,
                          default={"opaque": payload_origin})
    first = AgentNode("first", variables=[variable])
    second = AgentNode("second", variables=[variable])
    first.output_contract = contract
    flow = Flow(nodes={"first": first, "second": second}, edges=[],
                wiring={"first": {}, "second": {}})
    assert {origin.key for origin in iter_flow_origins(flow)} == {
        "type", "nested-type", "variable", "variable/default",
    }
    seen = []

    def rewrite(origin):
        seen.append(origin.key)
        return _origin(f"mapped:{origin.key}")

    rewritten = rewrite_flow_origins(flow, rewrite)
    copied = rewritten.nodes["first"].variables[0]
    assert copied is rewritten.nodes["second"].variables[0]
    assert copied.contract is rewritten.nodes["first"].output_contract
    assert sorted(seen) == ["nested-type", "type", "variable", "variable/default"]
    assert copied.default["opaque"] == payload_origin
    assert variable.origin == _origin("variable")


def _rich_flow() -> tuple[CompiledFlow, set[str], OriginRef]:
    """Build all current origin slots with deliberate carrier and child aliases."""
    opaque_payload_origin = _origin("payload/not-provenance")
    shared_payload = {"opaque": opaque_payload_origin}

    body = Node("body", origin=_origin("root/body/node"))
    body.params = [
        _parameter(
            "value",
            "root/body/parameter",
            default=shared_payload,
        )
    ]
    body.pre_asserts = [Assertion("True", "pre", _origin("root/body/pre"))]
    body.post_asserts = [Assertion("True", "post", _origin("root/body/post"))]
    body.output_contract = _contract("root/body/output")
    body.env = {"opaque": opaque_payload_origin}

    signature_contract = _contract("child/signature/contract")
    signature = _parameter(
        "item",
        "child/signature/parameter",
        contract=signature_contract,
    )
    child_start = StartNode(START_ID, params=[signature])
    child_start.origin = _origin("child/start/node")
    child_body = Node(
        "child_body",
        output_contract=signature_contract,
        origin=_origin("child/body/node"),
    )
    child_end = EndNode.value(END_ID, output_name="result")
    child_end.origin = _origin("child/end/node")
    child_output = Binding(
        "${child_body.output}",
        _origin("child/output"),
    )
    child = CompiledFlow.from_parts(
        nodes={
            START_ID: child_start,
            "child_body": child_body,
            END_ID: child_end,
        },
        edges=[],
        outputs=[FlowOutput("result", child_output)],
        wiring={
            START_ID: {},
            "child_body": {
                "item": Binding("${__start__.output.item}", _origin("child/wiring"))
            },
            END_ID: {"result": child_output},
        },
    )

    call = _call_driver(child, signature)
    call.origin = _origin("root/call/node")
    mapped = _map_driver(child, signature)
    mapped.origin = _origin("root/map/node")
    loop = _loop_driver(child, signature)
    loop.origin = _origin("root/loop/node")

    start = StartNode(START_ID, params=[])
    end = EndNode.value(END_ID, output_name="result")
    root_output = Binding("${body.output}", _origin("root/output"))
    flow = CompiledFlow.from_parts(
        nodes={
            START_ID: start,
            "body": body,
            "call": call,
            "map": mapped,
            "loop": loop,
            END_ID: end,
        },
        edges=[],
        outputs=[FlowOutput("result", root_output)],
        wiring={
            START_ID: {},
            "body": {"value": Binding(shared_payload, _origin("root/wiring"))},
            "call": {},
            "map": {},
            "loop": {},
            END_ID: {"result": root_output},
        },
    )

    expected = {
        "root/wiring",
        "root/output",
        "root/body/node",
        "root/body/parameter/contract",
        "root/body/parameter/contract/leaf",
        "root/body/parameter",
        "root/body/parameter/default",
        "root/body/pre",
        "root/body/post",
        "root/body/output",
        "root/body/output/leaf",
        "root/call/node",
        "root/map/node",
        "root/loop/node",
        "child/signature/contract",
        "child/signature/contract/leaf",
        "child/signature/parameter",
        "child/signature/parameter/default",
        "child/start/node",
        "child/body/node",
        "child/end/node",
        "child/wiring",
        "child/output",
    }
    return flow, expected, opaque_payload_origin


def test_iter_flow_origins_reaches_every_slot_once_and_is_repeatable() -> None:
    """Traversal is structural, deterministic, opaque to payloads, and alias-safe."""
    flow, expected, payload_origin = _rich_flow()

    first = list(iter_flow_origins(flow))
    second = list(iter_flow_origins(flow))

    assert first == second
    assert all(left is right for left, right in zip(first, second, strict=True))
    assert {origin.key for origin in first} == expected
    assert len(first) == len(expected)
    assert payload_origin not in first


def test_parameter_origins_follow_the_carriers_canonical_order() -> None:
    """A parameter visits contract root/entries before its own declaration sites."""
    parameter = Parameter(
        "value",
        contract=ValueContract(
            Type.scalar(ValueKind.STRING),
            ContractOrigins(
                default=_origin("contract/root"),
                entries=(
                    (("a",), _origin("contract/a")),
                    (("z",), _origin("contract/z")),
                ),
            ),
        ),
        has_default=True,
        default="fallback",
        origin=_origin("parameter"),
        default_origin=_origin("parameter/default"),
    )
    node = Node("only")
    node.params = [parameter]
    flow = Flow(
        nodes={"only": node},
        edges=[],
        wiring={"only": {}},
        start_id="only",
        end_id="only",
    )

    assert [origin.key for origin in iter_flow_origins(flow)] == [
        "contract/root",
        "contract/a",
        "contract/z",
        "parameter",
        "parameter/default",
    ]


def test_partial_output_contract_origins_traverse_and_rewrite() -> None:
    """Only proven fields contribute origins through the common node seam."""
    node = Node(
        "only",
        output_contract=PartialRecordContract(
            fields=(
                (
                    "known",
                    ValueContract(
                        Type.scalar(ValueKind.INTEGER),
                        ContractOrigins(default=_origin("output/known")),
                    ),
                ),
                ("opaque", None),
            )
        ),
    )
    flow = Flow(
        nodes={"only": node},
        edges=[],
        wiring={"only": {}},
        start_id="only",
        end_id="only",
    )

    assert [origin.key for origin in iter_flow_origins(flow)] == ["output/known"]
    rewritten = rewrite_flow_origins(
        flow,
        lambda origin: _origin(f"mapped/{origin.key}"),
    )
    contract = rewritten.nodes["only"].output_contract
    assert type(contract) is PartialRecordContract
    assert contract.fields[0][1].origins.default == _origin("mapped/output/known")
    assert node.output_contract.fields[0][1].origins.default == _origin("output/known")


def test_case_predicate_origins_traverse_and_rewrite_by_arm_identity() -> None:
    """Ordered CASE sites participate in graph rewrites without double-mapping aliases."""
    shared = Case("first", "false", origin=_origin("case/shared"))
    node = CaseNode(
        "case",
        [shared, shared, Case("second", "true", origin=_origin("case/second"))],
    )
    flow = Flow(
        nodes={"case": node},
        edges=[],
        wiring={"case": {}},
        start_id="case",
        end_id="case",
    )

    assert [origin.key for origin in iter_flow_origins(flow)] == [
        "case/shared",
        "case/second",
    ]

    calls: list[str] = []
    rewritten = rewrite_flow_origins(
        flow,
        lambda origin: (calls.append(origin.key) or _origin(f"mapped/{origin.key}")),
    )

    rewritten_cases = rewritten.nodes["case"].cases
    assert rewritten_cases[0] is rewritten_cases[1]
    assert rewritten_cases[0].origin == _origin("mapped/case/shared")
    assert rewritten_cases[2].origin == _origin("mapped/case/second")
    assert calls == ["case/shared", "case/second"]
    assert shared.origin == _origin("case/shared")


def _shared_decl(origin: OriginRef | None) -> MemoryDecl:
    return MemoryDecl(
        "reflective", "reflective", {"max_items": 4}, DeclSite((), ("root",), "agent"), origin,
    )


def _aliased_flow(shared: MemoryDecl) -> CompiledFlow:
    return CompiledFlow.from_parts(
        nodes={
            "left": AgentNode("left", memories=(shared,)),
            "right": AgentNode("right", memories=(shared,)),
        },
        edges=[],
        wiring={"left": {}, "right": {}},
    )


def test_shared_memory_decl_origin_traverses_and_rewrites_once() -> None:
    """Two AGENTs may deliberately alias one immutable declaration carrier."""
    shared = _shared_decl(_origin("policy/shared"))
    flow = _aliased_flow(shared)

    assert list(iter_flow_origins(flow)) == [_origin("policy/shared")]
    calls: list[OriginRef] = []

    class DerivedOrigin(OriginRef):
        """Mapper output whose key must be canonicalized to the base carrier."""

    rewritten = rewrite_flow_origins(
        flow,
        lambda origin: (calls.append(origin) or DerivedOrigin(f"mapped/{origin.key}")),
    )

    assert calls == [_origin("policy/shared")]
    left, right = rewritten.nodes["left"].memories[0], rewritten.nodes["right"].memories[0]
    assert left is right
    assert type(left.origin) is OriginRef
    assert left.origin == _origin("mapped/policy/shared")
    assert flow.nodes["left"].memories[0] is shared
    assert shared.origin == _origin("policy/shared")


def test_source_free_memory_decl_adds_no_origin() -> None:
    assert list(iter_flow_origins(_aliased_flow(_shared_decl(None)))) == []


def test_memory_decl_origin_rebase_routes_the_single_shared_site() -> None:
    shared = _shared_decl(_origin("policy/shared"))
    flow = _aliased_flow(shared)
    frame = DiagnosticFrame(
        label="memory.yaml",
        text="long_term_memory: reflective\n",
        range=SourceRange(1, 19, 1, 29),
    )

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return frame if origin == shared.origin else None

    mapped: list[OriginRef] = []
    rebased, table = rebase_flow_origins(
        flow,
        lambda origin: (mapped.append(origin) or _origin(f"alias/{origin.key}")),
        resolver=Resolver(),
    )

    assert mapped == [_origin("policy/shared")]
    assert rebased.nodes["left"].memories[0] is rebased.nodes["right"].memories[0]
    assert table.resolve(_origin("alias/policy/shared")) == frame


@pytest.mark.parametrize(
    "factory",
    [_call_driver, _map_driver, _loop_driver],
    ids=["call", "map", "loop"],
)
def test_each_composite_exposes_its_signature_and_baked_child(
    factory: _CompositeFactory,
) -> None:
    """Every higher-order node delegates both local caches and recursive children."""
    signature = Parameter("item", origin=_origin("signature"))
    child_start = StartNode(START_ID, params=[signature])
    child_start.origin = _origin("child")
    child = CompiledFlow.from_parts(
        nodes={START_ID: child_start},
        edges=[],
        wiring={START_ID: {}},
    )
    driver = factory(child, signature)
    flow = Flow(
        nodes={driver.id: driver},
        edges=[],
        wiring={driver.id: {}},
        start_id=driver.id,
        end_id=driver.id,
    )

    assert {origin.key for origin in iter_flow_origins(flow)} == {
        "signature",
        "child",
    }

    rewritten = rewrite_flow_origins(
        flow,
        lambda origin: _origin(f"mapped/{origin.key}"),
    )
    rewritten_driver = rewritten.nodes[driver.id]
    rewritten_child = rewritten_driver.child
    rewritten_signature = rewritten_child.nodes[START_ID].params[0]
    assert rewritten_driver.child_inputs[0] is rewritten_signature
    assert {origin.key for origin in iter_flow_origins(rewritten)} == {
        "mapped/signature",
        "mapped/child",
    }


def test_rewrite_flow_origins_is_a_pure_alias_preserving_graph_copy() -> None:
    """One rewrite maps shared carriers once while preserving every copied alias."""
    flow, expected, payload_origin = _rich_flow()
    before = list(iter_flow_origins(flow))
    calls: list[OriginRef] = []

    def mapper(origin: OriginRef) -> OriginRef:
        calls.append(origin)
        return _origin(f"mapped/{origin.key}")

    rewritten = rewrite_flow_origins(flow, mapper)

    assert rewritten is not flow
    assert list(iter_flow_origins(flow)) == before
    assert {origin.key for origin in calls} == expected
    assert len(calls) == len(expected)
    assert {origin.key for origin in iter_flow_origins(rewritten)} == {
        f"mapped/{key}" for key in expected
    }

    assert rewritten.outputs[0].binding is rewritten.wiring[END_ID]["result"]
    assert flow.outputs[0].binding is flow.wiring[END_ID]["result"]

    rewritten_child = rewritten.nodes["call"].child
    assert rewritten_child is not flow.nodes["call"].child
    assert rewritten.nodes["map"].child is rewritten_child
    assert rewritten.nodes["loop"].child is rewritten_child
    rewritten_signature = rewritten_child.nodes[START_ID].params[0]
    assert rewritten.nodes["call"].child_inputs[0] is rewritten_signature
    assert rewritten.nodes["map"].child_inputs[0] is rewritten_signature
    assert rewritten.nodes["loop"].child_inputs[0] is rewritten_signature
    assert rewritten_signature.contract is rewritten_child.nodes["child_body"].output_contract
    assert (
        rewritten_child.outputs[0].binding
        is rewritten_child.wiring[END_ID]["result"]
    )

    rewritten_payload = rewritten.wiring["body"]["value"].source
    assert rewritten_payload is rewritten.nodes["body"].params[0].default
    assert rewritten_payload["opaque"] == payload_origin
    assert rewritten_payload["opaque"].key == "payload/not-provenance"
    assert rewritten.nodes["body"].env["opaque"].key == "payload/not-provenance"


def test_rewrite_preserves_payload_back_references_to_frozen_graph_objects() -> None:
    """Opaque payload cycles keep pointing at their canonical copied objects."""
    binding_payload: list[Any] = []
    binding = Binding(binding_payload, _origin("binding"))
    binding_payload.append(binding)

    default_payload: list[Any] = []
    parameter = Parameter(
        "value",
        has_default=True,
        default=default_payload,
        origin=_origin("parameter"),
    )
    default_payload.append(parameter)

    output_payload: list[Any] = []
    output = FlowOutput(
        "result",
        Binding(output_payload, _origin("output")),
    )
    output_payload.append(output)

    node = Node("node")
    node.params = [parameter]
    flow = CompiledFlow.from_parts(
        nodes={"node": node},
        edges=[],
        outputs=[output],
        wiring={"node": {"value": binding}},
    )

    rewritten = rewrite_flow_origins(
        flow,
        lambda origin: _origin(f"mapped/{origin.key}"),
    )

    rewritten_binding = rewritten.wiring["node"]["value"]
    rewritten_parameter = rewritten.nodes["node"].params[0]
    rewritten_output = rewritten.outputs[0]
    assert rewritten_binding.source[0] is rewritten_binding
    assert rewritten_parameter.default[0] is rewritten_parameter
    assert rewritten_output.binding.source[0] is rewritten_output
    assert rewritten_binding.origin == _origin("mapped/binding")
    assert rewritten_parameter.origin == _origin("mapped/parameter")
    assert rewritten_output.binding.origin == _origin("mapped/output")

    assert binding.source[0] is binding
    assert parameter.default[0] is parameter
    assert output.binding.source[0] is output


def test_rewrite_rebuilds_payload_hash_containers_after_origin_changes() -> None:
    """Carrier keys and set members remain findable under their mapped hashes."""
    assertion = Assertion("True", "true", _origin("assertion"))
    payload = {
        "mapping": {assertion: "sentinel"},
        "set": {assertion},
    }
    node = Node("node")
    node.pre_asserts = [assertion]
    flow = Flow(
        nodes={"node": node},
        edges=[],
        wiring={"node": {"value": Binding(payload)}},
        start_id="node",
        end_id="node",
    )

    rewritten = rewrite_flow_origins(
        flow,
        lambda origin: _origin(f"mapped/{origin.key}"),
    )

    rewritten_assertion = rewritten.nodes["node"].pre_asserts[0]
    rewritten_payload = rewritten.wiring["node"]["value"].source
    assert rewritten_payload["mapping"][rewritten_assertion] == "sentinel"
    assert rewritten_assertion in rewritten_payload["set"]
    assert next(iter(rewritten_payload["mapping"])) is rewritten_assertion
    assert next(iter(rewritten_payload["set"])) is rewritten_assertion
    assert rewritten_assertion.origin == _origin("mapped/assertion")

    assert payload["mapping"][assertion] == "sentinel"
    assert assertion in payload["set"]


def test_traversal_and_rewrite_terminate_on_a_deliberate_child_cycle() -> None:
    """Identity guards visit a recursive object graph once and retain its cycle."""
    cycle = Flow(
        nodes={},
        edges=[],
        wiring={},
        start_id="self",
        end_id="self",
    )
    driver = CallNode("self", flow_id="self", child=cycle)
    driver.origin = _origin("cycle/node")
    cycle.nodes["self"] = driver
    cycle.wiring["self"] = {}

    assert list(iter_flow_origins(cycle)) == [_origin("cycle/node")]

    calls: list[OriginRef] = []

    def mapper(origin: OriginRef) -> OriginRef:
        calls.append(origin)
        return _origin(f"mapped/{origin.key}")

    rewritten = rewrite_flow_origins(cycle, mapper)

    assert calls == [_origin("cycle/node")]
    assert rewritten is not cycle
    assert rewritten.nodes["self"].child is rewritten
    assert rewritten.nodes["self"].origin == _origin("mapped/cycle/node")
    assert cycle.nodes["self"].child is cycle
    assert cycle.nodes["self"].origin == _origin("cycle/node")


def test_source_free_python_graph_ignores_origin_refs_in_semantic_payloads() -> None:
    """Only declared provenance slots are visited; arbitrary values remain opaque."""
    payload_origin = _origin("ordinary/value")
    payload = {"origin-shaped": payload_origin}
    node = Node("python")
    node.params = [
        Parameter(
            "value",
            has_default=True,
            default=payload,
        )
    ]
    node.env = {"ordinary": payload_origin}
    flow = CompiledFlow.from_parts(
        nodes={"python": node},
        edges=[],
        outputs=[FlowOutput("result", payload_origin)],
        wiring={"python": {"value": payload}},
    )

    assert list(iter_flow_origins(flow)) == []

    def unexpected(_origin_ref: OriginRef) -> OriginRef:
        pytest.fail("semantic payload was mistaken for provenance")

    rewritten = rewrite_flow_origins(flow, unexpected)

    assert rewritten is not flow
    assert rewritten.nodes["python"] is not node
    assert rewritten.wiring["python"]["value"].source["origin-shaped"] == payload_origin
    assert rewritten.outputs[0].binding.source == payload_origin
    assert rewritten.nodes["python"].params[0].default["origin-shaped"] == payload_origin
    assert rewritten.nodes["python"].env["ordinary"] == payload_origin


def test_generated_origin_context_is_not_local_authored_provenance() -> None:
    """Authored-origin traversal neither visits nor rewrites generated ancestry."""
    node = Node("child", origin=_origin("child/node"))
    node.origin_context = (_origin("outer/call"),)
    flow = Flow(
        nodes={"child": node},
        edges=[],
        wiring={"child": {}},
        start_id="child",
        end_id="child",
    )
    visited: list[OriginRef] = []

    def mapper(origin: OriginRef) -> OriginRef:
        visited.append(origin)
        return _origin(f"mapped/{origin.key}")

    assert list(iter_flow_origins(flow)) == [_origin("child/node")]

    rewritten = rewrite_flow_origins(flow, mapper)

    assert visited == [_origin("child/node")]
    assert rewritten.nodes["child"].origin == _origin("mapped/child/node")
    assert rewritten.nodes["child"].origin_context == (_origin("outer/call"),)
    assert node.origin_context == (_origin("outer/call"),)


def test_invalid_mutable_node_origin_fails_before_iteration_or_mapping() -> None:
    """A directly corrupted Python node cannot leak a non-origin through the API."""
    node = Node("invalid")
    node.origin = "not-an-origin"  # type: ignore[assignment]
    flow = Flow(
        nodes={"invalid": node},
        edges=[],
        wiring={"invalid": {}},
        start_id="invalid",
        end_id="invalid",
    )

    with pytest.raises(TypeError, match="node origin must be an OriginRef"):
        list(iter_flow_origins(flow))
    with pytest.raises(TypeError, match="node origin must be an OriginRef"):
        rewrite_flow_origins(flow, lambda origin: origin)


def test_invalid_child_is_rejected_before_deepcopying_payloads() -> None:
    """Structural child errors win over arbitrary deepcopy behavior."""

    class DeepcopyBomb:
        """Invalid child whose copy path must never run."""

        def __deepcopy__(self, _memo: dict[int, object]) -> Any:
            raise RuntimeError("deepcopy bomb")

    driver = CallNode("call", flow_id="invalid", child=DeepcopyBomb())
    flow = Flow(
        nodes={"call": driver},
        edges=[],
        wiring={"call": {}},
        start_id="call",
        end_id="call",
    )

    with pytest.raises(TypeError, match="origin child must be a Flow"):
        list(iter_flow_origins(flow))
    with pytest.raises(TypeError, match="origin child must be a Flow"):
        rewrite_flow_origins(flow, lambda origin: origin)


@pytest.mark.parametrize("invalid", [None, "not-an-origin"])
def test_rewrite_flow_origins_rejects_invalid_mapper_results(invalid: Any) -> None:
    """A present origin cannot be erased or replaced with another value type."""
    binding = Binding(1, _origin("binding"))
    flow = Flow(
        nodes={"only": Node("only")},
        edges=[],
        wiring={"only": {"value": binding}},
        start_id="only",
        end_id="only",
    )

    with pytest.raises(TypeError, match="origin mapper must return an OriginRef"):
        rewrite_flow_origins(flow, lambda _origin_ref: invalid)

    assert flow.wiring["only"]["value"] is binding
    assert binding.origin == _origin("binding")


def test_rewrite_flow_origins_propagates_mapper_exceptions() -> None:
    """Mapper failures are loud and cannot partially mutate the authored graph."""

    class MappingError(RuntimeError):
        """Sentinel mapper failure."""

    binding = Binding(1, _origin("binding"))
    flow = Flow(
        nodes={"only": Node("only")},
        edges=[],
        wiring={"only": {"value": binding}},
        start_id="only",
        end_id="only",
    )

    def fail(_origin_ref: OriginRef) -> OriginRef:
        raise MappingError("cannot rewrite")

    with pytest.raises(MappingError, match="cannot rewrite"):
        rewrite_flow_origins(flow, fail)

    assert flow.wiring["only"]["value"] is binding
    assert binding.origin == _origin("binding")


def _manifestable_flow() -> CompiledFlow:
    """Return a canonical graph whose diagnostic-only origins cover every carrier."""
    input_contract = _contract("manifest/input/contract")
    output_contract = _contract("manifest/output/contract")
    start = StartNode(
        START_ID,
        params=[
            Parameter(
                "value",
                contract=input_contract,
                required=True,
                origin=_origin("manifest/input"),
            )
        ],
    )
    start.origin = _origin("manifest/start")
    body = CodeNode("body", code='return inputs["value"]')
    body.origin = _origin("manifest/body")
    body.params = [
        Parameter(
            "value",
            contract=input_contract,
            required=True,
            origin=_origin("manifest/body/input"),
        )
    ]
    body.output_contract = output_contract
    body.pre_asserts = [
        Assertion("${value} >= 0", "value >= 0", _origin("manifest/pre"))
    ]
    body.post_asserts = [
        Assertion("${output} >= 0", "output >= 0", _origin("manifest/post"))
    ]
    end = EndNode.value(END_ID, output_name="result")
    end.origin = _origin("manifest/end")
    output = Binding("${body.output}", _origin("manifest/output"))
    return CompiledFlow.from_parts(
        nodes={START_ID: start, "body": body, END_ID: end},
        edges=[
            Edge("start->body", START_ID, "body", input_group="value"),
            Edge("body->end", "body", END_ID, input_group="result"),
        ],
        outputs=[FlowOutput("result", output)],
        wiring={
            START_ID: {"value": Binding(1, _origin("manifest/start/value"))},
            "body": {
                "value": Binding(
                    "${__start__.output.value}",
                    _origin("manifest/body/value"),
                )
            },
            END_ID: {"result": output},
        },
    )


def test_rewriting_origins_preserves_manifest_and_stable_identity() -> None:
    """Diagnostic rebasing cannot change authored or dynamic program identity."""
    flow = _manifestable_flow()
    before_manifest = flow_manifest(flow)
    before_fingerprint = flow_fingerprint(flow)
    before_expansion = expansion_digest(flow)

    rewritten = rewrite_flow_origins(
        flow,
        lambda origin: _origin(f"mapped/{origin.key}"),
    )

    assert flow_manifest(rewritten) == before_manifest
    assert flow_fingerprint(rewritten) == before_fingerprint
    assert expansion_digest(rewritten) == before_expansion
    assert flow_manifest(flow) == before_manifest
    assert flow_fingerprint(flow) == before_fingerprint
    assert expansion_digest(flow) == before_expansion
    assert all(
        origin.key.startswith("mapped/") for origin in iter_flow_origins(rewritten)
    )


def test_rebase_maps_equal_origins_once_and_routes_the_exact_old_identity() -> None:
    """Equal keys share one new identity and route across distinct carrier classes."""
    old = _origin("opaque/~and/slash")
    binding = Binding(1, old)

    class DerivedOrigin(OriginRef):
        """Distinct Python class that retains the same opaque key contract."""

    node = Node("only", origin=DerivedOrigin("opaque/~and/slash"))
    flow = Flow(
        nodes={"only": node},
        edges=[],
        wiring={"only": {"value": binding}},
        start_id="only",
        end_id="only",
    )
    frame = DiagnosticFrame(
        label="memory",
        text="value\n",
        range=SourceRange(1, 1, 1, 6),
    )
    resolved: list[OriginRef] = []

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            resolved.append(origin)
            return frame if origin == old else None

    mapped: list[OriginRef] = []

    def mapper(origin: OriginRef) -> OriginRef:
        mapped.append(origin)
        return _origin("scope/~opaque/key")

    rebased, table = rebase_flow_origins(flow, mapper, resolver=Resolver())

    assert mapped == [old]
    assert list(iter_flow_origins(flow)) == [old, node.origin]
    assert list(iter_flow_origins(rebased)) == [
        _origin("scope/~opaque/key"),
        _origin("scope/~opaque/key"),
    ]
    assert table.origins == frozenset({_origin("scope/~opaque/key")})
    assert table.resolve(_origin("scope/~opaque/key")) == frame
    assert resolved == [old]


def test_rebase_is_chainable_and_preserves_graph_identity_and_manifests() -> None:
    """A later namespace delegates through the earlier table without graph mutation."""
    flow = _manifestable_flow()
    original_origins = list(iter_flow_origins(flow))
    original_manifest = flow_manifest(flow)
    original_fingerprint = flow_fingerprint(flow)
    original_digest = expansion_digest(flow)

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return DiagnosticFrame(
                label=origin.key,
                text="value\n",
                range=SourceRange(1, 1, 1, 6),
            )

    first, first_table = rebase_flow_origins(
        flow,
        lambda origin: _origin(f"first/{origin.key}"),
        resolver=Resolver(),
    )
    first_origins = list(iter_flow_origins(first))
    second, second_table = rebase_flow_origins(
        first,
        lambda origin: _origin(f"second/{origin.key}"),
        resolver=first_table,
    )

    assert list(iter_flow_origins(flow)) == original_origins
    assert list(iter_flow_origins(first)) == first_origins
    assert all(origin.key.startswith("first/") for origin in first_origins)
    assert all(
        origin.key.startswith("second/first/")
        for origin in iter_flow_origins(second)
    )
    sample = original_origins[0]
    final = _origin(f"second/first/{sample.key}")
    assert second_table.resolve(final).label == sample.key  # type: ignore[union-attr]
    assert flow_manifest(second) == original_manifest
    assert flow_fingerprint(second) == original_fingerprint
    assert expansion_digest(second) == original_digest


def test_rebase_rejects_missing_resolution_and_distinct_origin_collisions() -> None:
    """Present origins require a delegate and distinct old keys cannot collapse."""
    flow = Flow(
        nodes={"only": Node("only", origin=_origin("node"))},
        edges=[],
        wiring={"only": {"value": Binding(1, _origin("binding"))}},
        start_id="only",
        end_id="only",
    )

    with pytest.raises(ValueError, match="originful flow requires an origin resolver"):
        rebase_flow_origins(flow, lambda origin: origin, resolver=None)

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return None

    with pytest.raises(ValueError, match="mapped origin collision 'same'"):
        rebase_flow_origins(
            flow,
            lambda _origin_ref: _origin("same"),
            resolver=Resolver(),
        )

    assert [origin.key for origin in iter_flow_origins(flow)] == ["binding", "node"]


@pytest.mark.parametrize("invalid", [None, "not-an-origin"])
def test_rebase_rejects_invalid_mapper_results(invalid: Any) -> None:
    """A namespace mapper must return an `OriginRef` for every distinct old ref."""
    flow = Flow(
        nodes={"only": Node("only", origin=_origin("node"))},
        edges=[],
        wiring={"only": {}},
        start_id="only",
        end_id="only",
    )

    with pytest.raises(TypeError, match="origin mapper must return an OriginRef"):
        rebase_flow_origins(
            flow,
            lambda _origin_ref: invalid,
            resolver=ResolverTable(),
        )


def test_rebase_revalidates_origin_subclass_keys() -> None:
    """Subclass overrides cannot introduce an empty old or mapped opaque key."""

    class InvalidOrigin(OriginRef):
        def __post_init__(self) -> None:
            pass

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return None

    invalid = InvalidOrigin("")
    flow = Flow(
        nodes={"only": Node("only", origin=invalid)},
        edges=[],
        wiring={"only": {}},
        start_id="only",
        end_id="only",
    )

    with pytest.raises(ValueError, match="origin key must be a non-empty string"):
        rebase_flow_origins(flow, lambda origin: origin, resolver=Resolver())

    flow.nodes["only"].origin = _origin("valid")
    with pytest.raises(ValueError, match="origin key must be a non-empty string"):
        rebase_flow_origins(flow, lambda _origin: invalid, resolver=Resolver())


def test_rebase_source_free_graph_returns_an_independent_copy_and_empty_table() -> None:
    """Python-built graphs need no front-end resolver when no origins are present."""
    node = Node("python")
    flow = Flow(
        nodes={"python": node},
        edges=[],
        wiring={"python": {}},
        start_id="python",
        end_id="python",
    )

    def unexpected(_origin_ref: OriginRef) -> OriginRef:
        pytest.fail("source-free graph called the namespace mapper")

    rebased, table = rebase_flow_origins(flow, unexpected, resolver=None)

    assert rebased is not flow
    assert rebased.nodes["python"] is not node
    assert list(iter_flow_origins(rebased)) == []
    assert len(table) == 0

    with pytest.raises(
        ValueError,
        match="source-free flow cannot have an origin resolver",
    ):
        rebase_flow_origins(flow, unexpected, resolver=ResolverTable())


def test_rebase_retains_a_deliberate_child_cycle() -> None:
    """Atomic rebasing inherits traversal's identity guards and cycle-safe copying."""
    cycle = Flow(
        nodes={},
        edges=[],
        wiring={},
        start_id="self",
        end_id="self",
    )
    driver = CallNode("self", flow_id="self", child=cycle)
    driver.origin = _origin("cycle/node")
    cycle.nodes["self"] = driver
    cycle.wiring["self"] = {}

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return None

    rebased, table = rebase_flow_origins(
        cycle,
        lambda origin: _origin(f"mapped/{origin.key}"),
        resolver=Resolver(),
    )

    assert rebased.nodes["self"].child is rebased
    assert rebased.nodes["self"].origin == _origin("mapped/cycle/node")
    assert cycle.nodes["self"].child is cycle
    assert table.origins == frozenset({_origin("mapped/cycle/node")})
