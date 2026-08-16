"""Identity-safe traversal and rewriting of compiled opaque source origins."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from agent_composer.compile import iter_flow_origins, rewrite_flow_origins
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
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Node
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.code import CodeNode
from agent_composer.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
)
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.map import MapNode
from agent_composer.nodes.start import StartNode
from agent_composer.typesys import Type, ValueKind


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
    child_end = EndNode.record(END_ID, output_names=["result"])
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
    end = EndNode.record(END_ID, output_names=["result"])
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
    end = EndNode.record(END_ID, output_names=["result"])
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
