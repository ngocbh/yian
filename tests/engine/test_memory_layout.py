"""Pure static memory-layout derivation and atomic scope application."""

from __future__ import annotations

import copy

import pytest

from agent_composer.compile.manifest import flow_fingerprint, flow_manifest
from agent_composer.compile.memory_layout import (
    apply_memory_layout,
    resolve_memory_layout,
)
from agent_composer.compile.model import END_ID, START_ID, CompiledFlow
from agent_composer.compose import load_flow
from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    MemoryBoundaryKind,
    MemoryFrameRef,
    MemoryOwner,
    MemoryPolicySpec,
    MemoryScope,
    MemorySetupError,
)
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.code import CodeNode
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode


_NESTED = """id: root
name: root
memory_policy: outer
defs:
  child:
    memory_policy:
      name: inner
      config: {limit: 3}
    nodes:
      answer:
        kind: agent
        memory_policy: local
        prompt: answer
        output: str
      normalize:
        kind: code
        code: tests.engine._compose_codefns:echo
        input: {topic: child}
        output: str
    output: ${answer.output}
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""


def _target(layout, *locator: str):
    """Return one resolved target by its structural locator."""
    return next(target for target in layout.targets if target.locator == locator)


def test_disabled_layout_needs_no_flow_id_and_leaves_every_scope_none() -> None:
    flow = CompiledFlow(
        {"plain": CodeNode("plain", code="return 1")},
        [],
        wiring={"plain": {}},
    )
    flow.nodes["plain"].memory_scope = MemoryScope("old", ("old", "plain"))

    layout = resolve_memory_layout(flow)

    assert layout.enabled is False
    assert layout.root_flow_id is None
    assert layout.frames == ()
    assert [(target.locator, target.scope) for target in layout.targets] == [
        (("plain",), None)
    ]
    # Resolution is pure and ignores an existing derived scope.
    assert flow.nodes["plain"].memory_scope == MemoryScope(
        "old", ("old", "plain")
    )
    with pytest.raises(MemorySetupError, match="different memory scope"):
        apply_memory_layout(flow, layout)


def test_reachable_policy_requires_root_id_at_the_authored_policy_site() -> None:
    origin = OriginRef("nodes/agent/memory_policy")
    agent = AgentNode(
        "agent",
        memory_policy=MemoryPolicySpec("reflective", {}, origin),
    )
    flow = CompiledFlow({"agent": agent}, [], wiring={"agent": {}})

    with pytest.raises(MemorySetupError, match="requires.*root flow_id") as caught:
        resolve_memory_layout(flow)

    assert caught.value.origin is origin
    assert agent.memory_scope is None


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-both", "missing direct START and END"),
        ("missing-start", "missing direct START"),
        ("missing-end", "missing direct END"),
        ("start-subclass", "start_id must be a non-empty exact string"),
        ("end-empty", "end_id must be a non-empty exact string"),
        ("same", "distinct START and END"),
    ],
)
def test_memory_owning_python_flow_requires_exact_direct_boundaries(
    case: str,
    message: str,
) -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical boundary identifier."""

    origin = OriginRef("flow/memory_policy")
    nodes = {
        START_ID: StartNode(START_ID, params=[]),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    if case == "missing-both":
        nodes = {"body": CodeNode("body", code="return 1")}
    elif case == "missing-start":
        nodes.pop(START_ID)
    elif case == "missing-end":
        nodes.pop(END_ID)
    flow = CompiledFlow(
        nodes,
        [],
        wiring={node_id: {} for node_id in nodes},
        flow_id="root",
        memory_policy=MemoryPolicySpec("reflective", {}, origin),
    )
    if case == "start-subclass":
        flow.start_id = StringSubclass(START_ID)
    elif case == "end-empty":
        flow.end_id = ""
    elif case == "same":
        flow.end_id = START_ID

    with pytest.raises(MemorySetupError, match=message) as caught:
        resolve_memory_layout(flow)

    assert caught.value.origin is origin
    assert all(node.memory_scope is None for node in nodes.values())


def test_layout_orders_outer_flow_inner_flow_and_local_agent_frames() -> None:
    flow = load_flow(_NESTED).compiled

    layout = resolve_memory_layout(flow)

    assert [
        (frame.spec.ref.static_path, frame.spec.ref.owner, frame.spec.policy_name)
        for frame in layout.frames
    ] == [
        (("root",), MemoryOwner.FLOW, "outer"),
        (("root", "invoke"), MemoryOwner.FLOW, "inner"),
        (("root", "invoke", "answer"), MemoryOwner.AGENT, "local"),
    ]
    assert _target(layout, "invoke").scope.frame_stack == (
        layout.frames[0].spec.ref,
    )
    assert _target(layout, "invoke", "normalize").scope.frame_stack == (
        layout.frames[0].spec.ref,
        layout.frames[1].spec.ref,
    )
    assert _target(layout, "invoke", "answer").scope.frame_stack == tuple(
        frame.spec.ref for frame in layout.frames
    )
    assert _target(layout, "invoke").scope.static_path == ("root", "invoke")
    assert _target(layout, "invoke", "answer").scope.static_path == (
        "root",
        "invoke",
        "answer",
    )


def test_only_memory_owning_flow_boundaries_are_marked() -> None:
    with_owned_child = resolve_memory_layout(load_flow(_NESTED).compiled)
    child_frame = with_owned_child.frames[1].spec.ref
    assert _target(with_owned_child, "invoke", "__start__").scope.boundary.kind is (
        MemoryBoundaryKind.START
    )
    assert (
        _target(with_owned_child, "invoke", "__start__").scope.boundary.frame_ref
        == child_frame
    )
    assert _target(with_owned_child, "invoke", "__end__").scope.boundary.kind is (
        MemoryBoundaryKind.END
    )

    without_child_policy = load_flow(
        _NESTED.replace(
            "    memory_policy:\n      name: inner\n      config: {limit: 3}\n",
            "",
        )
    ).compiled
    inherited = resolve_memory_layout(without_child_policy)
    assert _target(inherited, "invoke", "__start__").scope.boundary is None
    assert _target(inherited, "invoke", "__end__").scope.boundary is None
    assert _target(inherited, "invoke", "answer").scope.frame_stack == (
        inherited.frames[0].spec.ref,
        inherited.frames[1].spec.ref,
    )
    assert [frame.spec.ref.owner for frame in inherited.frames] == [
        MemoryOwner.FLOW,
        MemoryOwner.AGENT,
    ]


def test_composite_drivers_never_create_memory_frames() -> None:
    layout = resolve_memory_layout(load_flow(_NESTED).compiled)

    assert [
        (frame.spec.ref.static_path, frame.spec.ref.owner)
        for frame in layout.frames
    ] == [
        (("root",), MemoryOwner.FLOW),
        (("root", "invoke"), MemoryOwner.FLOW),
        (("root", "invoke", "answer"), MemoryOwner.AGENT),
    ]
    # The CALL at this same static path sees only its enclosing flow frame; the
    # child FLOW frame begins inside the called child.
    assert _target(layout, "invoke").scope.frame_stack == (
        layout.frames[0].spec.ref,
    )


def test_repeated_definition_and_inline_children_get_callsite_paths() -> None:
    repeated_def = load_flow(
        """id: root
name: root
defs:
  child:
    memory_policy: reflective
    nodes:
      answer:
        kind: code
        code: tests.engine._compose_codefns:echo
        input: {topic: child}
        output: str
    output: ${answer.output}
nodes:
  left: {kind: call, call: child}
  right: {kind: call, call: child}
output: ${right.output}
"""
    ).compiled
    definition_layout = resolve_memory_layout(repeated_def)
    assert [frame.spec.ref.static_path for frame in definition_layout.frames] == [
        ("root", "left"),
        ("root", "right"),
    ]

    inline = load_flow(
        """id: root
name: root
nodes:
  left:
    kind: call
    nodes:
      answer:
        kind: agent
        memory_policy: reflective
        prompt: left
        output: str
    output: ${answer.output}
  right:
    kind: call
    nodes:
      answer:
        kind: agent
        memory_policy: reflective
        prompt: right
        output: str
    output: ${answer.output}
output: ${right.output}
"""
    ).compiled
    inline_layout = resolve_memory_layout(inline)
    assert [frame.spec.ref.static_path for frame in inline_layout.frames] == [
        ("root", "left", "answer"),
        ("root", "right", "answer"),
    ]
    assert all(frame.spec.ref.owner is MemoryOwner.AGENT for frame in inline_layout.frames)


def test_apply_is_idempotent_and_does_not_change_authored_identity() -> None:
    flow = load_flow(_NESTED).compiled
    manifest = flow_manifest(flow)
    fingerprint = flow_fingerprint(flow)
    layout = resolve_memory_layout(flow)

    assert resolve_memory_layout(flow) == layout
    apply_memory_layout(flow, layout)
    assert resolve_memory_layout(flow) == layout
    first_scopes = tuple(target.node.memory_scope for target in layout.targets)
    apply_memory_layout(flow, layout)

    assert tuple(target.node.memory_scope for target in layout.targets) == first_scopes
    assert flow_manifest(flow) == manifest
    assert flow_fingerprint(flow) == fingerprint


def test_apply_scope_mismatch_is_atomic() -> None:
    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    final = layout.targets[-1]
    final.node.memory_scope = MemoryScope(
        root_flow_id="root",
        static_path=("root", "different"),
    )

    with pytest.raises(MemorySetupError, match="different memory scope"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets[:-1])
    assert final.node.memory_scope == MemoryScope(
        root_flow_id="root",
        static_path=("root", "different"),
    )


def test_apply_revalidates_forged_nested_layout_scope_before_stamping() -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical root identity."""

    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    forged = layout.targets[-1].scope
    object.__setattr__(forged, "root_flow_id", StringSubclass("root"))

    with pytest.raises(TypeError, match="root_flow_id must be a string"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets)


def test_apply_revalidates_forged_nested_frame_spec_before_stamping() -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical policy identity."""

    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    object.__setattr__(
        layout.frames[0].spec,
        "policy_name",
        StringSubclass("outer"),
    )

    with pytest.raises(TypeError, match="policy_name must be a string"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets)


def test_apply_revalidates_forged_equal_existing_scope_atomically() -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical root identity."""

    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    final = layout.targets[-1]
    forged = copy.deepcopy(final.scope)
    object.__setattr__(forged, "root_flow_id", StringSubclass("root"))
    final.node.memory_scope = forged

    with pytest.raises(MemorySetupError, match="noncanonical memory scope"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets[:-1])
    assert final.node.memory_scope is forged


def test_apply_stale_object_target_is_atomic() -> None:
    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    replacement = copy.deepcopy(flow.nodes["invoke"])
    flow.nodes["invoke"] = replacement

    with pytest.raises(MemorySetupError, match="same node"):
        apply_memory_layout(flow, layout)

    assert replacement.memory_scope is None
    assert all(target.node.memory_scope is None for target in layout.targets)


def test_apply_rejects_shared_child_targets_before_stamping() -> None:
    child = load_flow(
        """id: child
name: child
memory_policy: reflective
nodes:
  answer:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: {topic: child}
    output: str
output: ${answer.output}
"""
    ).compiled
    flow = CompiledFlow(
        {
            "left": CallNode("left", flow_id="child", child=child),
            "right": CallNode("right", flow_id="child", child=child),
        },
        [],
        wiring={"left": {}, "right": {}},
        flow_id="root",
    )
    layout = resolve_memory_layout(flow)

    with pytest.raises(MemorySetupError, match="multiple static paths"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets)


def test_recursive_static_child_cycle_is_rejected() -> None:
    flow = CompiledFlow({}, [], wiring={}, flow_id="root")
    cycle = CallNode("cycle", flow_id="root", child=flow)
    flow.nodes["cycle"] = cycle
    flow.wiring["cycle"] = {}

    with pytest.raises(MemorySetupError, match="recursive child-flow cycle"):
        resolve_memory_layout(flow)

    assert cycle.memory_scope is None


def test_flow_frame_spec_retains_config_and_origin_outside_node_scopes() -> None:
    flow = load_flow(_NESTED, source_label="memory.yaml").compiled
    layout = resolve_memory_layout(flow)
    inner = layout.frames[1]

    assert inner.spec.config == {"limit": 3}
    assert inner.origin == flow.nodes["invoke"].child.memory_policy.origin
    assert inner.spec.ref == MemoryFrameRef(
        "root",
        ("root", "invoke"),
        MemoryOwner.FLOW,
    )
    assert not hasattr(inner.spec, "origin")
