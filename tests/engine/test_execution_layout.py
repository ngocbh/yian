"""Memory-independent static execution paths and atomic scope application."""

from __future__ import annotations

import copy

import pytest

from agent_composer.compile.manifest import flow_fingerprint, flow_manifest
from agent_composer.compile.model import END_ID, START_ID, CompiledFlow, Flow
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.code import CodeNode
from agent_composer.nodes.contracts import ExecutionPosition, ExecutionScope
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode


_REPEATED_DEFINITION = """id: root
name: root
defs:
  child:
    nodes:
      answer:
        kind: code
        code: |
          return 1
        output: int
    output: ${answer.output}
nodes:
  left: {kind: call, call: child}
  right: {kind: call, call: child}
output: ${right.output}
"""


def test_execution_scope_has_an_explicit_manifest_policy():
    node = CodeNode("plain", code="return 1")
    flow = CompiledFlow({"plain": node}, [])
    assert node.execution_scope is None
    before = flow_manifest(flow)
    fingerprint = flow_fingerprint(flow)

    node.execution_scope = ExecutionScope(("plain",))

    assert flow_manifest(flow) == before
    assert flow_fingerprint(flow) == fingerprint
    node.misspelled_execution_scope = node.execution_scope
    with pytest.raises(ValueError, match="unknown"):
        flow_manifest(flow)


@pytest.mark.parametrize("flow_type", [Flow, CompiledFlow])
def test_execution_layout_exists_without_memory_policies(flow_type):
    from agent_composer.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )

    node = CodeNode("authored~3/step", code="return 1", title="display name")
    node.origin = OriginRef("unrelated:source")
    flow = flow_type(
        {
            node.id: node,
            START_ID: StartNode(START_ID, params=[]),
            END_ID: EndNode.record(END_ID, output_names=[]),
        },
        [],
    )
    layout = resolve_execution_layout(flow)
    expected = {
        (END_ID,): ExecutionScope((END_ID,), boundary="end"),
        (START_ID,): ExecutionScope((START_ID,), boundary="start"),
        (node.id,): ExecutionScope((node.id,)),
    }
    assert {target.locator: target.scope for target in layout.targets} == expected
    assert [target.locator for target in layout.targets] == sorted(expected)
    assert all(target.node.execution_scope is None for target in layout.targets)
    node.title = "another display name"
    node.origin = OriginRef("different:source")
    assert resolve_execution_layout(flow) == layout

    apply_execution_layout(flow, layout)
    assert all(target.node.execution_scope == target.scope for target in layout.targets)
    assert all(target.node.memory_scope is None for target in layout.targets)
    apply_execution_layout(flow, layout)
    assert resolve_execution_layout(flow) == layout

    plain = flow_type({"plain": CodeNode("plain", code="return 1")}, [])
    plain_layout = resolve_execution_layout(plain)
    apply_execution_layout(plain, plain_layout)
    assert plain.nodes["plain"].execution_scope == ExecutionScope(("plain",))
    assert resolve_execution_layout(flow_type({}, [])).targets == ()


def test_callsites_to_one_definition_have_distinct_paths():
    from agent_composer.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )
    from agent_composer.compose import load_flow
    from agent_composer.compile.llm_cascade import apply_effective_llm_map

    flow = load_flow(_REPEATED_DEFINITION).compiled
    apply_effective_llm_map(flow, {})
    layout = resolve_execution_layout(flow)
    targets = {target.locator: target for target in layout.targets}
    left = targets[("left", "answer")]
    right = targets[("right", "answer")]
    assert left.node is not right.node
    assert left.scope == ExecutionScope(("left", "answer"), flow_path=("left",))
    assert right.scope == ExecutionScope(("right", "answer"), flow_path=("right",))
    assert targets[("left", START_ID)].scope.boundary == "start"
    assert targets[("right", END_ID)].scope.boundary == "end"

    apply_execution_layout(flow, layout)
    assert left.node.execution_scope == left.scope
    assert right.node.execution_scope == right.scope


def test_aliased_child_targets_are_rejected_before_any_scope_is_stamped():
    from agent_composer.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )
    from agent_composer.nodes.call import CallNode

    child = Flow({"answer": CodeNode("answer", code="return 1")}, [])
    flow = CompiledFlow({
        "left": CallNode("left", flow_id="child", child=child),
        "right": CallNode("right", flow_id="child", child=child),
    }, [])
    layout = resolve_execution_layout(flow)
    with pytest.raises(ValueError, match="multiple static paths"):
        apply_execution_layout(flow, layout)
    assert all(target.node.execution_scope is None for target in layout.targets)


@pytest.mark.parametrize("child_policy", [True, False])
def test_execution_scope_agrees_with_existing_memory_ownership_paths(child_policy):
    from agent_composer.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )
    from agent_composer.compile.llm_cascade import apply_effective_llm_map
    from agent_composer.compile.memory_layout import (
        apply_memory_layout, resolve_memory_layout,
    )
    from agent_composer.compose import load_flow

    source = _REPEATED_DEFINITION.replace("name: root\n", "name: root\nlong_term_memory: outer\n")
    source = source.replace(
        "kind: code\n        code: |\n          return 1",
        "kind: agent\n        prompt: answer\n        long_term_memory: local",
    )
    if child_policy:
        source = source.replace("  child:\n", "  child:\n    long_term_memory: inner\n")
    flow = load_flow(source).compiled
    apply_effective_llm_map(flow, {
        path: {"provider": "openai", "model": "test-model"}
        for path in ("left/answer", "right/answer")
    })
    memory_layout = resolve_memory_layout(flow)
    apply_memory_layout(flow, memory_layout)
    old_memory = {target.locator: target.node.memory_scope for target in memory_layout.targets}
    execution_layout = resolve_execution_layout(flow)
    apply_execution_layout(flow, execution_layout)

    assert resolve_memory_layout(flow) == memory_layout
    for target in execution_layout.targets:
        memory_scope = old_memory[target.locator]
        assert target.node.memory_scope is memory_scope
        assert memory_scope.static_path == ("root", *target.scope.static_path)
        assert target.scope.occurrence_path == ()
        if memory_scope.boundary is not None:
            assert memory_scope.boundary.kind.value == target.scope.boundary
            assert memory_scope.boundary.frame_ref.static_path == (
                "root", *target.scope.flow_path,
            )
    child_start = next(
        target for target in execution_layout.targets
        if target.locator == ("left", START_ID)
    )
    assert child_start.scope.flow_path == ("left",)
    assert child_start.scope.boundary == "start"
    if not child_policy:
        assert old_memory[child_start.locator].boundary is None


def _two_node_layout():
    from agent_composer.compile.execution_layout import resolve_execution_layout

    flow = CompiledFlow({
        "first": CodeNode("first", code="return 1"),
        "last": CodeNode("last", code="return 2"),
    }, [])
    return flow, resolve_execution_layout(flow)


@pytest.mark.parametrize("change", ["new-target", "replaced-target", "invalid-key"])
def test_stale_graph_is_rejected_without_stamping_any_target(change):
    from agent_composer.compile.execution_layout import apply_execution_layout

    flow, layout = _two_node_layout()
    if change == "new-target":
        flow.nodes["new"] = CodeNode("new", code="return 3")
    elif change == "replaced-target":
        flow.nodes["last"] = copy.deepcopy(flow.nodes["last"])
    else:
        flow.nodes["bad-key"] = flow.nodes.pop("last")

    with pytest.raises(ValueError):
        apply_execution_layout(flow, layout)
    assert all(node.execution_scope is None for node in flow.nodes.values())
    assert all(target.node.execution_scope is None for target in layout.targets)


@pytest.mark.parametrize("change", ["missing-target", "duplicate-target", "mutable-targets"])
def test_invalid_layout_is_rejected_without_stamping_any_target(change):
    from agent_composer.compile.execution_layout import (
        ResolvedExecutionLayout, apply_execution_layout,
    )

    flow, layout = _two_node_layout()
    if change == "missing-target":
        layout = ResolvedExecutionLayout(layout.targets[:-1])
    elif change == "duplicate-target":
        layout = ResolvedExecutionLayout((*layout.targets, layout.targets[-1]))
    else:
        object.__setattr__(layout, "targets", list(layout.targets))

    with pytest.raises((TypeError, ValueError)):
        apply_execution_layout(flow, layout)
    assert all(node.execution_scope is None for node in flow.nodes.values())


@pytest.mark.parametrize("location", ["layout", "existing"])
@pytest.mark.parametrize("change", ["different-path", "path-subclass", "position-index"])
def test_scope_validation_is_atomic_even_for_forged_frozen_values(location, change):
    from agent_composer.compile.execution_layout import apply_execution_layout

    class StringSubclass(str):
        pass

    flow, layout = _two_node_layout()
    final = layout.targets[-1]
    scope = copy.deepcopy(final.scope)
    if change == "different-path":
        object.__setattr__(scope, "static_path", ("other",))
    elif change == "path-subclass":
        object.__setattr__(scope, "static_path", (StringSubclass("last"),))
    else:
        position = ExecutionPosition("map_item", ("map",), 1)
        object.__setattr__(position, "index", True)
        object.__setattr__(scope, "occurrence_path", (position,))
    if location == "layout":
        object.__setattr__(final, "scope", scope)
    else:
        final.node.execution_scope = scope

    with pytest.raises((TypeError, ValueError)):
        apply_execution_layout(flow, layout)
    assert flow.nodes["first"].execution_scope is None
    assert final.node.execution_scope is (scope if location == "existing" else None)


def test_recursive_static_child_cycle_is_rejected_without_mutation():
    from agent_composer.compile.execution_layout import resolve_execution_layout
    from agent_composer.nodes.call import CallNode

    flow = Flow({}, [])
    node = CallNode("recurse", flow_id="child", child=flow)
    flow.nodes[node.id] = node
    with pytest.raises(ValueError, match="recursive child-flow cycle"):
        resolve_execution_layout(flow)
    assert node.execution_scope is None


def test_static_children_are_discovered_through_the_node_owned_hook():
    from agent_composer.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )
    from agent_composer.nodes.base import Node, NodeKind

    child = Flow({"leaf": CodeNode("leaf", code="return 1")}, [])

    class Wrapper(Node):
        kind = NodeKind.CODE

        def origin_children(self):
            return (child,)

    flow = Flow({"wrapper": Wrapper("wrapper")}, [])
    layout = resolve_execution_layout(flow)
    apply_execution_layout(flow, layout)
    assert child.nodes["leaf"].execution_scope == ExecutionScope(
        ("wrapper", "leaf"), flow_path=("wrapper",),
    )


def test_only_the_containing_flows_actual_boundaries_are_marked():
    from agent_composer.compile.execution_layout import resolve_execution_layout

    flow = Flow({
        "entry": StartNode("entry", params=[]),
        "result": EndNode.record("result", output_names=[]),
        "helper": StartNode("helper", params=[]),
    }, [], start_id="entry", end_id="result")
    layout = resolve_execution_layout(flow)
    assert {target.locator: target.scope.boundary for target in layout.targets} == {
        ("entry",): "start", ("helper",): None, ("result",): "end",
    }


def test_static_loop_scope_starts_at_driver_zero_and_reapplies_atomically():
    from agent_composer.compile.execution_layout import apply_execution_layout, resolve_execution_layout
    from agent_composer.nodes.loop import LoopNode

    node = LoopNode("loop", flow_id="body", times=0, max_iters=0, predicate_kind="times")
    flow = Flow({"first": CodeNode("first", code="return 1"), "loop": node}, [])
    layout = resolve_execution_layout(flow)
    target = next(item for item in layout.targets if item.node is node)
    assert target.scope.occurrence_path == (ExecutionPosition("loop_driver", ("loop",), 0),)
    assert flow.nodes["first"].execution_scope is None
    apply_execution_layout(flow, layout)
    apply_execution_layout(flow, layout)
    assert resolve_execution_layout(flow) == layout
    assert flow.nodes["first"].execution_scope.occurrence_path == ()


@pytest.mark.parametrize("position", [
    (), (ExecutionPosition("loop_driver", ("loop",), 1),),
    (ExecutionPosition("loop_driver", ("other",), 0),),
    (ExecutionPosition("loop_body", ("loop",), 0),),
    (ExecutionPosition("call", ("outer",)), ExecutionPosition("loop_driver", ("loop",), 0)),
])
def test_static_loop_layout_rejects_nonbootstrap_positions_before_stamping(position):
    from agent_composer.compile.execution_layout import apply_execution_layout, resolve_execution_layout
    from agent_composer.nodes.loop import LoopNode

    node = LoopNode("loop", flow_id="body")
    flow = Flow({"first": CodeNode("first", code="return 1"), "loop": node}, [])
    layout = resolve_execution_layout(flow)
    target = next(item for item in layout.targets if item.node is node)
    object.__setattr__(target.scope, "occurrence_path", position)
    with pytest.raises(ValueError):
        apply_execution_layout(flow, layout)
    assert all(item.execution_scope is None for item in flow.nodes.values())
