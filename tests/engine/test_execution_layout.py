"""Memory-independent static execution paths and atomic scope application."""

from __future__ import annotations

import copy

import pytest

from yian.compile.manifest import flow_fingerprint, flow_manifest
from yian.compile.model import END_ID, START_ID, CompiledFlow, Flow
from yian.diagnostics import OriginRef
from yian.nodes.code import CodeNode
from yian.nodes.contracts import ExecutionPosition, ExecutionScope
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode


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
    from yian.compile.execution_layout import (
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
    from yian.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )
    from yian.compose import load_flow
    from yian.compile.llm_cascade import apply_effective_llm_map

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


def test_agent_variable_owners_are_authored_callsites_and_survive_call_cloning():
    from yian.compile.execution_layout import apply_execution_layout, resolve_execution_layout
    from yian.compile.expand import call_subgraph
    from yian.compile.llm_cascade import apply_effective_llm_map
    from yian.compose import load_flow
    from yian.nodes.agent import AgentNode

    source = _REPEATED_DEFINITION.replace(
        "kind: code\n        code: |\n          return 1\n        output: int",
        "kind: agent\n        variables: {instruction: 'str = Summarize clearly'}\n"
        "        prompt: ${instruction}\n        output: str",
    )
    flow = load_flow(source).compiled
    apply_effective_llm_map(flow, {
        path: {"provider": "test", "model": "test"}
        for path in ("left/answer", "right/answer")
    })
    layout = resolve_execution_layout(flow)
    owners = {
        (target.scope.static_path, variable.name)
        for target in layout.targets
        for variable in target.node.declared_variables()
    }
    assert owners == {
        (("left", "answer"), "instruction"),
        (("right", "answer"), "instruction"),
    }
    assert all(target.node.execution_scope is None for target in layout.targets)

    apply_execution_layout(flow, layout)
    for callsite in ("left", "right"):
        spawner = flow.nodes[callsite]
        fragment = call_subgraph(
            spawner.child, callsite, {}, execution_scope=spawner.execution_scope,
        )
        template = spawner.child.nodes["answer"]
        clone = next(node for node in fragment.nodes.values() if isinstance(node, AgentNode))
        variable, = clone.declared_variables()
        assert (clone.execution_scope.static_path, variable.name) in owners
        assert clone.execution_scope.flow_path == (callsite,)
        assert clone.execution_scope.occurrence_path == (ExecutionPosition("call", (callsite,)),)
        assert variable is not template.declared_variables()[0]
        assert variable.default == "Summarize clearly"
        assert clone.memory_scope is None


def test_map_items_and_loop_iterations_share_the_authored_agent_variable_owner():
    from yian.compile.execution_layout import apply_execution_layout, resolve_execution_layout
    from yian.compile.expand import loop_continue_subgraph, map_subgraph
    from yian.compile.llm_cascade import apply_effective_llm_map
    from yian.compose import load_flow
    from yian.nodes.agent import AgentNode
    from yian.nodes.loop import LoopNode

    flow = load_flow("""id: variable_owners
name: variable_owners
defs:
  body:
    input: {n: int}
    nodes:
      writer:
        kind: agent
        input: {n: '${input.n}'}
        variables: {instruction: 'str = Summarize clearly'}
        prompt: '${instruction}: ${n}'
        output: {n: int}
    output: ${writer.output}
  repeated:
    input: {n: int}
    nodes:
      repeat:
        kind: loop
        call: body
        times: 2
        input: {n: '${input.n}'}
    output: ${repeat.output}
nodes:
  outer:
    kind: map
    call: repeated
    over: [3, 3]
    input: {n: '${item}'}
output: ${outer.output}
""").compiled
    apply_effective_llm_map(flow, {
        "outer/repeat/writer": {"provider": "test", "model": "test"},
    })
    layout = resolve_execution_layout(flow)
    authored = next(target for target in layout.targets if isinstance(target.node, AgentNode))
    owner = (authored.scope.static_path, authored.node.declared_variables()[0].name)
    assert owner == (("outer", "repeat", "writer"), "instruction")
    apply_execution_layout(flow, layout)
    spawner = flow.nodes["outer"]
    fragment = map_subgraph(
        spawner.child, spawner.id, [{"n": 3}, {"n": 3}], execution_scope=spawner.execution_scope,
    )
    drivers = sorted(
        (node for node in fragment.nodes.values() if isinstance(node, LoopNode)),
        key=lambda node: node.execution_scope.occurrence_path[0].index,
    )
    assert len(drivers) == 2
    occurrences = set()
    variables = []
    for item, current in enumerate(drivers):
        for iteration in range(2):
            next_driver = current.respawn(iteration + 1)
            body = loop_continue_subgraph(
                current.child, current.origin_id, {"n": 3}, iteration, next_driver,
                execution_scope=current.execution_scope,
            )
            clone = next(node for node in body.nodes.values() if isinstance(node, AgentNode))
            variable, = clone.declared_variables()
            assert (clone.execution_scope.static_path, variable.name) == owner
            assert clone.execution_scope.occurrence_path == (
                ExecutionPosition("map_item", ("outer",), item),
                ExecutionPosition("loop_body", ("outer", "repeat"), iteration),
            )
            assert variable.default == "Summarize clearly"
            assert clone.memory_scope is None
            occurrences.add(clone.execution_scope.occurrence_path)
            variables.append(variable)
            current = next_driver
    assert len(occurrences) == 4
    assert len({id(variable) for variable in variables}) == 4
    assert authored.node.execution_scope.occurrence_path == ()


def test_aliased_child_targets_are_rejected_before_any_scope_is_stamped():
    from yian.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )
    from yian.nodes.call import CallNode

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
    from yian.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )
    from yian.compile.llm_cascade import apply_effective_llm_map
    from yian.compile.memory_layout import (
        apply_memory_layout, resolve_memory_layout,
    )
    from yian.compose import load_flow

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
        assert memory_scope.static_path == target.scope.static_path
        assert memory_scope.memories[0] == "outer"
        assert target.scope.occurrence_path == ()
    child_start = next(
        target for target in execution_layout.targets
        if target.locator == ("left", START_ID)
    )
    assert child_start.scope.flow_path == ("left",)
    assert child_start.scope.boundary == "start"


def _two_node_layout():
    from yian.compile.execution_layout import resolve_execution_layout

    flow = CompiledFlow({
        "first": CodeNode("first", code="return 1"),
        "last": CodeNode("last", code="return 2"),
    }, [])
    return flow, resolve_execution_layout(flow)


@pytest.mark.parametrize("change", ["new-target", "replaced-target", "invalid-key"])
def test_stale_graph_is_rejected_without_stamping_any_target(change):
    from yian.compile.execution_layout import apply_execution_layout

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
    from yian.compile.execution_layout import (
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
    from yian.compile.execution_layout import apply_execution_layout

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
    from yian.compile.execution_layout import resolve_execution_layout
    from yian.nodes.call import CallNode

    flow = Flow({}, [])
    node = CallNode("recurse", flow_id="child", child=flow)
    flow.nodes[node.id] = node
    with pytest.raises(ValueError, match="recursive child-flow cycle"):
        resolve_execution_layout(flow)
    assert node.execution_scope is None


def test_static_children_are_discovered_through_the_node_owned_hook():
    from yian.compile.execution_layout import (
        apply_execution_layout, resolve_execution_layout,
    )
    from yian.nodes.base import Node, NodeKind

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
    from yian.compile.execution_layout import resolve_execution_layout

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
    from yian.compile.execution_layout import apply_execution_layout, resolve_execution_layout
    from yian.nodes.loop import LoopNode

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
    from yian.compile.execution_layout import apply_execution_layout, resolve_execution_layout
    from yian.nodes.loop import LoopNode

    node = LoopNode("loop", flow_id="body")
    flow = Flow({"first": CodeNode("first", code="return 1"), "loop": node}, [])
    layout = resolve_execution_layout(flow)
    target = next(item for item in layout.targets if item.node is node)
    object.__setattr__(target.scope, "occurrence_path", position)
    with pytest.raises(ValueError):
        apply_execution_layout(flow, layout)
    assert all(item.execution_scope is None for item in flow.nodes.values())
