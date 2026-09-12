"""The pure CALL builder `call_subgraph` — pins its output to the CALL residual's clone shape.

`call_subgraph(child, callsite, record)` wraps `clone_child`, bakes `commit_as=callsite` on the
cloned child END filler, and returns a `Flow` (the self-describing fragment a CALL spawner grows
into). This test pins: `start_id` is the namespaced child START; the terminal (`end_id`) is the
namespaced child END carrying `commit_as == callsite`; and every node id is `ns`-prefixed under the
callsite.
"""

import copy
from dataclasses import replace

import pytest

from agent_composer.compile.expand import call_subgraph, clone_child, map_callsite, map_subgraph, ns
from agent_composer.compile.manifest import expansion_digest, flow_fingerprint
from agent_composer.compile.model import END_ID, START_ID, Flow, validate_growth_metadata
from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    MemoryBoundary,
    MemoryBoundaryKind,
    MemoryFrameRef,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOwner,
    MemoryScope,
)
from agent_composer.nodes.contracts import ChildFlowBoundary, ExecutionScope, SeedConnection

from tests.engine.test_expand import _child_flow


def test_call_subgraph_matches_grow_call_type():
    child = _child_flow()
    sg = call_subgraph(child, callsite="c0", record={"x": 1})

    assert isinstance(sg, Flow)
    # start_id == the namespaced child START (the sole seed point).
    assert sg.start_id == ns("c0", child.start_id)
    # Terminal (end_id) == the namespaced child END; its Output commits under the callsite.
    assert sg.end_id == ns("c0", END_ID)
    terminal = sg.nodes[ns("c0", END_ID)]
    assert terminal.commit_as == "c0"
    # Every cloned node id is namespaced under the callsite.
    assert all(nid.startswith("c0/") for nid in sg.nodes)
    assert set(sg.nodes) == {ns("c0", nid) for nid in child.nodes}
    assert sg.input_connections == (SeedConnection(sg.start_id, "x", ("x",)),)
    assert sg.child_boundaries == (ChildFlowBoundary(sg.start_id, sg.end_id),)


@pytest.mark.parametrize("malformation", [
    "mutable-connections", "mutable-boundaries", "mutable-path", "boolean-index",
    "negative-index", "deep-selector", "duplicate-target", "missing-start",
    "undeclared-input", "unwired-input", "invalid-parameters", "duplicate-boundary",
    "missing-end", "wrong-role", "different-flow", "wrong-owner", "forged-position",
])
def test_growth_metadata_revalidates_exact_targets_and_nested_fields(malformation):
    fragment = call_subgraph(_child_flow(), "call", {"x": 1})
    start = fragment.nodes[fragment.start_id]
    end = fragment.nodes[fragment.end_id]
    start.execution_scope = ExecutionScope(("call", START_ID), ("call",), boundary="start")
    end.execution_scope = ExecutionScope(("call", END_ID), ("call",), boundary="end")
    connection = fragment.input_connections[0]
    if malformation == "mutable-connections":
        fragment.input_connections = list(fragment.input_connections)
    elif malformation == "mutable-boundaries":
        fragment.child_boundaries = list(fragment.child_boundaries)
    elif malformation in ("mutable-path", "boolean-index", "negative-index", "deep-selector"):
        path = {
            "mutable-path": ["x"], "boolean-index": (True, "x"),
            "negative-index": (-1, "x"), "deep-selector": (0, "x", "nested"),
        }[malformation]
        object.__setattr__(connection, "seed_path", path)
    elif malformation == "duplicate-target":
        fragment.input_connections += (connection,)
    elif malformation == "missing-start":
        fragment.input_connections = (replace(connection, child_start_id="absent"),)
    elif malformation == "undeclared-input":
        fragment.input_connections = (replace(connection, input_name="absent"),)
    elif malformation == "unwired-input":
        fragment.wiring[fragment.start_id] = {}
    elif malformation == "invalid-parameters":
        start.params = ()
    elif malformation == "duplicate-boundary":
        fragment.child_boundaries += fragment.child_boundaries
    elif malformation == "missing-end":
        fragment.child_boundaries = (ChildFlowBoundary(fragment.start_id, "absent"),)
    elif malformation == "wrong-role":
        start.execution_scope = replace(start.execution_scope, boundary="end")
    elif malformation == "different-flow":
        end.execution_scope = replace(end.execution_scope, flow_path=("other",))
    elif malformation == "wrong-owner":
        start.execution_scope = replace(start.execution_scope, static_path=("other", START_ID))
    else:
        object.__setattr__(start.execution_scope, "occurrence_path", [])

    error = TypeError if malformation == "invalid-parameters" else (TypeError, ValueError)
    with pytest.raises(error):
        validate_growth_metadata(fragment)


def test_growth_metadata_and_execution_topology_have_a_separate_digest_policy():
    fragment = call_subgraph(_child_flow(), "call", {"x": 1})
    authored = flow_fingerprint(fragment)
    digest = expansion_digest(fragment)
    fragment.input_connections = (replace(fragment.input_connections[0], seed_path=(3, "x")),)
    assert flow_fingerprint(fragment) == authored
    assert expansion_digest(fragment) != digest
    digest = expansion_digest(fragment)
    fragment.nodes[fragment.start_id].execution_scope = ExecutionScope(
        ("call", START_ID), ("call",), boundary="start",
    )
    assert flow_fingerprint(fragment) == authored
    assert expansion_digest(fragment) != digest
    digest = expansion_digest(fragment)
    fragment.input_connections = ()
    fragment.child_boundaries = ()
    assert flow_fingerprint(fragment) == authored
    assert expansion_digest(fragment) != digest


@pytest.mark.parametrize("kind", ["call", "map"])
def test_clone_preserves_nested_recipes_without_reusing_runtime_target_ids(kind):
    child = _child_flow()
    nested = call_subgraph(_child_flow(), "inner", {"x": 2})
    child.nodes.update(nested.nodes)
    child.wiring.update(nested.wiring)
    child.edges.extend(nested.edges)
    child.input_connections = nested.input_connections
    child.child_boundaries = nested.child_boundaries
    before = copy.deepcopy((child.input_connections, child.child_boundaries))

    if kind == "call":
        fragment = call_subgraph(child, "outer", {"x": 1})
        prefix, selector = "outer", ("x",)
    else:
        fragment = map_subgraph(child, "outer", [{"x": 1}])
        prefix, selector = map_callsite("outer", 0), (0, "x")

    assert fragment.input_connections == (
        SeedConnection(ns(prefix, child.start_id), "x", selector),
        SeedConnection(ns(prefix, nested.start_id), "x", selector),
    )
    assert fragment.child_boundaries == (
        ChildFlowBoundary(ns(prefix, child.start_id), ns(prefix, child.end_id)),
        ChildFlowBoundary(ns(prefix, nested.start_id), ns(prefix, nested.end_id)),
    )
    validate_growth_metadata(fragment)
    assert (child.input_connections, child.child_boundaries) == before


def test_map_clone_rejects_a_preserved_recipe_that_would_need_two_item_indices():
    child = _child_flow()
    nested = call_subgraph(_child_flow(), "inner", {"x": 2})
    child.nodes.update(nested.nodes)
    child.wiring.update(nested.wiring)
    child.edges.extend(nested.edges)
    child.input_connections = (replace(nested.input_connections[0], seed_path=(2, "x")),)
    child.child_boundaries = nested.child_boundaries

    with pytest.raises(ValueError, match="seed_path"):
        clone_child(child, "outer", {"x": 1}, seed_prefix=(0,))

    assert child.input_connections[0].seed_path == (2, "x")


def test_call_subgraph_prefixes_every_cloned_node_including_boundaries():
    child = _child_flow()
    context = (OriginRef("root/nodes/call"),)

    sg = call_subgraph(
        child,
        callsite="c0",
        record={"x": 1},
        origin_context=context,
    )

    assert {node.origin_context for node in sg.nodes.values()} == {context}
    assert sg.nodes[sg.start_id].origin_context == context
    assert sg.nodes[sg.end_id].origin_context == context


def test_call_subgraph_inherits_occurrence_and_preserves_child_boundary() -> None:
    child = _child_flow()
    child_frame = MemoryFrameRef(
        "root",
        ("root", "outer_map", "call"),
        MemoryOwner.FLOW,
    )
    for node_id, node in child.nodes.items():
        boundary = None
        if node_id == START_ID:
            boundary = MemoryBoundary(MemoryBoundaryKind.START, child_frame)
        elif node_id == END_ID:
            boundary = MemoryBoundary(MemoryBoundaryKind.END, child_frame)
        node.memory_scope = MemoryScope(
            "root",
            ("root", "outer_map", "call", node_id),
            frame_stack=(child_frame,),
            boundary=boundary,
        )
    outer = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("root", "outer_map"),
        2,
    )
    spawner_scope = MemoryScope(
        "root",
        ("root", "outer_map", "call"),
        occurrence_path=(outer,),
    )

    fragment = call_subgraph(
        child,
        callsite="call",
        record={"x": 1},
        memory_scope=spawner_scope,
    )

    assert {
        node.memory_scope.occurrence_path for node in fragment.nodes.values()
    } == {(outer,)}
    assert fragment.nodes[fragment.start_id].memory_scope.boundary == MemoryBoundary(
        MemoryBoundaryKind.START,
        child_frame,
    )
    assert fragment.nodes[fragment.end_id].memory_scope.boundary == MemoryBoundary(
        MemoryBoundaryKind.END,
        child_frame,
    )


def test_call_subgraph_revalidates_supplied_memory_scope() -> None:
    child = _child_flow()
    occurrence = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("root", "outer"),
        0,
    )
    scope = MemoryScope(
        "root",
        ("root", "outer", "call"),
        occurrence_path=(occurrence,),
    )
    object.__setattr__(occurrence, "index", False)

    with pytest.raises(TypeError, match="index must be an integer"):
        call_subgraph(
            child,
            callsite="call",
            record={"x": 1},
            memory_scope=scope,
        )

    assert all(node.memory_scope is None for node in child.nodes.values())
