"""The pure CALL builder `call_subgraph` — pins its output to the CALL residual's clone shape.

`call_subgraph(child, callsite, record)` wraps `clone_child`, bakes `commit_as=callsite` on the
cloned child END filler, and returns a `Flow` (the self-describing fragment a CALL spawner grows
into). This test pins: `start_id` is the namespaced child START; the terminal (`end_id`) is the
namespaced child END carrying `commit_as == callsite`; and every node id is `ns`-prefixed under the
callsite.
"""

import pytest

from agent_composer.compile.expand import call_subgraph, ns
from agent_composer.compile.model import END_ID, START_ID, Flow
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
        ("root", "call"),
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
            ("root", "call", node_id),
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
        ("root", "call"),
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
        ("root", "call"),
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
