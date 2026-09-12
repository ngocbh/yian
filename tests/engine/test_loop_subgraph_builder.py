"""The pure LOOP CONTINUE builder `loop_continue_subgraph` — pins body + next-driver shape.

`loop_continue_subgraph(child, origin, carried, k, driver)` splices body_k (NO commit_as — its
Output feeds the next driver by plain wiring) + the fresh `L~(k+1)` driver + the producer edge
`body_k.END -> L~(k+1)`. Bodies are keyed on ORIGIN (not the running driver id), so live `run` and
durable reconstruction builds the SAME `L#k/…` namespace.
"""
import copy

import pytest

from agent_composer.compile.expand import loop_continue_subgraph, map_callsite, ns
from agent_composer.compile.model import CompiledFlow, Edge, END_ID, Flow, START_ID
from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    MemoryFrameRef,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOwner,
    MemoryScope,
)
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.contracts import Binding, ChildFlowBoundary, Parameter, SeedConnection
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.start import StartNode

from tests.engine.test_expand import _child_flow


def _driver(origin, k, child):
    d = LoopNode(f"{origin}~{k}", flow_id="f", child=child, predicate_kind="while",
                 predicate="${x}", max_iters=10, iteration=k, origin_id=origin)
    d.params = child.nodes[child.start_id].params
    return d


def _empty_child():
    """Return a valid zero-field loop body."""
    return CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID, params=[]),
            END_ID: EndNode.record(END_ID, output_names=[]),
        },
        edges=[Edge("boundary", START_ID, END_ID, ordering=True)],
        wiring={START_ID: {}, END_ID: {}},
    )


def test_continue_subgraph_body_keyed_on_origin_not_driver():
    child = _child_flow()
    driver = _driver("lp", 1, child)                     # origin=lp; body must key on lp, not lp~0
    sg = loop_continue_subgraph(child, origin="lp", carried={"x": 1}, k=0, driver=driver)
    assert isinstance(sg, Flow)
    callsite = map_callsite("lp", 0)                      # "lp#0" (origin-keyed, NOT lp~0#0)
    assert sg.start_id == ns(callsite, child.start_id)
    assert sg.end_id == ns(callsite, END_ID)
    assert all(nid.startswith(callsite + "/") or nid == "lp~1" for nid in sg.nodes)


def test_continue_body_end_has_no_commit_as():
    child = _child_flow()
    sg = loop_continue_subgraph(child, "lp", {"x": 1}, 0, _driver("lp", 1, child))
    assert sg.nodes[ns(map_callsite("lp", 0), END_ID)].commit_as is None


def test_continue_splices_next_driver_and_edge():
    child = _child_flow()
    driver = _driver("lp", 1, child)
    sg = loop_continue_subgraph(child, "lp", {"x": 1}, 0, driver)
    assert "lp~1" in sg.nodes
    body_end = ns(map_callsite("lp", 0), END_ID)
    assert any(e.from_ == body_end and e.to == "lp~1" for e in sg.edges)
    assert sg.wiring["lp~1"] == {"x": Binding(f"${{{body_end}.output.x}}")}


def test_continue_context_prefixes_body_but_not_driver_local_origin():
    child = _child_flow()
    outer = OriginRef("root/nodes/call")
    loop_origin = OriginRef("child/nodes/loop")
    driver = _driver("lp", 1, child)
    driver.origin = loop_origin
    driver.origin_context = (outer,)

    sg = loop_continue_subgraph(
        child,
        origin="lp",
        carried={"x": 1},
        k=0,
        driver=driver,
        origin_context=(outer, loop_origin),
    )

    body_ids = set(sg.nodes) - {driver.id}
    assert body_ids
    assert {
        sg.nodes[node_id].origin_context for node_id in body_ids
    } == {(outer, loop_origin)}
    assert sg.nodes[driver.id].origin_context == (outer,)
    assert sg.nodes[driver.id].failure_trace() == (outer, loop_origin)


def test_continue_seeds_only_fields_declared_by_the_body_start():
    """Carry loop-only state forward without adding undeclared child START wiring."""
    child = _child_flow()
    driver = _driver("lp", 1, child)
    driver.params = [Parameter(name="x"), Parameter(name="exited")]

    sg = loop_continue_subgraph(
        child,
        "lp",
        {"x": 1, "exited": False},
        0,
        driver,
    )

    assert sg.wiring[sg.start_id] == {"x": Binding(1)}
    body_end = ns(map_callsite("lp", 0), END_ID)
    assert sg.wiring[driver.id] == {
        "x": Binding(f"${{{body_end}.output.x}}"),
        "exited": Binding(f"${{{body_end}.output.exited}}"),
    }
    assert sg.input_connections == (SeedConnection(sg.start_id, "x", ("x",)),)
    assert sg.child_boundaries == (ChildFlowBoundary(sg.start_id, body_end),)


def test_continue_orders_zero_field_body_before_next_driver():
    child = _empty_child()
    driver = LoopNode(
        "lp~1",
        flow_id="f",
        child=child,
        child_inputs=[],
        predicate_kind="times",
        times=2,
        max_iters=2,
        iteration=1,
        origin_id="lp",
    )
    driver.params = []

    sg = loop_continue_subgraph(child, "lp", {}, 0, driver)
    body_end = ns(map_callsite("lp", 0), END_ID)

    assert Edge(
        id=f"{body_end}->lp~1#0",
        from_=body_end,
        to="lp~1",
        ordering=True,
    ) in sg.edges
    assert sg.wiring["lp~1"] == {}
    assert sg.input_connections == ()
    assert sg.child_boundaries == (ChildFlowBoundary(sg.start_id, body_end),)


def test_continue_keys_body_on_origin_when_driver_id_differs():
    # A running driver `lp~2` with origin `lp` must still splice bodies under `lp#k`, never `lp~2#k`.
    child = _child_flow()
    driver = _driver("lp", 3, child)
    sg = loop_continue_subgraph(child, origin="lp", carried={"x": 5}, k=2, driver=driver)
    assert ns(map_callsite("lp", 2), child.start_id) in sg.nodes
    assert not any(nid.startswith("lp~2#") for nid in sg.nodes)
    assert "lp~3" in sg.nodes


def test_continue_appends_body_iteration_but_driver_keeps_outer_occurrence() -> None:
    child = _child_flow()
    outer_frame = MemoryFrameRef("root", ("root",), MemoryOwner.FLOW)
    for node_id, node in child.nodes.items():
        node.memory_scope = MemoryScope(
            "root",
            ("root", "outer_map", "lp", node_id),
            frame_stack=(outer_frame,),
        )
    outer = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("root", "outer_map"),
        1,
    )
    loop_scope = MemoryScope(
        "root",
        ("root", "outer_map", "lp"),
        frame_stack=(outer_frame,),
        occurrence_path=(outer,),
    )
    driver = _driver("lp", 3, child)
    driver.memory_scope = loop_scope

    fragment = loop_continue_subgraph(
        child,
        origin="lp",
        carried={"x": 1},
        k=2,
        driver=driver,
        memory_scope=loop_scope,
    )

    expected_body = (
        outer,
        MemoryOccurrenceSegment(
            MemoryOccurrenceKind.LOOP_ITERATION,
            ("root", "outer_map", "lp"),
            2,
        ),
    )
    assert {
        node.memory_scope.occurrence_path
        for node_id, node in fragment.nodes.items()
        if node_id != driver.id
    } == {expected_body}
    assert fragment.nodes[driver.id].memory_scope.occurrence_path == (outer,)


def test_continue_revalidates_supplied_driver_memory_scope() -> None:
    child = _child_flow()
    driver = _driver("lp", 1, child)
    base_scope = MemoryScope("root", ("root", "lp"))
    forged = copy.deepcopy(base_scope)
    occurrence = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("root", "outer"),
        0,
    )
    object.__setattr__(forged, "occurrence_path", (occurrence,))
    object.__setattr__(occurrence, "index", False)
    driver.memory_scope = forged

    with pytest.raises(TypeError, match="index must be an integer"):
        loop_continue_subgraph(
            child,
            origin="lp",
            carried={"x": 1},
            k=0,
            driver=driver,
            memory_scope=base_scope,
        )

    assert driver.memory_scope is forged
    assert all(node.memory_scope is None for node in child.nodes.values())
