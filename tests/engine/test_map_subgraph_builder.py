"""The pure MAP builder `map_subgraph` — pins its output to `_grow_map`'s clone+fan-in shape.

`map_subgraph(child, spawner_id, records)` clones the child once per element (deep-namespaced under
`map_callsite(spawner_id, i)`), synthesizes a `map#/__start__` StartNode that fans out to every
element START via ORDERING edges, plus the list-`END` fan-in (an `EndNode.list_` carrying
`commit_as=spawner_id`, fed one `e{i}` group per element), and returns a `Flow`. This test pins:
over 2 records there are 2 child clones + the synthetic start + 1 list-END; `start_id` is the
synthetic start with an ordering edge to every element START; `end_id` is the list-END; the list-END
commits under the spawner; the end-wiring has `e0`/`e1`. The N=0 case pins that the only body node is
the list-END and that the synthetic start wires straight to it (so it schedules and emits `[]`).
"""

from agent_composer.compile.expand import map_callsite, map_subgraph, ns
from agent_composer.compile.model import END_ID, START_ID, Flow
from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    MemoryFrameRef,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOwner,
    MemoryScope,
)
from agent_composer.nodes.contracts import Binding

from tests.engine.test_expand import _child_flow


def test_map_subgraph_matches_grow_map_type():
    child = _child_flow()
    records = [{"x": 1}, {"x": 2}]
    sg = map_subgraph(child, spawner_id="m0", records=records)

    assert isinstance(sg, Flow)
    map_start_id = ns("m0", START_ID)
    map_end_id = ns("m0", END_ID)

    # 2 child clones (one per element callsite) + the synthetic start + 1 synthesized list-END.
    for i in range(2):
        callsite = map_callsite("m0", i)
        assert set(ns(callsite, nid) for nid in child.nodes) <= set(sg.nodes)
    assert map_start_id in sg.nodes
    assert map_end_id in sg.nodes

    # start_id == the synthetic fan-out start; end_id == the list collector.
    assert sg.start_id == map_start_id
    assert sg.end_id == map_end_id

    # The list-END commits its list Output under the spawner id.
    assert sg.nodes[map_end_id].commit_as == "m0"

    # The fan-in wiring has one e{i} group per element.
    end_wiring = sg.wiring[map_end_id]
    assert set(end_wiring) == {"e0", "e1"}
    for i in range(2):
        out_id = ns(map_callsite("m0", i), child.end_id)
        assert end_wiring[f"e{i}"] == Binding(f"${{{out_id}.output}}")

    # The synthetic start fans out to EVERY element START via an ordering edge (its sole gate).
    for i in range(2):
        elem_start = ns(map_callsite("m0", i), child.start_id)
        matched = [e for e in sg.edges if e.from_ == map_start_id and e.to == elem_start]
        assert len(matched) == 1
        edge = matched[0]
        assert edge.ordering is True          # ordering, not data (no phantom input_group group)
        assert edge.input_group is None
        assert edge.optional is False         # depends_on: mirrors the top-level start->root gate

    # N>0 so the synthetic start does NOT wire straight to the list-END (it fans in over the clones).
    assert not any(e.from_ == map_start_id and e.to == map_end_id for e in sg.edges)


def test_map_subgraph_n_ge_2_start_fans_out_to_all_element_starts():
    # A focused check that MAP with N>=2 produces a Flow whose synthetic start has an ordering edge
    # to ALL N element starts (the fan-out replacing the old N-root list).
    child = _child_flow()
    records = [{"x": i} for i in range(4)]
    sg = map_subgraph(child, spawner_id="m0", records=records)

    map_start_id = ns("m0", START_ID)
    fanout = {e.to for e in sg.edges if e.from_ == map_start_id and e.ordering}
    expected = {ns(map_callsite("m0", i), child.start_id) for i in range(4)}
    assert fanout == expected


def test_map_subgraph_shares_context_across_distinct_clones_and_boundaries():
    child = _child_flow()
    context = (OriginRef("root/nodes/map"),)

    sg = map_subgraph(
        child,
        spawner_id="m0",
        records=[{"x": 1}, {"x": 2}],
        origin_context=context,
    )

    first_body = ns(map_callsite("m0", 0), "head")
    second_body = ns(map_callsite("m0", 1), "head")
    assert first_body != second_body
    assert sg.nodes[first_body].origin_context == context
    assert sg.nodes[second_body].origin_context == context
    assert sg.nodes[ns("m0", START_ID)].origin_context == context
    assert sg.nodes[ns("m0", END_ID)].origin_context == context


def test_map_subgraph_n0_wires_start_to_collector():
    child = _child_flow()
    context = (OriginRef("root/nodes/map"),)
    sg = map_subgraph(
        child,
        spawner_id="m0",
        records=[],
        origin_context=context,
    )

    map_start_id = ns("m0", START_ID)
    map_end_id = ns("m0", END_ID)
    # The only nodes are the synthetic start + the list-END; the start wires straight to the END
    # (an ordering edge) so the collector schedules off the fan-out and emits [].
    assert set(sg.nodes) == {map_start_id, map_end_id}
    assert sg.start_id == map_start_id
    assert sg.end_id == map_end_id
    edges_to_end = [e for e in sg.edges if e.to == map_end_id]
    assert len(edges_to_end) == 1
    assert edges_to_end[0].from_ == map_start_id
    assert edges_to_end[0].ordering is True
    assert sg.nodes[map_end_id].commit_as == "m0"
    assert sg.nodes[map_start_id].origin_context == context
    assert sg.nodes[map_end_id].origin_context == context


def test_map_subgraph_appends_item_occurrence_and_scopes_generated_fan_nodes() -> None:
    child = _child_flow()
    outer_frame = MemoryFrameRef("root", ("root",), MemoryOwner.FLOW)
    for node_id, node in child.nodes.items():
        node.memory_scope = MemoryScope(
            "root",
            ("root", "map", node_id),
            frame_stack=(outer_frame,),
        )
    outer = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.LOOP_ITERATION,
        ("root", "outer_loop"),
        4,
    )
    spawner_scope = MemoryScope(
        "root",
        ("root", "map"),
        frame_stack=(outer_frame,),
        occurrence_path=(outer,),
    )

    fragment = map_subgraph(
        child,
        spawner_id="map",
        records=[{"x": 1}, {"x": 2}],
        memory_scope=spawner_scope,
    )

    for index in range(2):
        expected = (
            outer,
            MemoryOccurrenceSegment(
                MemoryOccurrenceKind.MAP_ITEM,
                ("root", "map"),
                index,
            ),
        )
        clone_prefix = map_callsite("map", index)
        assert {
            node.memory_scope.occurrence_path
            for node_id, node in fragment.nodes.items()
            if node_id.startswith(f"{clone_prefix}/")
        } == {expected}

    generated_scope = MemoryScope(
        "root",
        ("root", "map"),
        frame_stack=(outer_frame,),
        occurrence_path=(outer,),
        boundary=None,
    )
    assert fragment.nodes[ns("map", START_ID)].memory_scope == generated_scope
    assert fragment.nodes[ns("map", END_ID)].memory_scope == generated_scope
