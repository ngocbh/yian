"""Seed-first growth: nodes describe expansion; compile constructs topology."""

import copy

import pytest

import yian.compile.expand as expand_module
from yian.compile.model import Flow, GrowthRejected
from yian.diagnostics import OriginRef
from yian.memory import (
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryScope,
)
from yian.nodes.base import Grow, Node, NodeKind
from yian.nodes.call import CallNode
from yian.nodes.contracts import Assertion, Parameter
from yian.nodes.loop import LoopNode
from yian.nodes.map import MapNode

from tests.engine.test_expand import _child_flow, _parent_with_spawner


def _growth_producers():
    """Return one live seed-producing spawner of every legal grow kind."""
    child = _child_flow()

    call = CallNode("call", flow_id="child", child=child)
    call.params = [Parameter(name="x")]
    call.post_asserts = [Assertion("output == x", "output == x")]
    call_grow = call.run({"x": 1})

    map_node = MapNode("map", flow_id="child", child=child)
    map_grow = map_node.run(
        {"over": [1, 2]}, bind_item=lambda item, index: {"x": item}
    )

    loop = LoopNode(
        "loop",
        flow_id="child",
        child=child,
        predicate_kind="while",
        predicate="x > 0",
        max_iters=3,
    )
    loop.params = child.nodes[child.start_id].params
    loop_grow = loop.run({"x": 1})

    return [
        (call, call_grow),
        (map_node, map_grow),
        (loop, loop_grow),
    ]


def _assert_same_fragment(actual: Flow, expected: Flow) -> None:
    """Compare construction-relevant topology and node-owned execution metadata."""
    assert isinstance(actual, Flow)
    assert actual.start_id == expected.start_id
    assert actual.end_id == expected.end_id
    assert set(actual.nodes) == set(expected.nodes)
    assert actual.edges == expected.edges
    assert actual.wiring == expected.wiring
    for node_id in actual.nodes:
        left = actual.nodes[node_id]
        right = expected.nodes[node_id]
        assert type(left) is type(right)
        assert left.commit_as == right.commit_as
        assert left.origin_id == right.origin_id
        assert left.origin_context == right.origin_context
        assert left.memory_scope == right.memory_scope
        assert left.post_asserts == right.post_asserts


def test_call_map_and_loop_return_seed_only():
    producers = _growth_producers()

    assert all(isinstance(grow, Grow) for _, grow in producers)
    assert all(not hasattr(grow, "subgraph") for _, grow in producers)
    assert producers[0][1].seed == {"x": 1}
    assert producers[1][1].seed == [{"x": 1}, {"x": 2}]
    assert producers[2][1].seed == ({"x": 1}, 0)


def test_loop_respawn_preserves_diagnostic_origin_separately_from_origin_id():
    """Every fresh driver keeps its authored site while retaining runtime attribution."""
    loop = LoopNode("loop", flow_id="child", origin_id="runtime-loop")
    authored_origin = OriginRef("root/node/loop")
    outer_origin = OriginRef("root/node/call")
    loop.origin = authored_origin
    loop.origin_context = (outer_origin,)

    clone = loop.respawn(2)

    assert clone.origin is authored_origin
    assert clone.origin_context == (outer_origin,)
    assert clone.failure_trace() == (outer_origin, authored_origin)
    assert clone.origin_id == "runtime-loop"


def test_compile_builder_derives_prefix_from_each_spawner_failure_trace():
    (call, call_grow), (map_node, map_grow), (loop, loop_grow) = _growth_producers()
    outer = OriginRef("root/nodes/outer")
    local_origins = [
        OriginRef("root/nodes/call"),
        OriginRef("root/nodes/map"),
        OriginRef("root/nodes/loop"),
    ]

    for spawner, local in zip((call, map_node, loop), local_origins, strict=True):
        spawner.origin_context = (outer,)
        spawner.origin = local

    call_fragment = expand_module.build_grow(call, call_grow.seed)
    assert {
        node.origin_context for node in call_fragment.nodes.values()
    } == {(outer, local_origins[0])}

    map_fragment = expand_module.build_grow(map_node, map_grow.seed)
    assert {
        node.origin_context for node in map_fragment.nodes.values()
    } == {(outer, local_origins[1])}

    loop_fragment = expand_module.build_grow(loop, loop_grow.seed)
    next_driver = loop_fragment.nodes["loop~1"]
    body_contexts = {
        node.origin_context
        for node_id, node in loop_fragment.nodes.items()
        if node_id != next_driver.id
    }
    assert body_contexts == {(outer, local_origins[2])}
    assert next_driver.origin_context == (outer,)
    assert next_driver.failure_trace() == (outer, local_origins[2])


def test_source_free_spawners_do_not_invent_generated_context():
    for spawner, grow in _growth_producers():
        fragment = expand_module.build_grow(spawner, grow.seed)
        assert all(node.origin_context == () for node in fragment.nodes.values())


def test_compile_builder_propagates_each_spawner_memory_occurrence() -> None:
    (call, call_grow), (map_node, map_grow), (loop, loop_grow) = _growth_producers()
    spawners = (call, map_node, loop)
    outer = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("outer",),
        3,
    )
    for spawner in spawners:
        spawner.child = copy.deepcopy(spawner.child)
        spawner.memory_scope = MemoryScope(
            ("outer", spawner.id),
            occurrence_path=(outer,),
        )
        for child_id, child_node in spawner.child.nodes.items():
            child_node.memory_scope = MemoryScope(("outer", spawner.id, child_id))

    call_fragment = expand_module.build_grow(call, call_grow.seed)
    assert {
        node.memory_scope.occurrence_path for node in call_fragment.nodes.values()
    } == {(outer,)}

    map_fragment = expand_module.build_grow(map_node, map_grow.seed)
    map_items = {
        node.memory_scope.occurrence_path[-1]
        for node_id, node in map_fragment.nodes.items()
        if "#" in node_id
    }
    assert map_items == {
        MemoryOccurrenceSegment(
            MemoryOccurrenceKind.MAP_ITEM,
            ("outer", "map"),
            0,
        ),
        MemoryOccurrenceSegment(
            MemoryOccurrenceKind.MAP_ITEM,
            ("outer", "map"),
            1,
        ),
    }

    loop_fragment = expand_module.build_grow(loop, loop_grow.seed)
    next_driver = loop_fragment.nodes["loop~1"]
    assert next_driver.memory_scope.occurrence_path == (outer,)
    assert {
        node.memory_scope.occurrence_path[-1]
        for node_id, node in loop_fragment.nodes.items()
        if node_id != next_driver.id
    } == {
        MemoryOccurrenceSegment(
            MemoryOccurrenceKind.LOOP_ITERATION,
            ("outer", "loop"),
            0,
        )
    }


def test_nested_map_then_call_extends_context_only_when_inner_expands():
    """A nested spawner carries outer ancestry without pre-stamping its baked child."""
    leaf = _child_flow()
    middle = _parent_with_spawner(leaf, {"x": 1})
    outer_origin = OriginRef("root/nodes/each")
    inner_origin = OriginRef("middle/nodes/inner")
    middle.nodes["sp"].origin = inner_origin
    outer = MapNode("each", flow_id="middle", child=middle)
    outer.origin = outer_origin

    map_fragment = expand_module.build_grow(outer, [{"x": 1}])
    inner = map_fragment.nodes["each#0/sp"]

    assert inner.origin_context == (outer_origin,)
    assert all(node.origin_context == () for node in inner.child.nodes.values())

    call_fragment = expand_module.build_grow(inner, {"x": 1})

    assert call_fragment.nodes["each#0/sp/head"].origin_context == (
        outer_origin,
        inner_origin,
    )


def test_compile_builder_reconstructs_every_growth_kind():
    from yian.compile.expand import (
        build_grow,
        call_subgraph,
        loop_continue_subgraph,
        map_subgraph,
    )

    (call, call_grow), (map_node, map_grow), (loop, loop_grow) = _growth_producers()

    call_expected = call_subgraph(
        call.child,
        callsite=call.id,
        record=dict(call_grow.seed),
        post_asserts=call.post_asserts,
        input_names={param.name for param in call.params or []},
    )
    _assert_same_fragment(build_grow(call, call_grow.seed), call_expected)

    map_expected = map_subgraph(
        map_node.child,
        spawner_id=map_node.id,
        records=[dict(record) for record in map_grow.seed],
    )
    _assert_same_fragment(build_grow(map_node, map_grow.seed), map_expected)

    carried, iteration = loop_grow.seed
    loop_expected = loop_continue_subgraph(
        loop.child,
        loop.origin_id,
        dict(carried),
        iteration,
        loop.respawn(iteration + 1),
    )
    _assert_same_fragment(build_grow(loop, list(loop_grow.seed)), loop_expected)


@pytest.mark.parametrize(
    ("producer_index", "seed", "original_type", "message"),
    [
        (0, [], TypeError, "CALL growth seed must be a record"),
        (0, {1: "value"}, TypeError, "record keys must be strings"),
        (0, {"": "value"}, ValueError, "record keys must be non-empty"),
        (1, {}, TypeError, "MAP growth seed must be a list of records"),
        (1, [[]], TypeError, "MAP item 0 growth seed must be a record"),
        (2, [{}], ValueError, "two-item"),
        (2, [[], 0], TypeError, "LOOP carried growth seed must be a record"),
        (2, [{}, True], TypeError, "iteration must be an integer"),
        (2, [{}, -1], ValueError, "iteration must be non-negative"),
    ],
    ids=(
        "call-container",
        "call-key-type",
        "call-empty-key",
        "map-container",
        "map-item",
        "loop-arity",
        "loop-record",
        "loop-boolean-index",
        "loop-negative-index",
    ),
)
def test_compile_builder_marks_only_explicit_legal_seed_rejection(
    producer_index: int,
    seed,
    original_type: type[Exception],
    message: str,
):
    """Malformed legal-kind seeds retain their typed rejection in one marker."""
    from yian.compile.expand import build_grow

    spawner = _growth_producers()[producer_index][0]

    with pytest.raises(GrowthRejected, match=message) as caught:
        build_grow(spawner, seed)

    assert type(caught.value.original) is original_type


def test_compile_builder_keeps_an_unsupported_kind_outside_growth_rejection():
    """A trait/kind compiler mismatch is not node-produced legal-kind seed data."""
    from yian.compile.expand import build_grow

    spawner = Node("unsupported")
    spawner.kind = NodeKind.CODE

    with pytest.raises(ValueError, match="cannot build graph growth") as caught:
        build_grow(spawner, {})

    assert type(caught.value) is ValueError


def test_compile_builder_preserves_helper_fault_identity_after_seed_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    """Only seed validation marks rejection; compiler construction faults stay raw."""
    from yian.compile.expand import build_grow

    call = _growth_producers()[0][0]
    expected = TypeError("compiler helper fault")

    def broken_helper(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(expand_module, "call_subgraph", broken_helper)

    with pytest.raises(TypeError) as caught:
        build_grow(call, {"x": 1})

    assert caught.value is expected
