"""Seed-first growth: nodes describe expansion; compile constructs topology."""

from agent_composer.compile.model import Flow
from agent_composer.nodes.base import Grow
from agent_composer.nodes.binding import ParamDecl
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.map import MapNode

from tests.engine.test_expand import _child_flow


def _growth_producers():
    """Return one live seed-producing spawner of every legal grow kind."""
    child = _child_flow()

    call = CallNode("call", flow_id="child", child=child)
    call.params = [ParamDecl(name="x")]
    call.post_asserts = ["output == x"]
    call.assert_texts = {"output == x": "output == x"}
    call_grow = call.run({"x": 1})

    map_node = MapNode("map", flow_id="child", child=child)
    map_grow = map_node.run(
        {"over": [1, 2]}, bind_item=lambda item: {"x": item}
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
        assert left.post_asserts == right.post_asserts
        assert left.assert_texts == right.assert_texts


def test_call_map_and_loop_return_seed_only():
    producers = _growth_producers()

    assert all(isinstance(grow, Grow) for _, grow in producers)
    assert all(not hasattr(grow, "subgraph") for _, grow in producers)
    assert producers[0][1].seed == {"x": 1}
    assert producers[1][1].seed == [{"x": 1}, {"x": 2}]
    assert producers[2][1].seed == ({"x": 1}, 0)


def test_compile_builder_reconstructs_every_growth_kind():
    from agent_composer.compile.expand import (
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
        assert_texts=call.assert_texts,
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
