"""Seed-first growth: nodes describe expansion; compile constructs topology."""

import pytest

import agent_composer.compile.expand as expand_module
from agent_composer.compile.model import Flow, GrowthRejected
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Grow, Node, NodeKind
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.contracts import Assertion, Parameter
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.map import MapNode

from tests.engine.test_expand import _child_flow


def _growth_producers():
    """Return one live seed-producing spawner of every legal grow kind."""
    child = _child_flow()

    call = CallNode("call", flow_id="child", child=child)
    call.params = [Parameter(name="x")]
    call.post_asserts = [Assertion("output == x", "output == x")]
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
        assert left.post_assert_ranges == right.post_assert_ranges


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
    loop.origin = authored_origin

    clone = loop.respawn(2)

    assert clone.origin is authored_origin
    assert clone.origin_id == "runtime-loop"


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
        post_assert_ranges=call.post_assert_ranges,
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
    from agent_composer.compile.expand import build_grow

    spawner = _growth_producers()[producer_index][0]

    with pytest.raises(GrowthRejected, match=message) as caught:
        build_grow(spawner, seed)

    assert type(caught.value.original) is original_type


def test_compile_builder_keeps_an_unsupported_kind_outside_growth_rejection():
    """A trait/kind compiler mismatch is not node-produced legal-kind seed data."""
    from agent_composer.compile.expand import build_grow

    spawner = Node("unsupported")
    spawner.kind = NodeKind.CODE

    with pytest.raises(ValueError, match="cannot build graph growth") as caught:
        build_grow(spawner, {})

    assert type(caught.value) is ValueError


def test_compile_builder_preserves_helper_fault_identity_after_seed_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    """Only seed validation marks rejection; compiler construction faults stay raw."""
    from agent_composer.compile.expand import build_grow

    call = _growth_producers()[0][0]
    expected = TypeError("compiler helper fault")

    def broken_helper(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(expand_module, "call_subgraph", broken_helper)

    with pytest.raises(TypeError) as caught:
        build_grow(call, {"x": 1})

    assert caught.value is expected
