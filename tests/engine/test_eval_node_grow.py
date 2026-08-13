"""P3 0b: eval_node accepts a spawner's `Grow` result -> a single NodeExpanded (payload IS the Grow).

A node whose `is_spawner` ClassVar is True may return a `Grow` to grow the live graph; eval_node
funnels it to `NodeExpanded(node.id, grow)`. A non-spawner (`is_spawner=False`) returning a `Grow`
is a clear error -> `NodeFailed` (only spawner kinds may grow the graph)."""

from agent_composer.events import NodeExpanded, NodeFailed, NodeStarted
from agent_composer.nodes.base import Grow, Node, NodeKind
from tests.engine._fakes import drive


def _grow() -> Grow:
    return Grow(seed={})


class _FakeSpawner(Node):
    """A minimal spawner (is_spawner=True) whose run returns a Grow — exercises the new
    `Grow`-as-spawner-result path independent of any real spawner kind."""

    kind = NodeKind.CODE
    is_spawner = True

    def _run_impl(self, inputs, **caps):
        return _grow()


class _FakeLeaf(Node):
    """A non-spawner (is_spawner=False) whose run returns a Grow — must fail (only spawners grow)."""

    kind = NodeKind.CODE
    is_spawner = False

    def _run_impl(self, inputs, **caps):
        return _grow()


def test_spawner_grow_becomes_node_expanded_carrying_the_grow():
    evs = list(drive(_FakeSpawner("s")))
    assert isinstance(evs[0], NodeStarted)
    assert isinstance(evs[-1], NodeExpanded)
    assert isinstance(evs[-1].grow, Grow)
    # No terminal NodeSucceeded/NodeFailed after the expansion.
    assert not any(isinstance(e, NodeFailed) for e in evs)


def test_leaf_grow_is_node_failed():
    evs = list(drive(_FakeLeaf("l")))
    assert isinstance(evs[0], NodeStarted)
    assert isinstance(evs[-1], NodeFailed)
    assert not any(isinstance(e, NodeExpanded) for e in evs)
