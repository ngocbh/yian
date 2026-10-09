"""``eval_node`` accepts only trait-declared spawner ``Grow`` results.

A node whose ``is_spawner`` trait is true may return ``Grow`` for ``FlowEngine`` to
interpret. A non-spawner returning ``Grow`` becomes ``EvaluationFailure``.
"""

from yian.nodes.base import Grow, Node, NodeKind
from yian.runtime.eval_node import EvaluationFailure
from tests.engine._fakes import drive


def _grow() -> Grow:
    return Grow(seed={})


class _FakeSpawner(Node):
    """A minimal spawner (is_spawner=True) whose run returns a Grow — exercises the new
    `Grow`-as-spawner-result path independent of any real spawner kind."""

    kind = NodeKind.CODE
    is_spawner = True

    def forward(self, inputs, **caps):
        return _grow()


class _FakeLeaf(Node):
    """A non-spawner (is_spawner=False) whose run returns a Grow — must fail (only spawners grow)."""

    kind = NodeKind.CODE
    is_spawner = False

    def forward(self, inputs, **caps):
        return _grow()


def test_spawner_grow_remains_a_raw_grow_result():
    result = drive(_FakeSpawner("s"))

    assert result == _grow()


def test_leaf_grow_is_an_evaluation_failure():
    result = drive(_FakeLeaf("l"))

    assert isinstance(result, EvaluationFailure)
    assert "not a spawner" in result.failure.message
