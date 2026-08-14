"""The `NodeResult` closed sum: Output | Route | Grow.

The FAILED path is ``raise`` -> ``EvaluationFailure`` (no failure result variant).
``Grow`` is produced by spawner nodes and interpreted by ``FlowEngine``. Routing
rides ``Route`` (tested in ``test_route.py``).
"""

from agent_composer.nodes.base import Grow, Output


def test_output_carries_value():
    assert Output(value=3).value == 3


def test_grow_is_defined():
    g = Grow(seed={"x": 1})
    assert g.seed == {"x": 1}
    assert g.prune == frozenset()
    assert not hasattr(g, "subgraph")
