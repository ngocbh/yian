"""`commit_as` lives on the three coordinated carriers — the field itself.

P2 replaces the engine's `alias`/`loop_alias` redirect dicts with a `commit_as` field
that travels on the node/Output/event. This test pins the field's presence and defaults
on all three homes (the engine wiring is exercised by the expansion/loop/durable tests).
"""

from agent_composer.events import NodeSucceeded
from agent_composer.nodes.base import Node, NodeKind, Output


class _BareNode(Node):
    """Minimal Node subclass to check the base `commit_as` default + assignment."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs, **caps):
        return Output(None)


def test_output_commit_as_default_and_set():
    assert Output(value=1).commit_as is None
    assert Output(value=1, commit_as="s").commit_as == "s"


def test_node_succeeded_commit_as_default_and_set():
    assert NodeSucceeded("n").commit_as is None
    assert NodeSucceeded("n", commit_as="s").commit_as == "s"


def test_node_commit_as_default_and_assignment():
    node = _BareNode("n")
    assert node.commit_as is None
    node.commit_as = "s"
    assert node.commit_as == "s"
