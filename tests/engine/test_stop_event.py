"""Host-stop capability behavior at the raw node boundary."""

from __future__ import annotations

import inspect
import threading

import pytest

from agent_composer.nodes.base import Node, NodeKind, Output, _HostStop
from agent_composer.run import run_flow
from agent_composer.runtime.eval_node import eval_node
from agent_composer.typesys.pool import VariablePool


class _CapNode(Node):
    """Record the capabilities supplied to one direct evaluation."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, *, opted_in: bool) -> None:
        super().__init__(node_id)
        self.needs_stop_event = opted_in
        self.caps: dict = {}

    def forward(self, inputs, **caps):
        self.caps = caps
        return Output("done")


class _NeverRunsNode(Node):
    """Expose whether a pre-set stop event allowed the node body to start."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.ran = False

    def forward(self, inputs):
        self.ran = True
        return Output("unexpected")


class _RaisesHostStop(Node):
    """Raise one exact host-control exception from inside node execution."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, exception: _HostStop) -> None:
        super().__init__(node_id)
        self.exception = exception

    def forward(self, inputs):
        raise self.exception


def test_stop_cap_is_exact_event_and_only_reaches_opted_in_node():
    stop_event = threading.Event()
    opted_in = _CapNode("aware", opted_in=True)
    ordinary = _CapNode("ordinary", opted_in=False)

    assert eval_node(
        opted_in,
        None,
        VariablePool(),
        stop_event=stop_event,
    ).outcome == Output("done")
    assert eval_node(
        ordinary,
        None,
        VariablePool(),
        stop_event=stop_event,
    ).outcome == Output("done")

    assert opted_in.caps == {"stop_event": stop_event}
    assert opted_in.caps["stop_event"] is stop_event
    assert ordinary.caps == {}


def test_direct_opted_in_eval_gets_private_unset_event_when_host_omits_one():
    node = _CapNode("aware", opted_in=True)

    assert eval_node(node, None, VariablePool()).outcome == Output("done")

    assert set(node.caps) == {"stop_event"}
    assert isinstance(node.caps["stop_event"], threading.Event)
    assert not node.caps["stop_event"].is_set()


def test_preset_stop_prevents_node_execution_and_propagates_host_control():
    stop_event = threading.Event()
    stop_event.set()
    node = _NeverRunsNode("never")

    with pytest.raises(_HostStop) as caught:
        eval_node(node, None, VariablePool(), stop_event=stop_event)

    assert caught.value.primary is False
    assert not node.ran


def test_node_host_stop_bypasses_evaluation_failure_with_identity():
    expected = _HostStop("terminal EOF")

    with pytest.raises(_HostStop) as caught:
        eval_node(
            _RaisesHostStop("work", expected),
            None,
            VariablePool(),
        )

    assert caught.value is expected


def test_public_run_flow_signature_has_no_stop_event():
    assert "stop_event" not in inspect.signature(run_flow).parameters
