"""Raw node evaluation for the append-only dispatcher.

The runtime consumes one closed worker result. Streaming remains transient and is
forwarded only through the injected callback; host-control and callback failures stay
outside the node-failure sum.
"""

import threading
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from agent_composer.events import SourceRange, SourceSpan, StreamChunk
from agent_composer.nodes.base import (
    Grow,
    Node,
    NodeKind,
    Output,
    Route,
    _HostStop,
)
from agent_composer.nodes.contracts import Parameter
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import BranchNode, FailNode, FuncNode, StreamNode, derive_wiring


def _raw(node, *, emit_stream=None, stop_event=None):
    """Evaluate one fake against its flow-owned wiring."""
    flow = SimpleNamespace(wiring=derive_wiring({node.id: node}))
    return eval_node(
        node,
        flow,
        VariablePool(),
        emit_stream=emit_stream,
        stop_event=stop_event,
    )


class _Spawner(Node):
    """Minimal trait-declared spawner for the raw Grow result arm."""

    kind = NodeKind.CODE
    is_spawner = True

    def _run_impl(self, inputs):
        return Grow(seed={"value": 1})


class _LeafGrow(Node):
    """Return Grow without the spawner trait to exercise the raw guard."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        return Grow(seed={})


class _RaisesHostStop(Node):
    """Expose the private host-control exception without a node failure wrapper."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        raise _HostStop("stop-now")


def test_raw_evaluation_returns_closed_success_route_and_grow_values():
    output = _raw(FuncNode("work", lambda _: "done"))
    routed = _raw(BranchNode("branch", "chosen"))
    grown = _raw(_Spawner("spawn"))

    assert output == Output("done")
    assert routed == Route("chosen")
    assert grown == Grow(seed={"value": 1})


def test_raw_streaming_uses_only_the_injected_transient_callback():
    chunks = []

    result = _raw(StreamNode("agent", ["a", "b"]), emit_stream=chunks.append)

    assert result == Output({"output": "ab"})
    assert [chunk.chunk for chunk in chunks] == ["a", "b"]
    assert all(isinstance(chunk, StreamChunk) for chunk in chunks)


def test_raw_failure_is_frozen_slotted_and_preserves_diagnostics():
    node = FuncNode("work", lambda inputs: inputs)
    node.params = [Parameter(name="required", required=True)]
    node._wiring_src = {}
    expected_range = SourceRange(7, 5, 7, 13)
    node.field_ranges[("input", "required")] = expected_range

    result = _raw(node)

    assert isinstance(result, EvaluationFailure)
    assert result.node_id == "work"
    assert result.error_type == "BindingError"
    assert result.locator == SourceSpan("work", "input", "required")
    assert result.range == expected_range
    assert result.traceback is not None and "BindingError" in result.traceback
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.error = "changed"


def test_raw_node_raise_and_invalid_grow_are_failures_not_observers():
    raised = _raw(FailNode("raise", "boom"))
    invalid_grow = _raw(_LeafGrow("leaf"))

    assert isinstance(raised, EvaluationFailure)
    assert (raised.error, raised.error_type) == ("boom", "RuntimeError")
    assert isinstance(invalid_grow, EvaluationFailure)
    assert "not a spawner" in invalid_grow.error


def test_raw_stream_callback_failure_propagates_unchanged():
    expected = RuntimeError("callback failed")

    def reject(_chunk):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        _raw(StreamNode("agent", ["a"]), emit_stream=reject)

    assert caught.value is expected


def test_raw_stream_callback_failure_closes_the_node_generator_before_returning():
    closed = []

    class ResourceStreamNode(Node):
        """Streaming fake whose generator records deterministic cleanup."""

        kind = NodeKind.CODE

        def _run_impl(self, inputs):
            try:
                yield StreamChunk(self.id, "output", "first")
                yield StreamChunk(self.id, "output", "second")
                return Output("done")
            finally:
                closed.append(self.id)

    expected = RuntimeError("callback failed")

    def reject(_chunk):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        _raw(ResourceStreamNode("resource"), emit_stream=reject)

    assert caught.value is expected
    assert closed == ["resource"]


def test_raw_host_stop_propagates_from_node_and_preset_event():
    with pytest.raises(_HostStop) as raised:
        _raw(_RaisesHostStop("work"))
    assert str(raised.value) == "stop-now"

    stop_event = threading.Event()
    stop_event.set()
    with pytest.raises(_HostStop) as preset:
        _raw(FuncNode("never", lambda _: "unexpected"), stop_event=stop_event)
    assert preset.value.primary is False
