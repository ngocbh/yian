"""The evaluator supplies in-loop observer sinks only to opted-in nodes.

`needs_live_events` gates `caps['emit_stream']` and `caps['emit_event']`; each is
present only when the host supplied that sink, and a failing host sink escapes
as the original host exception rather than a node failure.
"""

from types import SimpleNamespace

import pytest

from agent_composer.events import StreamChunk
from agent_composer.nodes.base import Node, NodeKind, Output
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import derive_wiring


class _Probe(Node):
    """Record received capabilities and publish through whichever sinks exist."""

    kind = NodeKind.CODE

    def forward(self, inputs, **caps):
        self.seen = sorted(caps)
        if "emit_stream" in caps:
            caps["emit_stream"](StreamChunk(self.id, "text", "hi"))
        if "emit_event" in caps:
            caps["emit_event"]("in-loop")
        return Output(value="done")


class _LiveProbe(_Probe):
    needs_live_events = True


def _evaluate(node, **sinks):
    flow = SimpleNamespace(wiring=derive_wiring({node.id: node}))
    return eval_node(node, flow, VariablePool(), **sinks).outcome


def test_node_without_trait_receives_no_sinks():
    node = _Probe("probe")
    chunks, events = [], []
    assert _evaluate(node, emit_stream=chunks.append, emit_event=events.append).value == "done"
    assert node.seen == [] and chunks == [] and events == []


@pytest.mark.parametrize(
    ("stream", "event", "expected"),
    [
        (True, True, ["emit_event", "emit_stream"]),
        (True, False, ["emit_stream"]),
        (False, True, ["emit_event"]),
        (False, False, []),
    ],
)
def test_opted_in_node_receives_only_supplied_sinks(stream, event, expected):
    node = _LiveProbe("probe")
    chunks, events = [], []
    _evaluate(
        node,
        emit_stream=chunks.append if stream else None,
        emit_event=events.append if event else None,
    )
    assert node.seen == expected
    assert chunks == ([StreamChunk("probe", "text", "hi")] if stream else [])
    assert events == (["in-loop"] if event else [])


@pytest.mark.parametrize("sink", ["emit_stream", "emit_event"])
def test_failing_host_sink_propagates_original_exception(sink):
    class HostBroken(Exception):
        pass

    def broken(_item):
        raise HostBroken("host")

    with pytest.raises(HostBroken) as raised:
        _evaluate(_LiveProbe("probe"), **{sink: broken})
    assert not isinstance(raised.value, EvaluationFailure)


def test_emit_event_must_be_callable():
    with pytest.raises(TypeError, match="emit_event must be callable"):
        _evaluate(_LiveProbe("probe"), emit_event="not callable")
