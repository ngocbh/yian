"""The engine forwards in-loop observers to `on_event` between start and outcome.

Serial and pooled execution both deliver them in node order, drop them from the
yielded observer stream, and supply no sink when the host set no callback.
"""

import pytest

from yian.events import NodeStarted, NodeSucceeded, StreamChunk
from yian.compile.model import END_ID, START_ID
from yian.nodes.base import Node, NodeKind, Output
from tests.engine._graph_builder import _engine, _graph


class _Live(Node):
    """Publish one chunk and two in-loop observers through whichever sinks exist."""

    kind = NodeKind.CODE
    needs_live_events = True

    def forward(self, inputs, **caps):
        self.seen = sorted(key for key in caps if key.startswith("emit_"))
        if "emit_stream" in caps:
            caps["emit_stream"](StreamChunk(self.id, "text", "hi"))
        if "emit_event" in caps:
            caps["emit_event"](("in-loop", 1))
            caps["emit_event"](("in-loop", 2))
        return Output(value="done")


@pytest.mark.parametrize("num_workers", [0, 2])
def test_in_loop_observers_reach_on_event_between_start_and_outcome(num_workers):
    node = _Live("work")
    timeline = []
    engine = _engine(
        _graph([node], [(START_ID, "work"), ("work", END_ID)]),
        num_workers=num_workers,
        on_event=timeline.append,
        on_stream=timeline.append,
    )
    yielded = list(engine.run())
    work = [item for item in timeline
            if getattr(item, "node_id", None) == "work" or isinstance(item, tuple)]
    assert isinstance(work[0], NodeStarted) and isinstance(work[-1], NodeSucceeded)
    assert work[1:-1] == [StreamChunk("work", "text", "hi"), ("in-loop", 1), ("in-loop", 2)]
    assert not any(isinstance(item, tuple) for item in yielded)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_no_callbacks_means_no_sinks(num_workers):
    node = _Live("work")
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), num_workers=num_workers)
    list(engine.run())
    assert node.seen == []


@pytest.mark.parametrize("num_workers", [0, 2])
def test_event_callback_alone_supplies_only_the_event_sink(num_workers):
    node = _Live("work")
    engine = _engine(
        _graph([node], [(START_ID, "work"), ("work", END_ID)]),
        num_workers=num_workers, on_event=lambda _event: None,
    )
    list(engine.run())
    assert node.seen == ["emit_event"]
