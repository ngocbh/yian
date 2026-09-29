"""End to end: `run_flow` streams AGENT reply text to `on_stream`.

Chunks reach `on_stream` strictly between the node's `NodeStarted` and
`NodeSucceeded`, inside their own model call's observers, in serial and pooled
execution. They never reach `RunResult.events` or the run log.
"""

import threading

import pytest

import agent_composer.llm_clients as llm_clients_mod
import agent_composer.tools as tools_mod
from agent_composer.compose.loader import load_flow
from agent_composer.events import (
    IN_LOOP_EVENTS,
    ModelCallFinished,
    ModelCallStarted,
    NodeStarted,
    NodeSucceeded,
    StreamChunk,
    ToolCallFinished,
)
from agent_composer.run import run_flow
from agent_composer.runtime.log import JsonlRunLog
from tests.engine._stream_fakes import ScriptedStreamChat, reply
from tests.engine.test_live_events_run import FLOW, PAIR, _script, _Tool


@pytest.fixture
def chat(monkeypatch):
    model = ScriptedStreamChat(script=_script(), mode=[])
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: model)
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup", _Tool())
    return model


@pytest.mark.parametrize("num_workers", [0, 2])
def test_tool_calling_agent_streams_inside_its_model_calls(chat, tmp_path, num_workers):
    timeline = []
    log = tmp_path / "run.jsonl"
    result = run_flow(
        load_flow(FLOW), {"topic": "acme"}, log=log, num_workers=num_workers,
        on_event=timeline.append, on_stream=timeline.append,
    )
    assert result.status == "succeeded" and result.output == "acme looks fine"
    assert chat.mode == ["stream", "stream"]

    writer = [item for item in timeline if getattr(item, "node_id", None) == "writer"]
    assert isinstance(writer[0], NodeStarted) and isinstance(writer[-1], NodeSucceeded)
    chunks = [item for item in writer if isinstance(item, StreamChunk)]
    first, second = (e.exchange_id for e in writer if isinstance(e, ModelCallStarted))
    assert "".join(c.chunk for c in chunks if c.exchange_id == first) == "checking"
    assert "".join(c.chunk for c in chunks if c.exchange_id == second) == "acme looks fine"
    assert [(c.exchange_id, c.final) for c in chunks if c.final] == [(first, True), (second, True)]
    # Each call's chunks arrive between its started and finished observers.
    for exchange in (first, second):
        own = [item for item in writer if getattr(item, "exchange_id", None) == exchange]
        assert isinstance(own[0], ModelCallStarted)
        assert isinstance(own[-1], (ModelCallFinished, ToolCallFinished))
        finished = next(i for i, item in enumerate(own) if isinstance(item, ModelCallFinished))
        assert all(i < finished for i, item in enumerate(own) if isinstance(item, StreamChunk))

    finished = [e for e in writer if isinstance(e, ModelCallFinished)]
    assert sum(e.usage.total_tokens for e in finished) == 42
    assert finished[0].usage.cache_read_tokens == 6

    durable = [e for e in timeline if not isinstance(e, (StreamChunk, *IN_LOOP_EVENTS))]
    assert result.events == durable
    with JsonlRunLog.resume(log) as reopened:
        names = {type(envelope.record).__name__ for envelope in reopened}
    assert "StreamChunk" not in names


def test_on_event_alone_still_invokes(chat):
    events = []
    result = run_flow(load_flow(FLOW), {"topic": "acme"}, on_event=events.append)
    assert result.output == "acme looks fine"
    assert chat.mode == ["invoke", "invoke"]
    assert sum(isinstance(e, ModelCallFinished) for e in events) == 2


def test_parallel_agents_stream_their_own_chunks_in_pooled_runs(monkeypatch):
    model = ScriptedStreamChat(
        script=[
            reply("looking", calls=[("lookup", {"topic": "acme"}, "call-1")]),
            reply("looking", calls=[("lookup", {"topic": "acme"}, "call-2")]),
            reply("all done"),
            reply("all done"),
        ],
        mode=[],
    )
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: model)
    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY, "lookup", _Tool(threading.Barrier(2, timeout=10))
    )
    timeline = []
    result = run_flow(
        load_flow(PAIR), {"topic": "acme"}, num_workers=2,
        on_event=timeline.append, on_stream=timeline.append,
    )
    assert result.output == {"left": "all done", "right": "all done"}
    assert model.mode == ["stream"] * 4

    exchanges = {}
    for node_id in ("left", "right"):
        own = [i for i, item in enumerate(timeline) if getattr(item, "node_id", None) == node_id]
        started = next(i for i in own if isinstance(timeline[i], NodeStarted))
        done = next(i for i in own if isinstance(timeline[i], NodeSucceeded))
        chunks = [i for i in own if isinstance(timeline[i], StreamChunk)]
        assert started < chunks[0] and chunks[-1] < done
        assert "".join(timeline[i].chunk for i in chunks) == "lookingall done"
        exchanges[node_id] = {timeline[i].exchange_id for i in chunks}
    assert len(exchanges["left"]) == len(exchanges["right"]) == 2
    assert not exchanges["left"] & exchanges["right"]
