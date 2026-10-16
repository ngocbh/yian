"""`run_flow`/`resume_flow` pass the host `context=` and per-call ids to tools in a run."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.llm_clients import LLMConfig
from yian.nodes.agent import AgentNode
from yian.run import resume_flow, run_flow
from yian.tools import TOOL_REGISTRY, ToolSet, register_tool
from yian.tools.context import ToolContext


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Keep process-global registrations isolated between tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


class _SimulatedCrash(BaseException):
    """Process-loss marker that bypasses node-failure conversion."""


class _ScriptedModel:
    """Return scripted replies and capture each request."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


class _Db:
    """A host app object that tools reach through `ctx.context`."""


class _Marked:
    """A host app object whose text forms are one unique marker."""

    MARKER = "ctx-marker-5b1f0e9c"

    def __repr__(self) -> str:
        return self.MARKER

    __str__ = __repr__


_TOOL_FLOW = """
id: tool_ctx
name: tool_ctx
nodes:
  place:
    kind: tool
    tool_id: place
output: ${place.output}
"""

_LOOP_FLOW = """
id: loop_ctx
name: loop_ctx
defs:
  turn:
    input:
      count: !int
    nodes:
      place:
        kind: tool
        tool_id: place
    output:
      count: ${input.count + 1}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      count: 0
    while: count < 3
    max: 5
output: ${loop.output}
"""

_MAP_FLOW = """
id: map_ctx
name: map_ctx
defs:
  one:
    input:
      item: !str
    nodes:
      place:
        kind: tool
        tool_id: place
    output: ${place.output}
input:
  items: !list(str)
nodes:
  m:
    kind: map
    over: ${input.items}
    call: one
    input:
      item: ${item}
output: ${m.output}
"""

_AGENT_FLOW = """
id: agent_ctx
name: agent_ctx
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [GRANT]
output: ${worker.output}
"""

_AGENT_CALL_ID = re.compile(r"^run-7f3a/worker/([0-9a-f]{32})/([01])$")


def _recorder(seen: list[ToolContext]):
    """Return a TOOL-node tool that records its context and returns its call id."""

    def place(ctx: ToolContext) -> str:
        """Place one order."""
        seen.append(ctx)
        return ctx.call_id

    return place


def _probe(seen: list[ToolContext]):
    """Return an AGENT tool that records its context."""

    def probe(ctx: ToolContext, text: str) -> str:
        """Look one thing up.

        Args:
            text: What to look up.
        """
        seen.append(ctx)
        return f"seen {text}"

    return probe


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _two_calls_then_answer(name: str) -> list[AIMessage]:
    """Two calls in one reply, one more in a second reply, then a final answer."""
    return [
        AIMessage(content="", tool_calls=[
            _tool_call(name, {"text": "a"}, "c0"),
            _tool_call(name, {"text": "b"}, "c1"),
        ]),
        AIMessage(content="", tool_calls=[_tool_call(name, {"text": "c"}, "c2")]),
        AIMessage(content="done"),
    ]


def _run_started_payload(path: Path) -> dict[str, Any]:
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["kind"] == "run_started"
    return first["payload"]


def test_tool_node_sees_the_host_context_and_its_call_id():
    seen: list[ToolContext] = []
    db = _Db()

    result = run_flow(
        load_flow(_TOOL_FLOW), {}, tools={"place": _recorder(seen)}, context=db, run_id="run-7f3a"
    )

    assert result.status == "succeeded", result.failure
    [ctx] = seen
    assert ctx.context is db
    assert ctx.run_id == "run-7f3a"
    assert ctx.node == "place"
    assert ctx.call_id == "run-7f3a/place"
    assert ctx.attempt == 1
    assert ctx.tool_name == "place"


def test_each_loop_iteration_gets_its_own_call_id():
    seen: list[ToolContext] = []

    result = run_flow(
        load_flow(_LOOP_FLOW), {}, tools={"place": _recorder(seen)}, context=_Db(), run_id="run-7f3a"
    )

    assert result.status == "succeeded", result.failure
    assert [ctx.call_id for ctx in seen] == [
        "run-7f3a/loop#0/place",
        "run-7f3a/loop#1/place",
        "run-7f3a/loop#2/place",
    ]


def test_each_map_item_gets_its_own_call_id():
    seen: list[ToolContext] = []

    result = run_flow(
        load_flow(_MAP_FLOW), {"items": ["a", "b", "c"]}, tools={"place": _recorder(seen)},
        run_id="run-7f3a",
    )

    assert result.status == "succeeded", result.failure
    assert sorted(ctx.call_id for ctx in seen) == [
        "run-7f3a/m#0/place",
        "run-7f3a/m#1/place",
        "run-7f3a/m#2/place",
    ]


def test_a_resumed_tool_node_sees_the_same_call_id_and_the_new_context(tmp_path: Path):
    log = tmp_path / "run.jsonl"
    flow = load_flow(_TOOL_FLOW)
    seen: list[ToolContext] = []

    def place(ctx: ToolContext) -> str:
        """Place one order, crashing the process on the first call."""
        seen.append(ctx)
        if len(seen) == 1:
            raise _SimulatedCrash()
        return ctx.call_id

    db, db2 = _Db(), _Db()
    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, tools={"place": place}, context=db, run_id="run-7f3a")

    resumed = resume_flow(flow, log=log, tools={"place": place}, context=db2)

    assert resumed.status == "succeeded", resumed.failure
    first, second = seen
    assert first.context is db
    assert second.context is db2
    assert second.call_id == first.call_id == "run-7f3a/place"
    assert resumed.output == "run-7f3a/place"


def test_resume_without_context_gives_none(tmp_path: Path):
    log = tmp_path / "run.jsonl"
    flow = load_flow(_TOOL_FLOW)
    seen: list[ToolContext] = []

    def place(ctx: ToolContext) -> str:
        """Place one order, crashing the process on the first call."""
        seen.append(ctx)
        if len(seen) == 1:
            raise _SimulatedCrash()
        return "ok"

    with pytest.raises(_SimulatedCrash):
        run_flow(flow, {}, log=log, tools={"place": place}, context=_Db())

    resumed = resume_flow(flow, log=log, tools={"place": place})

    assert resumed.status == "succeeded", resumed.failure
    assert seen[1].context is None


def test_the_log_never_holds_the_context(tmp_path: Path):
    flow = load_flow(_TOOL_FLOW)
    marked_log = tmp_path / "marked.jsonl"
    plain_log = tmp_path / "plain.jsonl"
    seen: list[ToolContext] = []

    marked = run_flow(flow, {}, log=marked_log, tools={"place": _recorder(seen)}, context=_Marked())
    plain = run_flow(flow, {}, log=plain_log, tools={"place": _recorder(seen)})

    assert marked.status == plain.status == "succeeded"
    assert isinstance(seen[0].context, _Marked)
    assert _Marked.MARKER not in marked_log.read_text(encoding="utf-8")
    assert _run_started_payload(marked_log).keys() == _run_started_payload(plain_log).keys()


def test_a_registered_tool_sees_the_context():
    seen: list[ToolContext] = []
    register_tool("tool_ctx_run_registered_place")(_recorder(seen))
    db = _Db()
    flow = load_flow(_TOOL_FLOW.replace("tool_id: place", "tool_id: tool_ctx_run_registered_place"))

    result = run_flow(flow, {}, context=db, run_id="run-7f3a")

    assert result.status == "succeeded", result.failure
    [ctx] = seen
    assert ctx.context is db
    assert ctx.call_id == "run-7f3a/place"


def _assert_agent_calls(seen: list[ToolContext], db: _Db) -> None:
    """Three calls: two in the first reply and one in the second."""
    assert len(seen) == 3
    for ctx in seen:
        assert ctx.context is db
        assert ctx.node == "worker"
        assert ctx.attempt == 1
        assert ctx.run_id == "run-7f3a"
    matches = [_AGENT_CALL_ID.match(ctx.call_id) for ctx in seen]
    assert all(matches), [ctx.call_id for ctx in seen]
    (first_exchange, first_index), (second_exchange, second_index), (third_exchange, _) = (
        match.groups() for match in matches
    )
    assert first_exchange == second_exchange
    assert (first_index, second_index) == ("0", "1")
    assert third_exchange != first_exchange
    assert len({ctx.call_id for ctx in seen}) == 3


def test_agent_tool_calls_see_the_context_and_per_call_ids(monkeypatch):
    seen: list[ToolContext] = []
    db = _Db()
    model = _ScriptedModel(_two_calls_then_answer("probe"))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)

    result = run_flow(
        load_flow(_AGENT_FLOW.replace("GRANT", "probe")), {},
        tools={"probe": _probe(seen)}, context=db, run_id="run-7f3a",
    )

    assert result.status == "succeeded", result.failure
    _assert_agent_calls(seen, db)


def test_agent_tool_from_a_group_sees_the_context(monkeypatch):
    seen: list[ToolContext] = []
    db = _Db()
    model = _ScriptedModel(_two_calls_then_answer("probe"))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)

    result = run_flow(
        load_flow(_AGENT_FLOW.replace("GRANT", "toolbox")), {},
        tools={"toolbox": [_probe(seen)]}, context=db, run_id="run-7f3a",
    )

    assert result.status == "succeeded", result.failure
    _assert_agent_calls(seen, db)


def test_agent_registered_tool_sees_the_context(monkeypatch):
    seen: list[ToolContext] = []
    db = _Db()
    register_tool("tool_ctx_run_registered_probe")(_probe(seen))
    model = _ScriptedModel(_two_calls_then_answer("tool_ctx_run_registered_probe"))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)

    result = run_flow(
        load_flow(_AGENT_FLOW.replace("GRANT", "tool_ctx_run_registered_probe")), {},
        context=db, run_id="run-7f3a",
    )

    assert result.status == "succeeded", result.failure
    _assert_agent_calls(seen, db)


def test_a_direct_agent_outside_a_run_gives_each_call_a_fresh_id(monkeypatch):
    seen: list[ToolContext] = []
    db = _Db()
    model = _ScriptedModel(_two_calls_then_answer("probe"))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    node = AgentNode("worker", prompt="go", tools=["probe"], llm_config=LLMConfig())

    output = node.forward({}, tools=ToolSet({"probe": _probe(seen)}, context=db))

    assert output.value == "done"
    assert [ctx.context for ctx in seen] == [db, db, db]
    assert all(re.fullmatch(r"[0-9a-f]{32}", ctx.call_id) for ctx in seen), seen
    assert len({ctx.call_id for ctx in seen}) == 3
    assert {ctx.run_id for ctx in seen} == {""}
