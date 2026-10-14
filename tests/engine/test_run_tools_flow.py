"""`run_flow` and `resume_flow` give each run its own tools and `ask_user` handler."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow
from yian.tools import TOOL_REGISTRY, register_tool


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Keep process-global registrations isolated between tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


_GREETING_FLOW = """
id: greeting
name: greeting
nodes:
  who:
    kind: tool
    tool_id: whoami
  ask:
    kind: human_input
    input: {name: "${who.output}"}
    prompt: "Hello ${name}?"
output: ${ask.output}
"""

_GLOBAL_TOOL_FLOW = """
id: global_tool
name: global_tool
nodes:
  call:
    kind: tool
    tool_id: run_flow_global_probe
output: ${call.output}
"""

_LOOP_ASK_FLOW = """
id: loop_ask
name: loop_ask
defs:
  turn:
    input:
      count: int
      last: str
    nodes:
      ask:
        kind: human_input
        prompt: "next turn"
        output: str
    output:
      count: ${input.count + 1}
      last: ${ask.output}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      count: 0
      last: ""
    while: count < 2
    max: 5
output: ${loop.output}
"""

_MAP_ASK_FLOW = """
id: map_ask
name: map_ask
defs:
  one:
    input:
      item: str
    nodes:
      ask:
        kind: human_input
        prompt: "item?"
        output: str
    output: ${ask.output}
input:
  items: list[str]
nodes:
  m:
    kind: map
    over: ${input.items}
    call: one
    input:
      item: ${item}
output: ${m.output}
"""

_CALL_ASK_FLOW = """
id: call_ask
name: call_ask
defs:
  one:
    nodes:
      ask:
        kind: human_input
        prompt: "q"
        output: str
    output: ${ask.output}
nodes:
  c1:
    kind: call
    call: one
  c2:
    kind: call
    call: one
    runs_after: [c1]
output: ${c2.output}
"""


def _answer_with_question(**request: Any) -> str:
    """Echo the rendered question back as the answer."""
    return f"answer to {request['question']}"


def test_run_flow_uses_per_run_tools_and_ask_user_handler():
    result = run_flow(
        load_flow(_GREETING_FLOW),
        {},
        tools={"whoami": lambda: "alice", "ask_user": _answer_with_question},
    )

    assert result.status == "succeeded", result.failure
    assert result.output == "answer to Hello alice?"
    assert "whoami" not in TOOL_REGISTRY and "ask_user" not in TOOL_REGISTRY


def test_a_tools_mapping_hides_the_global_registry():
    @register_tool("run_flow_global_probe")
    def probe() -> str:
        """Global probe."""
        return "global"

    flow = load_flow(_GLOBAL_TOOL_FLOW)

    assert run_flow(flow, {}).output == "global"
    hidden = run_flow(flow, {}, tools={})
    assert hidden.status == "failed"
    assert "unknown tool id 'run_flow_global_probe'" in hidden.failure.message


def test_concurrent_runs_never_see_each_others_tools_or_handler():
    both_asking = threading.Barrier(2, timeout=5)
    results: dict[str, Any] = {}

    def run_as(name: str) -> None:
        def ask_user(**request: Any) -> str:
            both_asking.wait()
            return f"{name} answers {request['question']}"

        results[name] = run_flow(
            load_flow(_GREETING_FLOW),
            {},
            tools={"whoami": lambda: name, "ask_user": ask_user},
        )

    threads = [threading.Thread(target=run_as, args=(name,)) for name in ("ann", "bob")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert results["ann"].output == "ann answers Hello ann?"
    assert results["bob"].output == "bob answers Hello bob?"


def test_resume_flow_asks_through_the_handler_passed_to_resume(tmp_path: Path):
    path = tmp_path / "resume-tools.jsonl"
    flow = load_flow(_GREETING_FLOW)

    def interrupt(**request: Any) -> str:
        raise _HostStop()

    with pytest.raises(_HostStop):
        run_flow(flow, {}, log=path, tools={"whoami": lambda: "carol", "ask_user": interrupt})

    resumed = resume_flow(
        flow,
        log=path,
        tools={"whoami": lambda: "unused", "ask_user": _answer_with_question},
    )

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == "answer to Hello carol?"


def test_each_loop_occurrence_asks_with_its_own_node_id():
    node_ids: list[str] = []

    def ask_user(**request: Any) -> str:
        node_ids.append(request["node_id"])
        return "ok"

    result = run_flow(load_flow(_LOOP_ASK_FLOW), {}, tools={"ask_user": ask_user})

    assert result.status == "succeeded", result.failure
    assert node_ids == ["loop#0/ask", "loop#1/ask"]


@pytest.mark.parametrize(
    ("source", "inputs", "expected"),
    [
        (_MAP_ASK_FLOW, {"items": ["a", "b"]}, ["m#0/ask", "m#1/ask"]),
        (_CALL_ASK_FLOW, {}, ["c1/ask", "c2/ask"]),
    ],
    ids=["map", "call"],
)
def test_each_map_item_and_call_asks_with_its_own_node_id(
    source: str, inputs: dict[str, Any], expected: list[str]
):
    node_ids: list[str] = []

    def ask_user(**request: Any) -> str:
        node_ids.append(request["node_id"])
        return "ok"

    result = run_flow(load_flow(source), inputs, tools={"ask_user": ask_user})

    assert result.status == "succeeded", result.failure
    assert sorted(node_ids) == expected


@pytest.mark.parametrize(
    "tools",
    [["whoami"], {"ask_user": "not callable"}, {"": lambda: "x"}],
)
def test_bad_tools_fail_before_the_log_is_created(tmp_path: Path, tools: Any):
    path = tmp_path / "never.jsonl"

    with pytest.raises(TypeError):
        run_flow(load_flow(_GREETING_FLOW), {}, log=path, tools=tools)
    assert not path.exists()


def test_human_input_without_a_run_handler_fails_the_node():
    result = run_flow(load_flow(_LOOP_ASK_FLOW), {})

    assert result.status == "failed"
    assert "no ask_user handler for this run" in result.failure.message


_GROUPS_FLOW = """
id: groups
name: groups
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [toolbox, workspace, company]
output: ${agent.output}
"""


class _FirstToolChat:
    """Call the first bound tool once, then answer with its result and the bound names."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def bind_tools(self, tools):
        self.names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(content=f"{last.content} via {self.names}")
        call = {"name": self.names[0], "args": {"text": "x"}, "id": "c1", "type": "tool_call"}
        return AIMessage(content="", tool_calls=[call])


@pytest.fixture
def first_tool_chat(monkeypatch):
    """Stand in for the provider with a fresh `_FirstToolChat` per AGENT call."""
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: _FirstToolChat())


def _tool(name: str, action=None) -> StructuredTool:
    """Build one tool named `name`; by default it returns `name:text`."""

    def run(text: str) -> str:
        return action(text) if action is not None else f"{name}:{text}"

    return StructuredTool.from_function(run, name=name, description=name)


def _groups(toolbox: list[Any]) -> dict[str, Any]:
    """Fill the three groups the flow grants, with `toolbox` first."""
    return {"toolbox": toolbox, "workspace": [_tool("ws")], "company": [_tool("co")]}


def test_agent_sees_the_union_of_three_host_filled_groups(first_tool_chat):
    result = run_flow(load_flow(_GROUPS_FLOW), {}, tools=_groups([_tool("box")]))

    assert result.status == "succeeded", result.failure
    assert result.output == "box:x via ['box', 'ws', 'co']"


def test_concurrent_runs_with_different_toolboxes_never_cross(first_tool_chat):
    both_calling = threading.Barrier(2, timeout=5)
    results: dict[str, Any] = {}
    errors: list[BaseException] = []

    def run_as(name: str) -> None:
        def lookup(text: str) -> str:
            both_calling.wait()
            return f"{name} looked up {text}"

        try:
            results[name] = run_flow(
                load_flow(_GROUPS_FLOW), {}, tools=_groups([_tool("lookup", lookup)])
            )
        except BaseException as exc:  # re-raised on the test thread below
            errors.append(exc)

    threads = [threading.Thread(target=run_as, args=(name,)) for name in ("ann", "bob")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    for error in errors:
        raise error

    assert results["ann"].output == "ann looked up x via ['lookup', 'ws', 'co']"
    assert results["bob"].output == "bob looked up x via ['lookup', 'ws', 'co']"


@pytest.mark.parametrize(
    ("tools", "error"),
    [
        (
            {"toolbox": [], "workspace": []},
            "AGENT node 'agent' grants unknown tool or group 'company'; "
            "tool groups in this run: ['toolbox', 'workspace']",
        ),
        (_groups([_tool("ws")]), "tool 'ws' is in both groups 'toolbox' and 'workspace'"),
        (_groups([_tool("ask_user")]), "group 'toolbox' has a tool named 'ask_user'"),
    ],
    ids=["missing-group", "name-clash", "group-ask-user"],
)
def test_bad_groups_fail_at_run_start(tmp_path: Path, tools: dict[str, Any], error: str):
    path = tmp_path / "never.jsonl"

    with pytest.raises(ValueError, match=re.escape(error)):
        run_flow(load_flow(_GROUPS_FLOW), {}, log=path, tools=tools)
    assert not path.exists()


def test_resume_uses_the_group_contents_passed_to_resume(tmp_path: Path, first_tool_chat):
    path = tmp_path / "resume-groups.jsonl"
    flow = load_flow(_GROUPS_FLOW)

    def interrupt(text: str) -> str:
        raise _HostStop()

    with pytest.raises(_HostStop):
        run_flow(flow, {}, log=path, tools=_groups([_tool("old", interrupt)]))

    with pytest.raises(ValueError, match="unknown tool or group 'toolbox'"):
        resume_flow(flow, log=path, tools={"workspace": [], "company": []})
    resumed = resume_flow(flow, log=path, tools=_groups([_tool("new")]))

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == "new:x via ['new', 'ws', 'co']"


_NESTED_GROUPS_FLOW = """
id: nested_groups
name: nested_groups
llm_config: {provider: openai, model: test-model}
defs:
  inner:
    nodes:
      agent:
        kind: agent
        prompt: go
        tools: [toolbox]
    output: ${agent.output}
nodes:
  outer:
    kind: call
    call: inner
output: ${outer.output}
"""


_MAP_GROUPS_FLOW = """
id: map_groups
name: map_groups
llm_config: {provider: openai, model: test-model}
defs:
  inner:
    input:
      item: str
    nodes:
      agent:
        kind: agent
        prompt: go
        tools: [toolbox]
    output: ${agent.output}
input:
  items: list[str]
nodes:
  each:
    kind: map
    over: ${input.items}
    call: inner
    input:
      item: ${item}
output: ${each.output}
"""

_LOOP_GROUPS_FLOW = """
id: loop_groups
name: loop_groups
llm_config: {provider: openai, model: test-model}
defs:
  turn:
    input:
      count: int
      last: str
    nodes:
      agent:
        kind: agent
        prompt: go
        tools: [toolbox]
    output:
      count: ${input.count + 1}
      last: ${agent.output}
nodes:
  again:
    kind: loop
    call: turn
    input:
      count: 0
      last: ""
    while: count < 1
    max: 3
output: ${again.output}
"""


@pytest.mark.parametrize(
    ("source", "inputs", "node_path"),
    [
        (_NESTED_GROUPS_FLOW, {}, "outer/agent"),
        (_MAP_GROUPS_FLOW, {"items": []}, "each/agent"),
        (_LOOP_GROUPS_FLOW, {}, "again/agent"),
    ],
    ids=["call", "map", "loop"],
)
def test_a_missing_group_in_a_child_flow_fails_at_run_start(
    tmp_path: Path, source: str, inputs: dict[str, Any], node_path: str
):
    path = tmp_path / "never.jsonl"

    with pytest.raises(
        ValueError, match=f"AGENT node '{node_path}' grants unknown tool or group 'toolbox'"
    ):
        run_flow(load_flow(source), inputs, log=path, tools={})
    assert not path.exists()


_PLAIN_GRANT_FLOW = """
id: plain_grant
name: plain_grant
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [lookup, ask_user]
output: ${agent.output}
"""


@pytest.mark.parametrize(
    ("tools", "error"),
    [
        (
            {"ask_user": _answer_with_question},
            "AGENT node 'agent' grants unknown tool or group 'lookup'",
        ),
        (
            {"lookup": _tool("lookup")},
            "AGENT node 'agent' grants 'ask_user': no ask_user handler for this run",
        ),
    ],
    ids=["unknown-plain-tool", "ask-user-without-handler"],
)
def test_an_unresolvable_plain_grant_fails_at_run_start(
    tmp_path: Path, tools: dict[str, Any], error: str
):
    path = tmp_path / "never.jsonl"

    with pytest.raises(ValueError, match=re.escape(error)):
        run_flow(load_flow(_PLAIN_GRANT_FLOW), {}, log=path, tools=tools)
    assert not path.exists()
