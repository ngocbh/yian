"""`run_flow` and `resume_flow` give each run its own tools and `ask_user` handler."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from agent_composer.compose import load_flow
from agent_composer.nodes.base import _HostStop
from agent_composer.run import resume_flow, run_flow
from agent_composer.tools import TOOL_REGISTRY, register_tool


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
    while: ${count} < 2
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
