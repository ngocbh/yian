"""A loaded flow is a reusable program; each run owns its mutable live graph."""

from __future__ import annotations

import copy
from pathlib import Path

import yian.llm_clients as llm_clients_mod
import pytest

from yian.compose.loader import load_flow
from yian.nodes.agent import AgentNode
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow


_RECURSIVE_FLOW = """
id: reusable
name: reusable
llm_config: {temperature: 0.2}
input:
  topics: list[str]
defs:
  call_child:
    nodes:
      answer:
        kind: agent
        prompt: call
        llm_config: {model: call-model}
    output: ${answer.output}
  map_child:
    input:
      topic: str
    nodes:
      inspect:
        kind: agent
        prompt: inspect
        llm_config: {model: map-model}
      emit:
        kind: code
        depends_on: [inspect]
        input:
          topic: ${input.topic}
        output: str
        code: tests.engine._compose_codefns:echo
    output: ${emit.output}
  loop_child:
    input:
      n: int
      exited: bool
    nodes:
      inspect:
        kind: agent
        prompt: inspect
        llm_config: {model: loop-model}
      step:
        kind: code
        depends_on: [inspect]
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
        code: tests.engine._compose_codefns:loop_bump
    output: ${step.output}
nodes:
  call_once:
    kind: call
    call: call_child
  map_each:
    kind: map
    call: map_child
    over: ${input.topics}
    input:
      topic: ${item}
  repeat:
    kind: loop
    call: loop_child
    input:
      n: 0
      exited: false
    while: not ${exited}
    max: 5
  finish:
    kind: agent
    depends_on: [call_once, map_each, repeat]
    prompt: finish
    llm_config: {model: finish-model}
output:
  called: ${call_once.output}
  mapped: ${map_each.output}
  repeated: ${repeat.output}
  finished: ${finish.output}
"""


_BLOCKING_CALL = """
id: blocking-call
name: blocking_call
defs:
  approval:
    nodes:
      ask:
        kind: human_input
        prompt: approve?
        output: str
    output: ${ask.output}
nodes:
  gate:
    kind: call
    call: approval
output: ${gate.output}
"""


def _graph_snapshot(flow):
    """Capture static topology recursively, including child object ownership."""
    agents = []
    children = []
    for node_id, node in flow.nodes.items():
        if isinstance(node, AgentNode):
            agents.append(
                (node_id, copy.deepcopy(node.own_llm_config), copy.deepcopy(node.llm_config))
            )
        child = getattr(node, "child", None)
        if child is not None:
            children.append((node_id, id(child), _graph_snapshot(child)))
    return {
        "nodes": tuple(flow.nodes),
        "edges": tuple(flow.edges),
        "wiring": copy.deepcopy(flow.wiring),
        "agents": tuple(agents),
        "children": tuple(children),
    }


def _load_blocking_call():
    """Load one reusable graph with a live blocking HUMAN_INPUT child."""
    return load_flow(_BLOCKING_CALL)


def test_loaded_recursive_flow_can_be_run_twice_without_graph_or_llm_leaks(monkeypatch):
    seen = []

    def fake_model_from_config(config):
        seen.append(dict(config))

        class _Model:
            def invoke(self, messages):
                class _Response:
                    content = "ok"

                return _Response()

        return _Model()

    monkeypatch.setattr(llm_clients_mod, "model_from_config", fake_model_from_config)
    loaded = load_flow(_RECURSIVE_FLOW)
    static = _graph_snapshot(loaded.compiled)

    first = run_flow(
        loaded,
        {"topics": ["alpha", "beta"]},
        llm_config={"provider": "openai", "openai_reasoning_effort": "low"},
    )
    assert first.status == "succeeded", first.failure
    assert _graph_snapshot(loaded.compiled) == static
    first_configs = list(seen)
    seen.clear()

    second = run_flow(
        loaded,
        {"topics": ["alpha", "beta"]},
        llm_config={"provider": "google", "google_thinking_level": "high"},
    )
    assert second.status == "succeeded", second.failure
    assert second.output == first.output
    assert _graph_snapshot(loaded.compiled) == static

    assert first_configs
    assert all(
        config["provider"] == "openai" and config["openai_reasoning_effort"] == "low"
        for config in first_configs
    )
    assert seen
    assert all(
        config["provider"] == "google" and config["google_thinking_level"] == "high"
        for config in seen
    )


def test_log_resume_restores_into_a_private_graph(tmp_path: Path):
    path = tmp_path / "private-recovery-graph.jsonl"
    interruption = _HostStop("interrupt before human answer")
    source = _load_blocking_call()
    source_static = _graph_snapshot(source.compiled)

    def interrupt(**request):
        raise interruption

    with pytest.raises(_HostStop) as caught:
        run_flow(source, {}, log=path, tools={"ask_user": interrupt})
    assert caught.value is interruption
    assert _graph_snapshot(source.compiled) == source_static

    fresh = _load_blocking_call()
    static = _graph_snapshot(fresh.compiled)
    resumed = resume_flow(fresh, log=path, tools={"ask_user": lambda **request: "yes"})

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == "yes"
    assert _graph_snapshot(fresh.compiled) == static
