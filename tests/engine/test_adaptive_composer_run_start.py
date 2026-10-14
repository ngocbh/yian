"""A run-start LLM config error for an adaptive gate's composer names the gate's `adaptive_questions.llm_config`."""

from __future__ import annotations

import copy

import pytest

import yian.run
from yian.compile.llm_cascade import apply_effective_llm_map, resolve_llm_cascade, static_agent_labels
from yian.compile.manifest import flow_fingerprint
from yian.compose import load_flow
from yian.run import run_flow

_LLM_SITE = "node 'ask' adaptive_questions.llm_config"

_GATE_FLOW = """\
id: f
name: f
input: {{proj: !str }}
nodes:
  ask:
    kind: human_input
    input: {{proj: '${{input.proj}}'}}
    adaptive_questions:
      prompt: "Q ${{proj}}"
      llm_config: {llm_config}
output: ${{ask.output}}
"""

_CHILD_FLOW = """\
id: p
name: p
input: {proj: !str }
defs:
  child:
    input: {proj: !str }
    nodes:
      ask:
        kind: human_input
        input: {proj: '${input.proj}'}
        adaptive_questions:
          prompt: "Q ${proj}"
          llm_config: {provider: azure, model: d}
    output: ${ask.output}
nodes:
  sub:
    kind: call
    call: child
    input: {proj: '${input.proj}'}
output: ${sub.output}
"""

_AGENT_FLOW = """\
id: f
name: f
nodes:
  w: {{kind: agent, prompt: hi, llm_config: {llm_config}}}
output: ${{w.output}}
"""

_AZURE = "{provider: azure, model: d}"
_OPENAI = "{provider: openai, model: m}"


@pytest.fixture(autouse=True)
def _no_azure_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)


def _run_start_error(text: str, inputs: dict[str, str]) -> str:
    with pytest.raises(ValueError) as caught:
        run_flow(load_flow(text), inputs)
    return str(caught.value)


def test_run_start_error_names_the_gate():
    message = _run_start_error(_GATE_FLOW.format(llm_config=_AZURE), {"proj": "x"})
    assert message.startswith(f"{_LLM_SITE}: provider azure needs llm_config.base_url")
    assert "ask__compose" not in message


def test_run_start_error_in_a_child_flow_names_the_gate_and_callsite():
    message = _run_start_error(_CHILD_FLOW, {"proj": "x"})
    assert message.startswith(f"{_LLM_SITE} in 'sub': provider azure")
    assert "ask__compose" not in message


def test_plain_agent_run_start_error_is_unchanged():
    message = _run_start_error(_AGENT_FLOW.format(llm_config=_AZURE), {})
    assert message.startswith("Agent 'w': provider azure")


def _replay_with_base_url(text: str, path: str) -> str:
    flow = load_flow(text)
    effective = resolve_llm_cascade(
        copy.deepcopy(flow.compiled), {}, defaults={"provider": "openai", "model": "m"},
    )
    effective[path]["base_url"] = "https://x.example.com/v1"
    with pytest.raises(ValueError) as caught:
        apply_effective_llm_map(copy.deepcopy(flow.compiled), effective)
    return str(caught.value)


def test_replayed_map_error_names_the_gate():
    gate = _replay_with_base_url(_GATE_FLOW.format(llm_config=_OPENAI), "ask__compose")
    assert gate.startswith(f"effective LLM config for {_LLM_SITE} has a base_url without a host key")
    agent = _replay_with_base_url(_AGENT_FLOW.format(llm_config=_OPENAI), "w")
    assert agent.startswith("effective LLM config for 'w' has a base_url")


def test_resume_recorded_field_error_names_the_gate():
    flow = load_flow(_GATE_FLOW.format(llm_config=_OPENAI))
    with pytest.raises(ValueError) as caught:
        yian.run._check_recorded_llm_fields(
            {"ask__compose": {"provider": "openai", "model": "m", "temperature": 0.5}},
            static_agent_labels(flow.compiled),
        )
    assert str(caught.value).startswith(f"{_LLM_SITE}: ")


def test_composer_label_is_render_only():
    flow = load_flow(_GATE_FLOW.format(llm_config=_OPENAI))
    assert flow.compiled.nodes["ask__compose"].llm_config_label == _LLM_SITE
    unlabeled = copy.deepcopy(flow.compiled)
    unlabeled.nodes["ask__compose"].llm_config_label = None
    assert flow_fingerprint(unlabeled) == flow_fingerprint(flow.compiled)
    plain = load_flow(_AGENT_FLOW.format(llm_config=_OPENAI))
    assert plain.compiled.nodes["w"].llm_config_label is None
