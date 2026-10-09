"""`inherit: false` — the whole-node cascade opt-out. Parsed off `llm_config`, the
reserved `inherit` key is stripped before LLMConfig validation and carried as a flag."""

import pytest

from yian.compose.loader import load_flow
from yian.compose.helpers.errors import LoadError


def _agent(loaded, nid):
    return loaded.compiled.nodes[nid]


def test_inherit_false_parsed_and_stripped():
    text = """
id: f
name: f
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config: {model: claude-opus-4-8, inherit: false}
output: ${a.output}
"""
    loaded = load_flow(text)
    a = _agent(loaded, "a")
    assert a.llm_inherit is False
    # `inherit` is stripped before LLMConfig validation; the own config has no `inherit` key
    assert a.own_llm_config == {"model": "claude-opus-4-8"}


def test_inherit_defaults_true():
    text = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi, llm_config: {model: x}}\noutput: ${a.output}\n"
    assert _agent(load_flow(text), "a").llm_inherit is True


def test_typo_key_still_loud():
    text = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi, llm_config: {temparature: 1}}\noutput: ${a.output}\n"
    with pytest.raises(LoadError):
        load_flow(text)
