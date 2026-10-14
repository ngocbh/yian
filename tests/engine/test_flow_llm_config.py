"""Flow-level `llm_config:` — parsing it onto ComposeFile and threading it onto the
compiled flow (the cascade's flow layer)."""

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.parse.file import parse_file
from yian.compose.loader import load_flow


def test_flow_level_llm_config_parsed():
    text = """
id: f
name: f
llm_config:
  provider: anthropic
  model: claude-sonnet-4-5
nodes:
  a:
    kind: agent
    prompt: hi
"""
    f = parse_file(text)
    assert f.llm_config == {"provider": "anthropic", "model": "claude-sonnet-4-5"}


def test_flow_level_llm_config_absent_is_empty():
    text = "id: f\nname: f\nnodes:\n  a:\n    kind: agent\n    prompt: hi\n"
    assert parse_file(text).llm_config == {}


def test_compiled_carries_flow_llm_config():
    text = """
id: f
name: f
llm_config: {provider: anthropic, model: claude-sonnet-4-5}
nodes:
  a: {kind: agent, prompt: hi}
output: ${a.output}
"""
    loaded = load_flow(text)
    assert loaded.compiled.flow_llm_config == {
        "provider": "anthropic", "model": "claude-sonnet-4-5"
    }


def test_compiled_flow_llm_config_defaults_empty():
    text = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi}\noutput: ${a.output}\n"
    assert load_flow(text).compiled.flow_llm_config == {}


_TYPO_CONFIGS = [
    "{modle: gpt-5}", "{openai: {temperature: hot}}", "{provider: anthropc}", "{inherit: false}"
]


@pytest.mark.parametrize("config", _TYPO_CONFIGS)
def test_flow_level_llm_config_is_checked_like_a_node_one(config):
    text = f"""id: f
name: f
llm_config: {config}
nodes:
  a: {{kind: agent, prompt: hi}}
output: ${{a.output}}
"""
    with pytest.raises(LoadError, match=r"flow: llm_config: 1 validation error for LLMConfig") as caught:
        load_flow(text, source_label="f.yaml")
    assert caught.value.diagnostic.frames[-1].range.start_line == 3


@pytest.mark.parametrize("config", _TYPO_CONFIGS)
def test_def_level_llm_config_is_checked_like_a_node_one(config):
    text = f"""id: f
name: f
defs:
  sub:
    llm_config: {config}
    nodes:
      a: {{kind: agent, prompt: hi}}
    output: ${{a.output}}
nodes:
  s: {{kind: call, call: sub}}
output: ${{s.output}}
"""
    with pytest.raises(
        LoadError, match=r"defs entry 'sub': llm_config: 1 validation error for LLMConfig"
    ) as caught:
        load_flow(text, source_label="f.yaml")
    assert caught.value.diagnostic.frames[-1].range.start_line == 5


def test_def_level_llm_config_must_be_a_mapping():
    text = """id: f
name: f
defs:
  sub:
    llm_config: gpt-5
    nodes:
      a: {kind: agent, prompt: hi}
    output: ${a.output}
nodes:
  s: {kind: call, call: sub}
output: ${s.output}
"""
    with pytest.raises(LoadError, match=r"defs entry 'sub': llm_config must be a mapping"):
        load_flow(text, source_label="f.yaml")
