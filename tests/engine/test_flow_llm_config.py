"""Flow-level `llm_config:` — parsing it onto ComposeFile and threading it onto the
compiled flow (the cascade's flow layer)."""

from agent_composer.compose.parse.file import parse_file
from agent_composer.compose.loader import load_flow


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
