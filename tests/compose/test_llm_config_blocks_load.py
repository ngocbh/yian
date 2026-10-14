"""Provider blocks in authored `llm_config:` layers are checked when the flow loads."""

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow

SENTINEL = "sk-sentinel-91c4d07e"


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="f.yaml")
    return caught.value


def test_flow_level_block_typo_fails_at_load():
    error = _load_error(f"""id: f
name: f
llm_config:
  provider: anthropic
  anthropic: {{max_tokns: 100, metadata: {{user_id: {SENTINEL}}}}}
nodes:
  a: {{kind: agent, prompt: hi}}
output: ${{a.output}}
""")
    message = str(error)
    assert "flow: llm_config:" in message
    assert "`anthropic.max_tokns` is not a supported field of the Anthropic Messages API" in message
    assert "allowed: cache_control, inference_geo, max_tokens," in message
    assert SENTINEL not in message
    assert error.diagnostic.frames[-1].range.start_line == 4


def test_def_level_block_wrong_type_fails_at_load():
    error = _load_error(f"""id: f
name: f
defs:
  sub:
    llm_config:
      qwen-cn: {{thinking_budget: "{SENTINEL}"}}
    nodes:
      a: {{kind: agent, prompt: hi}}
    output: ${{a.output}}
nodes:
  s: {{kind: call, call: sub}}
output: ${{s.output}}
""")
    message = str(error)
    assert "defs entry 'sub': llm_config:" in message
    assert "qwen-cn.thinking_budget" in message
    assert "valid integer" in message
    assert SENTINEL not in message
    assert error.diagnostic.frames[-1].range.start_line == 6


def test_node_level_reserved_field_fails_at_load():
    error = _load_error(f"""id: f
name: f
nodes:
  a:
    kind: agent
    prompt: hi
    llm_config:
      provider: openai
      openai: {{text: {{format: {{type: text}}}}, user: {SENTINEL}}}
output: ${{a.output}}
""")
    message = str(error)
    assert "node 'a': llm_config:" in message
    assert "`openai.text.format` is built by yian (structured output); remove it" in message
    assert SENTINEL not in message
    assert error.diagnostic.frames[-1].range.start_line == 8


def test_python_spelling_of_a_dash_block_fails_at_load():
    error = _load_error("""id: f
name: f
llm_config:
  glm_cn: {do_sample: true}
nodes:
  a: {kind: agent, prompt: hi}
output: ${a.output}
""")
    assert "write `glm-cn`" in str(error)


def test_valid_blocks_load_at_every_level():
    loaded = load_flow("""id: f
name: f
llm_config:
  provider: anthropic
  anthropic: {max_tokens: 8000}
defs:
  sub:
    llm_config:
      minimax-cn: {thinking: {type: adaptive}}
    nodes:
      a: {kind: agent, prompt: hi}
    output: ${a.output}
nodes:
  s: {kind: call, call: sub}
  b:
    kind: agent
    prompt: hi
    llm_config:
      google: {generationConfig: {thinkingConfig: {thinkingLevel: HIGH}}}
output: ${s.output}
""")
    assert loaded.compiled.flow_llm_config["anthropic"] == {"max_tokens": 8000}
