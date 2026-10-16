"""Authored `llm_config` layers read `${...}` values and tags when the flow loads."""

from __future__ import annotations

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.nodes.agent import AgentNode
from yian.nodes.contracts import Binding, ValueTemplate

_INPUTS = """\
input:
  m: !str
  p: !str
  t: !float
  u: !str
  i: !bool
  a: !object
  cfg: !object
"""


def _flow(llm_lines: str, *, typedefs: str = "", inputs: str = _INPUTS) -> str:
    """Return a flow whose agent `a` has the node lines `llm_lines` (indented 4)."""
    return f"""\
id: f
name: f
{typedefs}{inputs}nodes:
  a:
    kind: agent
    prompt: Go.
{llm_lines}output: ${{a.output}}
"""


def _agent(text: str) -> AgentNode:
    return load_flow(text, source_label="f.yaml").compiled.nodes["a"]


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="f.yaml")
    return caught.value


def _line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


def _line_of(text: str, needle: str) -> int:
    return next(n for n, line in enumerate(text.splitlines(), 1) if needle in line)


def test_a_field_value_loads_as_a_template():
    agent = _agent(_flow("    llm_config: {model: '${input.m}'}\n"))
    assert agent.own_llm_config == ValueTemplate(
        {"model": "${input.m}"}, agent.own_llm_config.origins
    )
    assert agent.llm_config is None


def test_a_block_field_value_loads_as_a_template():
    agent = _agent(
        _flow(
            "    llm_config:\n"
            "      provider: anthropic\n"
            "      anthropic: {temperature: '${input.t}'}\n"
        )
    )
    assert isinstance(agent.own_llm_config, ValueTemplate)
    assert agent.own_llm_config.value == {
        "provider": "anthropic",
        "anthropic": {"temperature": "${input.t}"},
    }


def test_a_whole_block_value_loads_as_a_template():
    agent = _agent(
        _flow("    llm_config: {provider: anthropic, anthropic: '${input.a}'}\n")
    )
    assert isinstance(agent.own_llm_config, ValueTemplate)
    assert agent.own_llm_config.value == {
        "provider": "anthropic",
        "anthropic": "${input.a}",
    }


def test_a_whole_record_value_loads_as_a_binding():
    agent = _agent(_flow("    llm_config: ${input.cfg}\n"))
    assert isinstance(agent.own_llm_config, Binding)
    assert agent.own_llm_config.source == "${input.cfg}"
    assert agent.own_llm_config.contract is None
    assert agent.llm_config is None
    assert agent.llm_inherit is True


def test_a_tagged_whole_record_keeps_its_type():
    typedefs = "typedefs:\n  Cfg:\n    model: str\n"
    agent = _agent(_flow("    llm_config: !Cfg ${input.cfg}\n", typedefs=typedefs))
    assert isinstance(agent.own_llm_config, Binding)
    assert set(agent.own_llm_config.contract.type.fields) == {"model"}


@pytest.mark.parametrize("value", ["gpt-4", "'x${input.m}'", "5"])
def test_a_whole_record_that_is_not_one_value_must_be_a_mapping(value):
    error = _load_error(_flow(f"    llm_config: {value}\n"))
    assert "node 'a': llm_config must be a mapping" in str(error)


def test_a_tag_on_a_field_value_is_read_and_consumed():
    agent = _agent(_flow("    llm_config: {model: !str '${input.m}'}\n"))
    template = agent.own_llm_config
    assert isinstance(template, ValueTemplate)
    binding = template.value["model"]
    assert isinstance(binding, Binding)
    assert binding.source == "${input.m}"
    assert binding.contract.type.kind.value == "str"


def test_a_tag_on_the_whole_mapping_is_rejected():
    error = _load_error(_flow("    llm_config: !object {model: '${input.m}'}\n"))
    assert "node 'a': llm_config" in str(error)
    assert "tag a field's value" in str(error)


def test_literal_inherit_is_split_off_a_template_layer():
    agent = _agent(_flow("    llm_config: {inherit: false, model: '${input.m}'}\n"))
    assert agent.llm_inherit is False
    assert agent.own_llm_config.value == {"model": "${input.m}"}
    assert all(path[0] != "inherit" for path, _ in agent.own_llm_config.origins)


def test_inherit_from_a_value_fails():
    text = _flow("    llm_config:\n      model: m\n      inherit: ${input.i}\n")
    error = _load_error(text)
    assert "node 'a': llm_config `inherit:` must be written as true or false" in str(error)
    assert _line(error) == _line_of(text, "inherit:")


@pytest.mark.parametrize("field", ["api_key", "organization", "project"])
def test_a_connection_field_from_a_value_fails_at_the_field(field):
    text = _flow(
        "    llm_config:\n"
        "      provider: openai\n"
        f"      {field}: ${{input.u}}\n"
    )
    error = _load_error(text)
    message = str(error)
    assert f"node 'a': llm_config.{field} " in message
    assert "not in a flow file" in message
    assert _line(error) == _line_of(text, f"{field}:")


def test_headers_from_a_value_fail_at_the_field():
    text = _flow(
        "    llm_config:\n"
        "      provider: openai\n"
        "      headers: {X-Team: '${input.u}'}\n"
    )
    error = _load_error(text)
    assert "node 'a': llm_config.headers goes wherever the host's api_key" in str(error)
    assert _line(error) == _line_of(text, "headers:")


def test_a_whole_record_type_declaring_api_key_fails():
    typedefs = "typedefs:\n  Cfg:\n    model: str\n    api_key: str\n"
    text = _flow("    llm_config: !Cfg ${input.cfg}\n", typedefs=typedefs)
    error = _load_error(text)
    message = str(error)
    assert "node 'a': llm_config: its type declares `api_key`" in message
    assert "llm_config.api_key is a host credential" in message
    assert _line(error) == _line_of(text, "llm_config:")


def test_base_url_from_a_value_fails():
    text = _flow(
        "    llm_config:\n"
        "      provider: openai\n"
        "      base_url: ${input.u}\n"
    )
    error = _load_error(text)
    assert "node 'a': llm_config.base_url decides where" in str(error)
    assert _line(error) == _line_of(text, "base_url:")


def test_base_url_from_a_value_fails_beside_azure():
    text = _flow(
        "    llm_config:\n"
        "      provider: azure\n"
        "      model: m\n"
        "      base_url: ${input.u}\n"
    )
    error = _load_error(text)
    message = str(error)
    assert "node 'a': llm_config.base_url can't come from a value" in message
    assert "Azure resource endpoint" in message
    assert _line(error) == _line_of(text, "base_url:")


def test_a_literal_base_url_beside_a_provider_value_fails():
    text = _flow(
        "    llm_config:\n"
        "      provider: ${input.p}\n"
        "      base_url: https://x.openai.azure.com\n"
    )
    error = _load_error(text)
    assert "node 'a': llm_config.base_url decides where" in str(error)
    assert _line(error) == _line_of(text, "base_url:")


def test_a_literal_base_url_beside_literal_azure_still_loads():
    agent = _agent(
        _flow(
            "    llm_config:\n"
            "      provider: azure\n"
            "      model: ${input.m}\n"
            "      base_url: https://x.openai.azure.com\n"
        )
    )
    assert agent.own_llm_config.value["base_url"] == "https://x.openai.azure.com"


def test_an_unknown_top_level_key_beside_a_value_fails():
    error = _load_error(_flow("    llm_config: {model: '${input.m}', temparature: 1}\n"))
    assert "node 'a': llm_config:" in str(error)
    assert "temparature" in str(error)


def test_a_literal_node_layer_is_still_a_plain_dict():
    agent = _agent(_flow("    llm_config: {provider: openai, model: gpt-x}\n"))
    assert type(agent.own_llm_config) is dict
    assert agent.own_llm_config == {"provider": "openai", "model": "gpt-x"}


def test_a_quoted_span_is_no_longer_literal_text():
    agent = _agent(_flow("    llm_config: {model: \"${input.m}\"}\n"))
    assert isinstance(agent.own_llm_config, ValueTemplate)


def test_a_doubled_dollar_reads_as_one_dollar():
    agent = _agent(_flow("    llm_config: {model: \"a$$b\"}\n"))
    assert agent.own_llm_config == {"model": "a$b"}


def test_a_python_built_agent_keeps_its_literal_config():
    agent = AgentNode("a", prompt="p", llm_config={"model": "m"})
    assert agent.llm_config == {"model": "m"}
    assert agent.own_llm_config == {"model": "m"}


def test_a_literal_block_field_beside_a_value_is_checked():
    text = _flow(
        "    llm_config:\n"
        "      provider: anthropic\n"
        "      anthropic:\n"
        "        temperature: ${input.t}\n"
        "        thinking: {type: enabled, budget_tokens: 10}\n"
    )
    error = _load_error(text)
    message = str(error)
    assert "node 'a': llm_config: `anthropic.thinking.budget_tokens`:" in message
    assert "greater than or equal to 1024" in message
    assert _line(error) == _line_of(text, "thinking:")


def test_an_unknown_block_key_beside_a_value_fails():
    text = _flow(
        "    llm_config:\n"
        "      provider: anthropic\n"
        "      anthropic:\n"
        "        temperature: ${input.t}\n"
        "        max_tokns: 100\n"
    )
    error = _load_error(text)
    message = str(error)
    assert "node 'a': llm_config: `anthropic.max_tokns`" in message
    assert "allowed:" in message
    assert _line(error) == _line_of(text, "max_tokns:")


def test_a_required_block_field_given_as_a_value_loads():
    agent = _agent(
        _flow(
            "    llm_config:\n"
            "      provider: anthropic\n"
            "      anthropic: {thinking: {type: enabled, budget_tokens: !int '${input.t}'}}\n"
        )
    )
    assert isinstance(agent.own_llm_config, ValueTemplate)


def test_a_literal_block_beside_a_value_is_checked_whole():
    text = _flow(
        "    llm_config:\n"
        "      provider: ${input.p}\n"
        "      anthropic: {max_tokns: 100}\n"
    )
    error = _load_error(text)
    assert "anthropic.max_tokns" in str(error)


_ASK = """\
id: f
name: f
input: {proj: !str , m: !str , t: !float }
nodes:
  ask:
    kind: human_input
    input: {proj: '${input.proj}'}
    adaptive_questions:
      prompt: "Q ${proj}"
      llm_config:
LLM
output: ${ask.output}
"""

_ASK_SITE = "node 'ask' adaptive_questions.llm_config"


def _ask(llm_lines: str) -> str:
    """Return a flow whose adaptive gate has `llm_lines` under `llm_config:` (indented 8)."""
    return _ASK.replace("LLM\n", llm_lines)


def _composer(text: str) -> AgentNode:
    return load_flow(text, source_label="f.yaml").compiled.nodes["ask__compose"]


def test_a_composer_value_loads_as_a_template():
    agent = _composer(_ask("        model: ${input.m}\n"))
    assert isinstance(agent.own_llm_config, ValueTemplate)
    assert agent.own_llm_config.value == {"model": "${input.m}"}
    assert agent.llm_config is None


def test_a_composer_tag_is_read_and_consumed():
    agent = _composer(_ask("        model: !str ${input.m}\n"))
    assert isinstance(agent.own_llm_config.value["model"], Binding)


def test_a_composer_excluded_field_from_a_value_names_the_gate():
    text = _ask("        provider: openai\n        api_key: ${input.m}\n")
    error = _load_error(text)
    message = str(error)
    assert message.startswith(f"{_ASK_SITE}.api_key is a host credential")
    assert "ask__compose" not in message
    assert _line(error) == _line_of(text, "api_key:")


def test_a_composer_literal_block_field_beside_a_value_names_the_gate():
    text = _ask(
        "        provider: anthropic\n"
        "        anthropic:\n"
        "          temperature: ${input.t}\n"
        "          thinking: {type: enabled, budget_tokens: 10}\n"
    )
    error = _load_error(text)
    message = str(error)
    assert message.startswith(f"{_ASK_SITE}: `anthropic.thinking.budget_tokens`:")
    assert _line(error) == _line_of(text, "thinking:")


def test_a_composer_tag_on_the_whole_mapping_names_the_gate():
    text = _ASK.replace("      llm_config:\nLLM\n", "      llm_config: !object {model: m}\n")
    error = _load_error(text)
    assert str(error).startswith(f"{_ASK_SITE}: a tag on the whole mapping")


def test_a_literal_composer_layer_is_unchanged():
    agent = _composer(_ask("        provider: openai\n        model: gpt-x\n"))
    assert type(agent.own_llm_config) is dict
    assert agent.llm_config == {"provider": "openai", "model": "gpt-x"}


def _flow_layer(flow_lines: str, *, typedefs: str = "") -> str:
    """Return a flow with the flow-level lines `flow_lines` and one agent `a`."""
    return f"id: f\nname: f\n{typedefs}{_INPUTS}{flow_lines}" + (
        "nodes:\n  a: {kind: agent, prompt: Go.}\noutput: ${a.output}\n"
    )


def _def_layer(def_lines: str) -> str:
    """Return a flow calling `sub`, a def whose body has the lines `def_lines` (indented 4)."""
    return f"""\
id: f
name: f
defs:
  sub:
    input: {{m: !str , cfg: !object }}
{def_lines}    nodes:
      a: {{kind: agent, prompt: Go.}}
    output: ${{a.output}}
nodes:
  s: {{kind: call, call: sub, input: {{m: x, cfg: {{}}}}}}
output: ${{s.output}}
"""


def _def_child(text: str):
    (child,) = load_flow(text, source_label="f.yaml").compiled.nodes["s"].origin_children()
    return child


def test_a_flow_field_value_loads_as_a_template():
    flow = load_flow(_flow_layer("llm_config: {model: '${input.m}'}\n"))
    assert flow.compiled.flow_llm_config == ValueTemplate(
        {"model": "${input.m}"}, flow.compiled.flow_llm_config.origins
    )


def test_a_flow_whole_record_value_loads_as_a_binding():
    text = _flow_layer("llm_config: ${input.cfg}\n")
    flow = load_flow(text, source_label="f.yaml")
    layer = flow.compiled.flow_llm_config
    assert isinstance(layer, Binding)
    assert layer.source == "${input.cfg}"
    frame = flow.origin_resolver.resolve(layer.origin)
    assert frame.range.start_line == _line_of(text, "llm_config:")


def test_a_flow_tag_is_read_and_consumed():
    flow = load_flow(_flow_layer("llm_config: {model: !str '${input.m}'}\n"))
    binding = flow.compiled.flow_llm_config.value["model"]
    assert isinstance(binding, Binding)
    assert binding.contract.type.kind.value == "str"


def test_a_tag_on_a_typedef_still_fails_beside_a_flow_value():
    typedefs = "typedefs:\n  Cfg:\n    model: !str str\n"
    text = _flow_layer("llm_config: {model: !str '${input.m}'}\n", typedefs=typedefs)
    error = _load_error(text)
    assert "typedef 'Cfg': a type definition is written as a type" in str(error)


def test_a_flow_excluded_field_from_a_value_fails_at_the_field():
    text = _flow_layer("llm_config:\n  provider: openai\n  api_key: ${input.u}\n")
    error = _load_error(text)
    assert str(error.diagnostic.message).startswith(
        "flow: llm_config.api_key is a host credential"
    )
    assert _line(error) == _line_of(text, "api_key:")


@pytest.mark.parametrize("value", ["false", "${input.i}"])
def test_inherit_at_flow_level_still_fails(value):
    text = _flow_layer(f"llm_config:\n  model: ${{input.m}}\n  inherit: {value}\n")
    error = _load_error(text)
    assert "flow: llm_config:" in str(error)
    assert "inherit" in str(error)


def test_a_literal_flow_layer_is_still_a_plain_dict():
    flow = load_flow(_flow_layer("llm_config: {provider: openai, model: \"a$$b\"}\n"))
    assert flow.compiled.flow_llm_config == {"provider": "openai", "model": "a$b"}
    assert type(flow.compiled.flow_llm_config) is dict


def test_a_def_field_value_loads_as_a_template():
    child = _def_child(_def_layer("    llm_config: {model: '${input.m}'}\n"))
    assert isinstance(child.flow_llm_config, ValueTemplate)
    assert child.flow_llm_config.value == {"model": "${input.m}"}


def test_a_def_whole_record_value_loads_as_a_binding():
    child = _def_child(_def_layer("    llm_config: !object ${input.cfg}\n"))
    assert isinstance(child.flow_llm_config, Binding)
    assert child.flow_llm_config.contract.type.kind.value == "object"


def test_a_def_excluded_field_from_a_value_fails_at_the_field():
    text = _def_layer("    llm_config: {provider: openai, project: '${input.m}'}\n")
    error = _load_error(text)
    assert "defs entry 'sub': llm_config.project goes wherever" in str(error)
    assert _line(error) == _line_of(text, "llm_config:")


def test_a_literal_def_layer_is_still_a_plain_dict():
    child = _def_child(_def_layer("    llm_config: {model: m}\n"))
    assert child.flow_llm_config == {"model": "m"}
