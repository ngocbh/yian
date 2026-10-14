"""The removed AGENT `mode:` key: located load errors, and no mode in the Python API or manifest."""

from __future__ import annotations

import pytest

import yian.nodes.agent as agent_pkg
from yian.compile.manifest import flow_manifest
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.nodes.agent import AgentNode

REMOVED = (
    "`mode:` was removed: an AGENT with tools runs the tool loop, "
    "one without tools makes a single call; delete this line"
)
TOOLS_NOW_USED = " (its `tools:` will now be used)"


def _flow(*lines: str) -> str:
    """One `chat` agent whose extra body lines start on line 7."""
    body = "".join(f"    {line}\n" for line in lines)
    return f"""\
id: removed
name: removed
nodes:
  chat:
    kind: agent
    prompt: Help the user.
{body}output: ${{chat.output}}
"""


def _load_error(source: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(source)
    return caught.value


def _line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


@pytest.mark.parametrize("value", ["tool_calling", "plain", "react", "[plain]"])
def test_any_mode_on_an_agent_is_a_located_load_error(value: str) -> None:
    error = _load_error(_flow(f"mode: {value}"))
    assert f"node 'chat': {REMOVED}" in str(error)
    assert "will now be used" not in str(error)
    assert _line(error) == 7


def test_mode_plain_with_tools_says_the_tools_will_now_be_used() -> None:
    error = _load_error(_flow("tools: [lookup]", "mode: plain"))
    assert str(error).endswith(f"node 'chat': {REMOVED}{TOOLS_NOW_USED}")
    assert _line(error) == 8


def test_mode_tool_calling_with_tools_gets_the_base_message() -> None:
    error = _load_error(_flow("tools: [lookup]", "mode: tool_calling"))
    assert f"node 'chat': {REMOVED}" in str(error)
    assert "will now be used" not in str(error)
    assert _line(error) == 8


def test_mode_in_a_compact_flow_is_located() -> None:
    source = """\
id: reply
name: reply
output: str
kind: agent
mode: plain
prompt: Say hi.
"""
    error = _load_error(source)
    assert f"node 'reply': {REMOVED}" in str(error)
    assert _line(error) == 5


def test_mode_on_a_child_flow_agent_is_rejected() -> None:
    source = """\
id: parent
name: parent
defs:
  child:
    nodes:
      inner:
        kind: agent
        mode: plain
        prompt: Say hi.
    output: ${inner.output}
nodes:
  run_child:
    kind: call
    call: child
output: ${run_child.output}
"""
    error = _load_error(source)
    assert f"node 'inner': {REMOVED}" in str(error)
    assert _line(error) == 8


def test_mode_from_a_yaml_merge_key_is_rejected() -> None:
    source = """\
id: merged
name: merged
x-base: &base {kind: agent, mode: plain}
nodes:
  chat:
    <<: *base
    prompt: hi
output: ${chat.output}
"""
    error = _load_error(source)
    assert f"node 'chat': {REMOVED}" in str(error)


ADAPTIVE = (
    "node 'ask' (kind=human_input): `adaptive_questions.mode:` was removed: an AGENT with "
    "tools runs the tool loop, one without tools makes a single call; delete this line"
)


def test_adaptive_questions_mode_block_form_is_located() -> None:
    source = """\
id: gate
name: gate
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask
      mode: plain
output: ${ask.output}
"""
    error = _load_error(source)
    assert ADAPTIVE in str(error)
    assert _line(error) == 8


def test_adaptive_questions_mode_flow_form_is_rejected() -> None:
    source = """\
id: gate
name: gate
nodes:
  ask:
    kind: human_input
    adaptive_questions: {prompt: Ask, mode: plain}
output: ${ask.output}
"""
    error = _load_error(source)
    assert ADAPTIVE in str(error)
    assert _line(error) == 6


def test_adaptive_questions_mode_past_a_recursive_alias_is_located() -> None:
    source = """\
id: gate
name: gate
nodes:
  ask: &a {kind: human_input, mode: plain, adaptive_questions: *a}
output: ${ask.output}
"""
    error = _load_error(source)
    assert ADAPTIVE in str(error)
    assert _line(error) == 4


def test_adaptive_questions_mode_wins_over_an_unknown_key() -> None:
    source = """\
id: gate
name: gate
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask
      bogus: 1
      mode: plain
output: ${ask.output}
"""
    error = _load_error(source)
    assert ADAPTIVE in str(error)
    assert "is not allowed" not in str(error)


def test_python_agent_node_has_no_mode() -> None:
    with pytest.raises(TypeError):
        AgentNode("a", mode="plain")
    for name in ("MODES", "register_mode", "DEFAULT_MODE"):
        assert not hasattr(agent_pkg, name)


def test_agent_manifest_has_no_mode() -> None:
    flow = load_flow(_flow())
    assert "mode" not in flow_manifest(flow.compiled)["nodes"]["chat"]


def test_steerable_agent_without_tools_loads() -> None:
    flow = load_flow(_flow("steerable: true", "output: str"))
    assert flow.compiled.nodes["chat"].steerable is True


def test_builtin_tools_is_rejected_at_the_tools_line() -> None:
    error = _load_error(_flow("tools: [builtin_tools]"))
    assert "builtin_tools" in str(error)
    assert _line(error) == 7
