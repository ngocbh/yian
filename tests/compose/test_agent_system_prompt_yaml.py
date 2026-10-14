"""`system_prompt:` on AGENT: YAML shape, reference scope, located errors, and flow identity."""

from __future__ import annotations

import pytest

from yian.compile.manifest import flow_fingerprint, flow_manifest
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.nodes.agent import AgentNode

PERSONA = '"You are ${name}, ${title}. Speak as ${name} in the first person."'


def _flow(system_line: str) -> str:
    return f"""\
id: chat
name: chat
input:
  name: !str
  title: !str
  message: !str
nodes:
  reply:
    kind: agent
    input: {{name: "${{input.name}}", title: "${{input.title}}", message: "${{input.message}}"}}
    variables:
      tone: !str warm
{system_line}
    prompt: ${{message}}
    output: str
output: ${{reply.output}}
"""


def _compact(system_line: str) -> str:
    return f"""\
id: reply
name: reply
input:
  name: !str
  title: !str
  message: !str
output: str
kind: agent
{system_line}
prompt: ${{message}}
"""


def test_system_prompt_reaches_the_agent_node() -> None:
    node = load_flow(_flow(f"    system_prompt: {PERSONA}")).compiled.nodes["reply"]
    assert node.system_prompt == "You are ${name}, ${title}. Speak as ${name} in the first person."


def test_system_prompt_defaults_to_none() -> None:
    assert load_flow(_flow("")).compiled.nodes["reply"].system_prompt is None
    assert AgentNode("reply").system_prompt is None


def test_compact_flow_accepts_system_prompt() -> None:
    node = load_flow(_compact(f"system_prompt: {PERSONA}")).compiled.nodes["reply"]
    assert node.system_prompt == "You are ${name}, ${title}. Speak as ${name} in the first person."


def test_system_prompt_sees_the_node_variables() -> None:
    node = load_flow(_flow('    system_prompt: "Be ${tone}."')).compiled.nodes["reply"]
    assert node.system_prompt == "Be ${tone}."


@pytest.mark.parametrize(
    ("ref", "hint"),
    [
        ("${nickname}", ""),
        ("${input.name}", "a pool ref belongs in an input `from:`"),
    ],
)
def test_unknown_ref_points_at_the_system_prompt_line(ref: str, hint: str) -> None:
    with pytest.raises(LoadError, match="system_prompt: .* is not a declared input or variable") as caught:
        load_flow(_flow(f'    system_prompt: "You are {ref}."'))
    assert hint in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == 13


def test_unknown_ref_in_a_compact_flow_is_located() -> None:
    with pytest.raises(LoadError, match="system_prompt: .* is not a declared input") as caught:
        load_flow(_compact('system_prompt: "You are ${nickname}."'))
    assert caught.value.diagnostic.frames[-1].range.start_line == 9


@pytest.mark.parametrize("value", ["123", "[a, b]", "{a: b}", "null", '""'])
def test_non_string_system_prompt_points_at_its_line(value: str) -> None:
    with pytest.raises(LoadError, match="`system_prompt:` must be a non-empty string") as caught:
        load_flow(_flow(f"    system_prompt: {value}"))
    assert caught.value.diagnostic.frames[-1].range.start_line == 13


def test_agent_node_rejects_a_non_string_system_prompt() -> None:
    with pytest.raises(TypeError, match="system_prompt must be a non-empty string or None"):
        AgentNode("reply", system_prompt=7)
    with pytest.raises(TypeError, match="system_prompt must be a non-empty string or None"):
        AgentNode("reply", system_prompt="")


def test_flow_without_system_prompt_keeps_its_fingerprint() -> None:
    # An absent `system_prompt:` adds nothing to identity. Pinned after `mode` left the
    # AGENT manifest.
    for compiled in (load_flow(_flow("")).compiled, load_flow(_compact("")).compiled):
        assert "system_prompt" not in flow_manifest(compiled)["nodes"]["reply"]
    assert flow_fingerprint(load_flow(_flow("")).compiled) == (
        "fc15ce95efcac79320ef54dad98e219283ab094c8521dbdcd8d6a9f65397e801"
    )
    assert flow_fingerprint(load_flow(_compact("")).compiled) == (
        "a86275835f7f45b1a3eb6f9f4604cec49955a760339c13f0de26d9a539c47312"
    )


def test_system_prompt_is_identity() -> None:
    base = flow_fingerprint(load_flow(_flow("")).compiled)
    persona = flow_fingerprint(load_flow(_flow(f"    system_prompt: {PERSONA}")).compiled)
    other = flow_fingerprint(load_flow(_flow('    system_prompt: "You are ${name}."')).compiled)
    assert len({base, persona, other}) == 3
