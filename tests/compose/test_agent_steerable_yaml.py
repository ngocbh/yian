"""`steerable:` on AGENT: YAML shape, located errors, and flow identity."""

from __future__ import annotations

import pytest

from yian.compile.manifest import flow_fingerprint
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.nodes.agent import AgentNode


def _flow(steerable_line: str, *, output: str = "    output: str") -> str:
    return f"""\
id: steer
name: steer
nodes:
  chat:
    kind: agent
    prompt: Help the user.
{steerable_line}
{output}
output: ${{chat.output}}
"""


def test_steerable_reaches_the_agent_node_and_defaults_off() -> None:
    assert load_flow(_flow("    steerable: true")).compiled.nodes["chat"].steerable is True
    assert load_flow(_flow("")).compiled.nodes["chat"].steerable is False


def test_steerable_agent_without_declared_output_is_text() -> None:
    flow = load_flow(_flow("    steerable: true", output=""))
    assert flow.compiled.nodes["chat"].steerable is True


@pytest.mark.parametrize(
    ("line", "output", "message"),
    [
        ("    steerable: yes please", "    output: str", "`steerable:` must be true or false"),
        ("    steerable: true", "    output: int", "steerable needs `str` output"),
        ("    steerable: true", "    output: {answer: str}", "steerable needs `str` output"),
    ],
)
def test_bad_steerable_points_at_the_steerable_line(line, output, message) -> None:
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(_flow(line, output=output))
    assert caught.value.diagnostic.frames[-1].range.start_line == 7


def test_steerable_is_identity_but_false_is_not() -> None:
    base = flow_fingerprint(load_flow(_flow("")).compiled)
    assert flow_fingerprint(load_flow(_flow("    steerable: false")).compiled) == base
    assert flow_fingerprint(load_flow(_flow("    steerable: true")).compiled) != base


def test_agent_node_rejects_a_non_bool_steerable() -> None:
    with pytest.raises(TypeError, match="steerable must be a bool"):
        AgentNode("chat", prompt="p", steerable=1)


def test_needs_steering_follows_steerable() -> None:
    assert AgentNode("chat", prompt="p", steerable=True).needs_steering is True
    assert AgentNode("chat", prompt="p").needs_steering is False
