"""AGENT `tools:`, `retries:`, and `llm_config.inherit` shapes are checked at load."""

from __future__ import annotations

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError


def _flow(field_line: str) -> str:
    return f"""\
id: shapes
name: shapes
nodes:
  a:
    kind: agent
    prompt: Help the user.
{field_line}
output: ${{a.output}}
"""


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("    tools: search", r"`tools:` must be a list of tool names; write `tools: \[search\]`"),
        ("    tools: [1]", r"`tools:` must be a list of tool names"),
        ("    tools: ['']", r"`tools:` must be a list of tool names"),
        ("    tools: null", r"`tools:` must be a list of tool names"),
        ("    retries: -1", r"`retries:` must be a non-negative integer, got -1"),
        ("    retries: abc", r"`retries:` must be a non-negative integer, got 'abc'"),
        ("    retries: true", r"`retries:` must be a non-negative integer, got True"),
        ("    retries: 1.5", r"`retries:` must be a non-negative integer, got 1.5"),
        ("    retries: null", r"`retries:` must be a non-negative integer, got None"),
        ("    llm_config: {inherit: 'false'}", r"`inherit:` must be true or false, got 'false'"),
        ("    llm_config: {inherit: null}", r"`inherit:` must be true or false, got None"),
        (
            "    llm_config: {inherit: on}",
            r"`inherit:` must be true or false, got 'on'; `on` is text; write `true`",
        ),
        ("    llm_config: {inherit: 0, model: m}", r"`inherit:` must be true or false, got 0"),
    ],
)
def test_bad_agent_field_points_at_its_line(line: str, message: str) -> None:
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(_flow(line))
    assert caught.value.diagnostic.frames[-1].range.start_line == 7


@pytest.mark.parametrize(
    ("line", "attr", "value"),
    [
        ("    tools: [search, fetch]", "tools", ["search", "fetch"]),
        ("    tools: []", "tools", []),
        ("    retries: 0", "retries", 0),
        ("    retries: 5", "retries", 5),
        ("    llm_config: {inherit: false}", "llm_inherit", False),
        ("    llm_config: {inherit: true, model: m}", "llm_inherit", True),
        ("", "retries", 2),
    ],
)
def test_good_agent_field_loads(line: str, attr: str, value: object) -> None:
    assert getattr(load_flow(_flow(line)).compiled.nodes["a"], attr) == value


def test_adaptive_questions_retries_is_checked_at_load() -> None:
    text = """\
id: gate
name: gate
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask what you need.
      retries: -1
output: ${ask.output}
"""
    message = r"`adaptive_questions.retries:` must be a non-negative integer, got -1"
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(text)
    assert caught.value.diagnostic.frames[-1].range.start_line == 7
