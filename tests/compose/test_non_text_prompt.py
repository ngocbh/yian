"""A `prompt:` or question text YAML reads as non-text is a load error at that value."""

from __future__ import annotations

import re

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError

NON_TEXT = [
    ("5", "an integer"),
    ("0", "an integer"),
    ("true", "a boolean"),
    ("false", "a boolean"),
    ("1.5", "a float"),
    ("2026-10-14", "a date"),
    ("2026-10-14 10:00:00", "a date and time"),
    ("[a, b]", "a list"),
    ("[]", "a list"),
    ("{x: 1}", "a mapping"),
    ("{}", "a mapping"),
]

TEXT = ['"5"', "'[a, b]'", "Hello ${proj}"]


def _agent(value: str) -> str:
    """An AGENT flow whose `prompt:` is on line 7."""
    return f"""\
id: f
name: f
input: {{proj: str}}
nodes:
  a:
    kind: agent
    prompt: {value}
    input: {{proj: '${{input.proj}}'}}
output: ${{a.output}}
"""


def _gate(value: str) -> str:
    """A HUMAN_INPUT flow whose `prompt:` is on line 7."""
    return f"""\
id: f
name: f
input: {{proj: str}}
nodes:
  ask:
    kind: human_input
    prompt: {value}
    input: {{proj: '${{input.proj}}'}}
output: ${{ask.output}}
"""


def _adaptive(value: str) -> str:
    """A HUMAN_INPUT flow whose `adaptive_questions.prompt:` brief is on line 8."""
    return f"""\
id: f
name: f
input: {{proj: str}}
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: {value}
    input: {{proj: '${{input.proj}}'}}
output: ${{ask.output}}
"""


def _compact(kind: str, value: str) -> str:
    """A compact single-node flow of `kind` whose `prompt:` is `value`."""
    return f"""\
id: reply
name: reply
input:
  proj: str
output: str
kind: {kind}
prompt: {value}
"""


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_agent_prompt_is_a_load_error(value: str, phrase: str) -> None:
    error = _load_error(_agent(value))
    assert (
        f"node 'a' (kind=agent): `prompt:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _line(error) == 7


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_gate_prompt_is_a_load_error(value: str, phrase: str) -> None:
    error = _load_error(_gate(value))
    assert (
        f"node 'ask' (kind=human_input): `prompt:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _line(error) == 7


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_adaptive_brief_is_a_load_error(value: str, phrase: str) -> None:
    error = _load_error(_adaptive(value))
    assert (
        "node 'ask' (kind=human_input): `adaptive_questions.prompt:` must be text, but YAML "
        f"read it as {phrase}; put it in quotes"
    ) in str(error)
    assert "requires a `prompt:`" not in str(error)
    assert _line(error) == 8


def test_adaptive_brief_error_comes_before_exclusivity() -> None:
    error = _load_error("""\
id: f
name: f
nodes:
  ask:
    kind: human_input
    questions:
      - question: Which?
    adaptive_questions: {prompt: 5}
output: ${ask.output}
""")
    assert "`adaptive_questions.prompt:` must be text" in str(error)
    assert "mutually exclusive" not in str(error)


@pytest.mark.parametrize("kind", ["agent", "human_input"])
def test_compact_prompt_must_be_text(kind: str) -> None:
    error = _load_error(_compact(kind, "5"))
    assert (
        f"node 'reply' (kind={kind}): `prompt:` must be text, but YAML read it as an integer"
    ) in str(error)


def test_merged_prompt_must_be_text() -> None:
    error = _load_error("""\
id: f
name: f
x-base: &b {prompt: 5}
nodes:
  a:
    kind: agent
    <<: *b
output: ${a.output}
""")
    assert re.search(r"node 'a' \(kind=agent\): `prompt:` must be text", str(error))
    assert _line(error) == 3


def test_defs_body_prompt_must_be_text() -> None:
    error = _load_error("""\
id: f
name: f
input:
  topic: str
defs:
  enrich:
    input:
      topic: str
    nodes:
      summarize:
        kind: agent
        input:
          topic: ${input.topic}
        output: str
        prompt: 5
    output: ${summarize.output}
nodes:
  brief:
    kind: call
    call: enrich
    input:
      topic: ${input.topic}
output: ${brief.output}
""")
    assert "`prompt:` must be text" in str(error)


@pytest.mark.parametrize("builder", [_agent, _gate, _adaptive])
@pytest.mark.parametrize("value", TEXT)
def test_text_prompts_still_load(builder, value: str) -> None:
    load_flow(builder(value))


@pytest.mark.parametrize("value", ["null", "''"])
def test_empty_agent_prompt_is_unchanged(value: str) -> None:
    load_flow(_agent(value))
