"""A bad adaptive brief is reported on the gate at the brief."""

from __future__ import annotations

import pytest

from yian.compose import LoadError, load_flow

# The gate's `prompt:` is on line 8 and the brief on line 10; without the gate
# `prompt:`, the brief moves up to line 9.
_FLOW = """\
id: f
name: f
input: {{proj: !str }}
nodes:
  ask:
    kind: human_input
    input: {{proj: '${{input.proj}}'}}
{gate_prompt}    adaptive_questions:
      prompt: "{brief}"
output: ${{ask.output}}
"""

_GATE_PROMPT = '    prompt: "Pick one for ${proj}"\n'

_SITE = "node 'ask' adaptive_questions.prompt: "


def _flow(brief: str, *, gate_prompt: str = _GATE_PROMPT) -> str:
    return _FLOW.format(brief=brief, gate_prompt=gate_prompt)


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


@pytest.mark.parametrize(
    ("gate_prompt", "brief_line"),
    [(_GATE_PROMPT, 10), ("", 9)],
    ids=["gate-prompt", "no-gate-prompt"],
)
@pytest.mark.parametrize(
    ("brief", "fragment"),
    [
        ("Q ${nope}", "${nope} is not a declared input or variable"),
        (
            "Q ${input.proj}",
            "${input.proj} is not a declared input or variable (a pool ref belongs in an "
            "input `from:`, not a strict AGENT prompt)",
        ),
        ("Q ${upper(${proj})}", "write `${upper(proj)}`"),
        ("Q ${proj +}", "could not parse"),
    ],
)
def test_bad_brief_is_named_and_located_at_the_brief(
    gate_prompt, brief_line, brief, fragment
):
    error = _load_error(_flow(brief, gate_prompt=gate_prompt))
    message = str(error)
    line = next(ln for ln in message.splitlines() if _SITE in ln)
    assert line.strip().startswith(_SITE)
    assert fragment in line.split(_SITE, 1)[1]
    assert "ask__compose" not in message
    assert _line(error) == brief_line


def test_brief_and_gate_prompt_errors_are_both_reported():
    error = _load_error(_flow("${b}", gate_prompt='    prompt: "${a}"\n'))
    lines = [
        ln.strip()
        for ln in str(error).splitlines()
        if ln.strip().startswith("node 'ask' ")
    ]
    assert [ln.split(" is not ", 1)[0] for ln in lines] == [
        "node 'ask' adaptive_questions.prompt: ${b}",
        "node 'ask' prompt: ${a}",
    ]
    assert _line(error) == 10


def test_compact_flow_brief_error_names_the_flow_node():
    error = _load_error(
        """\
id: f
name: f
input: {proj: !str }
kind: human_input
adaptive_questions:
  prompt: "Q ${nope}"
"""
    )
    message = str(error)
    assert "node 'f' adaptive_questions.prompt: ${nope}" in message
    assert "f__compose" not in message
    assert _line(error) == 6


def test_aliased_brief_shares_the_anchor_line():
    error = _load_error(
        """\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '${input.proj}'}
    prompt: &p "Q ${nope}"
    adaptive_questions: {prompt: *p}
output: ${ask.output}
"""
    )
    message = str(error)
    assert "node 'ask' prompt: ${nope}" in message
    assert "node 'ask' adaptive_questions.prompt: ${nope}" in message
    assert _line(error) == 8


def test_defs_body_brief_error():
    error = _load_error(
        """\
id: f
name: f
input:
  topic: !str
defs:
  pick:
    input:
      topic: !str
    nodes:
      ask:
        kind: human_input
        input:
          topic: ${input.topic}
        adaptive_questions:
          prompt: "Q ${nope}"
    output: ${ask.output}
nodes:
  run:
    kind: call
    call: pick
    input:
      topic: ${input.topic}
output: ${run.output}
"""
    )
    message = str(error)
    assert message.startswith("defs entry 'pick': ")
    assert "node 'ask' adaptive_questions.prompt: ${nope}" in message
    assert "ask__compose" not in message
    assert _line(error) == 15


def test_a_clean_brief_loads():
    nodes = load_flow(_flow("Q ${proj}")).compiled.nodes
    assert "ask__compose" in nodes
