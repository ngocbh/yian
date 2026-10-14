"""A human_input gate's prompt and question text are checked at load like an AGENT prompt."""

from __future__ import annotations

import pytest

from yian.compose import LoadError, load_flow

_FLOW = """\
id: f
name: f
input: {{proj: !str }}
nodes:
  ask:
    kind: human_input
    input: {{proj: '${{input.proj}}'}}
    prompt: "{prompt}"
    questions:
      - question: "{question}"
        header: Framework
        options:
          - label: "{label}"
            description: "{description}"
output: ${{ask.output}}
"""

GOOD = "${proj}"

_SITES = {
    "prompt": "node 'ask' prompt:",
    "question": "node 'ask' questions[0].question:",
    "label": "node 'ask' questions[0].options[0].label:",
    "description": "node 'ask' questions[0].options[0].description:",
}

_POOL_HINT = " (a pool ref belongs in an input `from:`, not human_input text)"


def _flow(**over: str) -> str:
    fields = dict(prompt=GOOD, question=GOOD, label=GOOD, description=GOOD) | over
    return _FLOW.format(**fields)


def test_a_clean_gate_loads():
    load_flow(_flow())
    load_flow(_flow(prompt="${proj :- 'none'}"))
    load_flow(_flow(prompt="Cost $$5 for ${proj}"))


@pytest.mark.parametrize("site", sorted(_SITES))
@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("${proj +}", "could not parse expression"),
        ("${nosuch(proj)}", "unknown prompt function 'nosuch'"),
        ("${isinstance(proj, str)}", "isinstance is not a prompt function"),
        ("${upper(${proj})}", "write `${upper(proj)}`"),
        ("x ${proj", "unbalanced"),
    ],
)
def test_bad_span_fails_to_load(site, text, fragment):
    with pytest.raises(LoadError) as caught:
        load_flow(_flow(**{site: text}))
    message = str(caught.value)
    line = next(ln for ln in message.splitlines() if _SITES[site] in ln)
    assert line.strip().startswith(f"{_SITES[site]} ")
    assert fragment in line


@pytest.mark.parametrize("site", sorted(_SITES))
@pytest.mark.parametrize("text", ["${input.proj}", "${input.proj :- 'd'}"])
def test_pool_ref_fails_to_load(site, text):
    with pytest.raises(LoadError) as caught:
        load_flow(_flow(**{site: text}))
    assert f"{_SITES[site]} ${{input.proj}} is not a declared input{_POOL_HINT}" in str(
        caught.value
    )


@pytest.mark.parametrize("site", sorted(_SITES))
@pytest.mark.parametrize("text", ["${other}", "${other :- 'x'}"])
def test_undeclared_name_fails_to_load(site, text):
    with pytest.raises(LoadError) as caught:
        load_flow(_flow(**{site: text}))
    message = str(caught.value)
    assert f"{_SITES[site]} ${{other}} is not a declared input" in message
    assert "pool ref" not in message


def test_every_bad_site_is_reported():
    bad = _flow(
        prompt="${nosuch(proj)}",
        question="${other}",
        label="${input.proj}",
        description="${proj +}",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(bad)
    message = str(caught.value)
    for prefix in _SITES.values():
        assert prefix in message


_DEFS_FLOW = """\
id: f
name: f
input:
  proj: !str
defs:
  confirm:
    input:
      proj: !str
    nodes:
      ask:
        kind: human_input
        input:
          proj: ${input.proj}
        prompt: "${nope}"
    output: ${ask.output}
nodes:
  run:
    kind: call
    call: confirm
    input:
      proj: ${input.proj}
output: ${run.output}
"""


def test_gate_in_a_defs_body_is_checked():
    with pytest.raises(LoadError) as caught:
        load_flow(_DEFS_FLOW)
    assert "node 'ask' prompt: ${nope} is not a declared input" in str(caught.value)


def test_prompt_error_points_at_the_prompt_line():
    flow = _flow(prompt="${other}")
    prompt_line = flow.splitlines().index('    prompt: "${other}"') + 1
    assert prompt_line == 8
    with pytest.raises(LoadError) as caught:
        load_flow(flow)
    assert caught.value.diagnostic.frames[-1].range.start_line == prompt_line


@pytest.mark.parametrize(
    ("site", "line"), [("question", 10), ("label", 13), ("description", 14)]
)
@pytest.mark.parametrize("text", ["${upper(${proj})}", "${other}"])
def test_question_text_error_points_at_the_string(site, line, text):
    flow = _flow(**{site: text})
    assert f'"{text}"' in flow.splitlines()[line - 1]
    with pytest.raises(LoadError) as caught:
        load_flow(flow)
    assert caught.value.diagnostic.frames[-1].range.start_line == line


_ADAPTIVE_FLOW = """\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: "${input.proj}"}
    prompt: "${nope}"
    adaptive_questions:
      prompt: "Q for ${proj}"
output: ${ask.output}
"""


def test_adaptive_gate_prompt_is_checked():
    with pytest.raises(LoadError) as caught:
        load_flow(_ADAPTIVE_FLOW)
    assert "node 'ask' prompt: ${nope} is not a declared input" in str(caught.value)
