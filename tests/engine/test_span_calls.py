"""A builtin called with arguments it doesn't take fails at load in every binding and prompt,
and before the run for one set in Python."""

import pytest

from yian.compose import LoadError, load_flow

_UPPER = "expression function 'upper' takes (s): too many positional arguments"


def _load_error(text: str) -> str:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert caught.value.diagnostic.frames[-1].range is not None
    return str(caught.value)


def _fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace(f"<{key}>", value)
    return template


_CODE = """
id: f
name: f
input: {v: !str }
nodes:
  x: {kind: code, code: "return inputs['n']", input: {n: '<n>'}, output: str}
output: <out>
"""

_TYPED = """
id: f
name: f
input: {v: !str }
nodes:
  x:
    kind: code
    code: "return inputs['n']"
    input:
      n: !str <n>
    output: str
output: ${x.output}
"""

_TOOL = """
id: f
name: f
input: {v: !str }
nodes:
  t: {kind: tool, tool_id: echo, input: {n: '<n>'}, output: str}
output: ${t.output}
"""

_BODY = """
id: f
name: f
input: {v: !str }
defs:
  body:
    input: {w: !str }
    nodes:
      s: {kind: code, code: "return inputs['n']", input: {n: '<inner>'}, output: str}
    output: ${s.output}
nodes:
  c: {kind: call, call: body, input: {w: '<w>'}}
output: ${c.output}
"""

_CALL_KWARG = """
id: f
name: f
input: {v: !str }
defs:
  body:
    input: {w: !str }
    nodes:
      s: {kind: code, code: "return inputs['n']", input: {n: '${input.w}'}, output: str}
    output: ${s.output}
nodes:
  x:
    kind: code
    code: "return inputs['n']"
    output: str
    input:
      n: call(body, w=<w>)
output: ${x.output}
"""

_MAP = """
id: f
name: f
input: {v: !str }
defs:
  body:
    input: {item: !str }
    nodes:
      s: {kind: code, code: "return inputs['n']", input: {n: '${input.item}'}, output: str}
    output: ${s.output}
nodes:
  m: {kind: map, over: '<over>', call: body, input: {item: '<item>'}}
output: ${m.output}
"""

_LOOP = """
id: f
name: f
input: {v: !str }
defs:
  step:
    input: {n: !str }
    nodes:
      s: {kind: code, code: "return {'n': inputs['n']}", input: {n: '${input.n}'}, output: {n: str}}
    output: ${s.output}
nodes:
  r: {kind: loop, call: step, input: {n: '<n>'}, until: "true", max: 2}
output: ${r.output}
"""

_WAIT = """
id: f
name: f
input: {d: !str }
nodes:
  w: {kind: wait, until: '<until>'}
output: ok
"""

_CASE_ON = """
id: f
name: f
input: {v: !str }
nodes:
  g:
    kind: case
    on: <on>
    cases:
      - when: "'A'"
        then: hot
    else: cold
  hot: {kind: code, code: "return 'hot'", output: str}
  cold: {kind: code, code: "return 'cold'", output: str}
output: ${hot.output | cold.output}
"""

_CASE_VALUE = """
id: f
name: f
input: {v: !str }
nodes:
  g:
    kind: case
    output: str
    cases:
      - when: input.v == 'a'
        then: "<then>"
    else: "x"
output: ${g.output}
"""

_AGENT = """
id: f
name: f
input: {v: !str }
nodes:
  a: {kind: agent, input: {t: '${input.v}'}, output: str, prompt: '<prompt>', system_prompt: '<system>'}
output: ${a.output}
"""

_HUMAN = """
id: f
name: f
input: {v: !str }
nodes:
  ask:
    kind: human_input
    input: {t: '${input.v}'}
    prompt: '<prompt>'
    questions:
      - question: '<question>'
        header: H
        options:
          - label: '<label>'
            description: '<description>'
output: ${ask.output}
"""

_ADAPTIVE = """
id: f
name: f
input: {v: !str }
nodes:
  ask:
    kind: human_input
    input: {t: '${input.v}'}
    adaptive_questions:
      prompt: '<prompt>'
output: ${ask.output}
"""

_GOOD_CODE = {"n": "${input.v}", "out": "${x.output}"}
_GOOD_HUMAN = {"prompt": "${t}", "question": "${t}", "label": "${t}", "description": "${t}"}

_SITES = {
    "node-input": (_fill(_CODE, **{**_GOOD_CODE, "n": "${upper(input.v, 1)}"}), "node 'x' input 'n' from: "),
    "node-input-embedded": (_fill(_CODE, **{**_GOOD_CODE, "n": "hi ${upper(input.v, 1)}"}), "node 'x' input 'n' from: "),
    "node-input-default": (_fill(_CODE, **{**_GOOD_CODE, "n": "${input.v :- upper(input.v, 1)}"}), "node 'x' input 'n' from: "),
    "ref-free": (_fill(_CODE, **{**_GOOD_CODE, "n": '${upper("a", 1)}'}), "node 'x' input 'n' from: "),
    "typed-binding": (_fill(_TYPED, n="${upper(input.v, 1)}"), "node 'x' input 'n' from: "),
    "tool-input": (_fill(_TOOL, n="${upper(input.v, 1)}"), "node 't' input 'n' from: "),
    "flow-output": (_fill(_CODE, **{**_GOOD_CODE, "out": "${upper(x.output, 1)}"}), "flow output from: "),
    "flow-output-record": (_fill(_CODE, **{**_GOOD_CODE, "out": "{a: '${upper(x.output, 1)}'}"}), "flow output from: "),
    "defs-body-input": (_fill(_BODY, inner="${upper(input.w, 1)}", w="${input.v}"), "node 's' input 'n' from: "),
    "call-input": (_fill(_BODY, inner="${input.w}", w="${upper(input.v, 1)}"), "node 'c' input 'w' from: "),
    "call-kwarg": (_fill(_CALL_KWARG, w="${upper(input.v, 1)}"), "node '__call_0' input 'w' from: "),
    "map-over": (_fill(_MAP, over="${upper(input.v, 1)}", item="${item}"), "node 'm' map over: "),
    "map-item-input": (_fill(_MAP, over="${[input.v]}", item="${upper(item, 1)}"), "node 'm' map input 'item' from: "),
    "loop-seed": (_fill(_LOOP, n="${upper(input.v, 1)}"), "node 'r' input 'n' from: "),
    "wait-until": (_fill(_WAIT, until="${upper(input.d, 1)}"), "node 'w' wait until: "),
    "case-on": (_fill(_CASE_ON, on="${upper(input.v, 1)}"), "node 'g' on: from: "),
    "case-value": (_fill(_CASE_VALUE, then="${upper(input.v, 1)}"), "node '__value_0' input 'v0' from: "),
    "agent-prompt": (_fill(_AGENT, prompt="Say ${upper(t, 1)}", system="S"), "node 'a' prompt: "),
    "agent-system-prompt": (_fill(_AGENT, prompt="Say", system="S ${upper(t, 1)}"), "node 'a' system_prompt: "),
    "adaptive-brief": (_fill(_ADAPTIVE, prompt="Ask about ${upper(t, 1)}"), "node 'ask' adaptive_questions.prompt: "),
    **{
        f"human-{site}": (
            _fill(_HUMAN, **{**_GOOD_HUMAN, site: "${upper(t, 1)}"}),
            f"node 'ask' {label}: ",
        )
        for site, label in [
            ("prompt", "prompt"),
            ("question", "questions[0].question"),
            ("label", "questions[0].options[0].label"),
            ("description", "questions[0].options[0].description"),
        ]
    },
}


@pytest.mark.parametrize("site", sorted(_SITES))
def test_mis_called_builtin_fails_to_load_at_its_site(site):
    text, where = _SITES[site]
    message = _load_error(text)
    line = next(ln for ln in message.splitlines() if _UPPER in ln)
    assert line.strip() == f"{where}{_UPPER}"


def test_unknown_function_in_a_binding_keeps_the_flow_call_message():
    message = _load_error(_fill(_CODE, **{**_GOOD_CODE, "n": "${nosuchfn(input.v)}"}))
    assert "a flow call 'nosuchfn' inside ${...} is not supported" in message


def test_a_binding_reports_both_an_unknown_ref_and_a_bad_call():
    message = _load_error(_fill(_CODE, **{**_GOOD_CODE, "n": "${upper(input.nosuch, 1)}"}))
    lines = [ln.strip() for ln in message.splitlines()]
    assert "node 'x' input 'n' from: reference ${input.nosuch} points to unknown flow input 'nosuch'" in lines
    assert f"node 'x' input 'n' from: {_UPPER}" in lines


def test_well_called_builtins_in_every_site_load():
    load_flow(_fill(_CODE, n="${upper(input.v)}", out="${render_as_json(x.output, indent=0)}"))
    load_flow(_fill(_MAP, over="${[upper(input.v)]}", item="${lower(item)}"))
    load_flow(_fill(_AGENT, prompt="Say ${upper(t)}", system='S ${join([t], sep="-")}'))
    load_flow(_fill(_HUMAN, **{**_GOOD_HUMAN, "question": "${upper(t)}"}))
    load_flow(_fill(_WAIT, until="${lower(input.d)}"))
