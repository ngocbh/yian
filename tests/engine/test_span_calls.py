"""A builtin called with arguments it doesn't take fails at load in every binding and prompt,
and before the run for one set in Python."""

import pytest

import tests.engine._compose_codefns as fns
import yian.run
from yian.compose import LoadError, load_flow
from yian.expr import ExpressionError
from yian.nodes.base import _HostStop
from yian.nodes.contracts import Binding, ValueTemplate
from yian.run import resume_flow, run_flow
from yian.typesys import Type, ValueKind

_UPPER = "expression function 'upper' takes (s): too many positional arguments"


def _load_error(text: str, located: str | None = None) -> str:
    """Load `text`, expecting a `LoadError` whose range covers `located` when given."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    rng = caught.value.diagnostic.frames[-1].range
    assert rng is not None
    if located is not None:
        assert rng.start_line == rng.end_line
        assert located in text.splitlines()[rng.start_line - 1][rng.start_col - 1 : rng.end_col]
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

_VALUE_ENTRY = """
id: f
name: f
input: {v: !str }
nodes:
  e: !str "<value>"
output: ${e.output}
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
    "nested-value": (_fill(_CODE, **{**_GOOD_CODE, "n": "x"}).replace("{n: 'x'}", "{n: {a: ['${upper(input.v, 1)}']}}"), "node 'x' input 'n' from: "),
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
    "value-entry": (_fill(_VALUE_ENTRY, value="${upper(input.v, 1)}"), "value entry 'e' from: "),
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
    message = _load_error(text, located="upper(")
    assert [ln.strip() for ln in message.splitlines() if _UPPER in ln] == [f"{where}{_UPPER}"]


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


_UPPER_RUN = r"expression function 'upper' takes \(s\): too many positional arguments$"


def _run_raises(flow, message, tmp_path, inputs=None):
    path = tmp_path / "run.jsonl"
    with pytest.raises(ExpressionError, match=message):
        run_flow(flow, inputs or {"v": "a"}, log=path)
    assert not path.exists()


def test_python_built_binding_mis_calling_a_builtin_fails_before_the_run(tmp_path):
    flow = load_flow(_fill(_CODE, **_GOOD_CODE))
    flow.compiled.wiring["x"]["n"] = Binding("${upper(input.v, 1)}")
    _run_raises(flow, rf"^node 'x' input 'n': {_UPPER_RUN}", tmp_path)


def test_python_built_flow_output_mis_calling_a_builtin_fails_before_the_run(tmp_path):
    flow = load_flow(_fill(_CODE, **_GOOD_CODE))
    flow.compiled.wiring["__end__"]["result"] = Binding("${upper(x.output, 1)}")
    _run_raises(flow, rf"^flow output: {_UPPER_RUN}", tmp_path)


def test_python_built_child_flow_output_names_its_owner(tmp_path):
    flow = load_flow(_fill(_BODY, inner="${input.w}", w="${input.v}"))
    flow.compiled.nodes["c"].child.wiring["__end__"]["result"] = Binding("${upper(s.output, 1)}")
    _run_raises(flow, rf"^node 'c' flow output: {_UPPER_RUN}", tmp_path)


def test_python_built_child_binding_names_its_static_path(tmp_path):
    flow = load_flow(_fill(_BODY, inner="${input.w}", w="${input.v}"))
    flow.compiled.nodes["c"].child.wiring["s"]["n"] = Binding("${upper(input.w, 1)}")
    _run_raises(flow, rf"^node 'c/s' input 'n': {_UPPER_RUN}", tmp_path)


def test_python_built_reserved_and_synthetic_keys_use_their_site_labels(tmp_path):
    case = load_flow(_fill(_CASE_ON, on="${input.v}"))
    case.compiled.wiring["g"]["__on"] = Binding("${upper(input.v, 1)}")
    _run_raises(case, rf"^node 'g' on: {_UPPER_RUN}", tmp_path)

    mapped = load_flow(_fill(_MAP, over="${[input.v]}", item="${item}"))
    mapped.compiled.wiring["m"]["item"] = Binding("${upper(item, 1)}")
    _run_raises(mapped, rf"^node 'm' map input 'item': {_UPPER_RUN}", tmp_path)

    waiting = load_flow(_fill(_WAIT, until="${input.d}"))
    waiting.compiled.wiring["w"]["until"] = Binding("${upper(input.d, 1)}")
    _run_raises(waiting, rf"^node 'w' wait until: {_UPPER_RUN}", tmp_path, {"d": "2026-01-01"})


def test_python_built_reserved_key_constructor_is_checked_without_a_type_table(tmp_path):
    """MAP `over:` is bound with no type table, so a name in the node's table still fails."""
    flow = load_flow(_fill(_MAP, over="${[input.v]}", item="${item}"))
    node = flow.compiled.nodes["m"]
    node.types = {**node.types, "Buy": Type.scalar(ValueKind.STRING)}
    flow.compiled.wiring["m"]["over"] = Binding("${[Buy(input.v)]}")
    _run_raises(flow, r"^node 'm' map over: unknown expression function 'Buy'$", tmp_path)


def test_python_built_map_child_binding_names_its_static_path(tmp_path):
    flow = load_flow(_fill(_MAP, over="${[input.v]}", item="${item}"))
    flow.compiled.nodes["m"].child.wiring["s"]["n"] = Binding("${upper(input.item, 1)}")
    _run_raises(flow, rf"^node 'm/s' input 'n': {_UPPER_RUN}", tmp_path)


def test_python_built_agent_prompt_mis_calling_a_builtin_fails_before_the_run(tmp_path):
    flow = load_flow(_fill(_AGENT, prompt="Say ${t}", system="S"))
    flow.compiled.nodes["a"].prompt = "Say ${upper(t, 1)}"
    _run_raises(flow, rf"^node 'a' prompt: {_UPPER_RUN}", tmp_path)


def test_python_built_question_text_mis_calling_a_builtin_fails_before_the_run(tmp_path):
    flow = load_flow(_fill(_HUMAN, **_GOOD_HUMAN))
    flow.compiled.nodes["ask"].questions[0]["question"] = "${upper(t, 1)}"
    _run_raises(flow, rf"^node 'ask' questions\[0\]\.question: {_UPPER_RUN}", tmp_path)


def test_python_built_binding_calling_an_unknown_function_fails_before_the_run(tmp_path):
    flow = load_flow(_fill(_CODE, **_GOOD_CODE))
    flow.compiled.wiring["x"]["n"] = Binding("${nosuchfn(input.v)}")
    _run_raises(flow, r"^node 'x' input 'n': unknown expression function 'nosuchfn'$", tmp_path)


def test_python_built_nested_binding_mis_calling_a_builtin_fails_before_the_run(tmp_path):
    flow = load_flow(_fill(_CODE, **_GOOD_CODE))
    flow.compiled.wiring["x"]["n"] = Binding(ValueTemplate({"a": ["${upper(input.v, 1)}"]}))
    _run_raises(flow, rf"^node 'x' input 'n': {_UPPER_RUN}", tmp_path)


def test_a_well_called_python_built_binding_still_runs():
    flow = load_flow(_fill(_CODE, **_GOOD_CODE))
    flow.compiled.wiring["x"]["n"] = Binding("${upper(input.v)}")
    assert run_flow(flow, {"v": "a"}).output == "A"


_TWO_NODES = """
id: f
name: f
input:
  v: !int
nodes:
  a:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.v}
    output: int
  b:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${a.output}
    output: int
output: ${b.output}
"""


def test_python_built_binding_mis_calling_a_builtin_fails_before_resume_appends(tmp_path, monkeypatch):
    """The first run skips the pre-run check, as a run logged before the check existed did:
    changing a binding after the run would change the flow fingerprint instead."""
    flow = load_flow(_TWO_NODES)
    flow.compiled.wiring["b"]["n"] = Binding("${render_as_json(a.output, 1, 2)}")
    path = tmp_path / "run.jsonl"

    def stop(inputs):
        raise _HostStop()

    with monkeypatch.context() as m:
        m.setattr(yian.run, "_check_templates", lambda flow, path=(): None)
        m.setattr(fns, "double", stop)  # node `a` stops after its durable start
        with pytest.raises(_HostStop):
            run_flow(flow, {"v": 1}, log=path)
    lines = path.read_text().splitlines()

    with pytest.raises(
        ExpressionError,
        match=r"^node 'b' input 'n': expression function 'render_as_json' takes \(value, indent=2\): "
        r"too many positional arguments$",
    ):
        resume_flow(flow, log=path)
    assert path.read_text().splitlines() == lines
