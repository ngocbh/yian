"""A child flow's agents may not inherit an `llm_config` value read only when a flow runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from yian.compile.llm_cascade import resolve_llm_cascade, resolve_run_llm
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.nodes.contracts import ValueTemplate

_FIX = "pass it to the child as an input and write `${input.model}` in the child"

_DEF_CALL = """\
id: f
name: f
input: {m: !str }
llm_config: {model: ROOT_MODEL}
defs:
  child:
    nodes:
      a:
        kind: agent
        prompt: Go.
CHILD_LLM
    output: ${a.output}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
    llm_config: {model: m0}
  c:
    kind: call
    call: child
    input: {}
    depends_on: [pick]
output: ${c.output}
"""


def _def_call(root_model: str = "'${pick.output}'", child_llm: str = "") -> str:
    return _DEF_CALL.replace("ROOT_MODEL", root_model).replace("CHILD_LLM\n", child_llm)


def _error(text: str, **kwargs) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="f.yaml", **kwargs)
    return caught.value


def _line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


def _line_of(text: str, needle: str) -> int:
    return next(n for n, line in enumerate(text.splitlines(), 1) if needle in line)


def test_a_run_time_flow_value_a_child_agent_would_inherit_fails_at_the_flow_field():
    text = _def_call()
    error = _error(text)
    message = error.diagnostic.message
    assert message.startswith("flow llm_config.model: "), message
    assert "agent 'c/a'" in message
    assert _FIX in message
    assert _line(error) == _line_of(text, "llm_config: {model: '${pick.output}'}")


def test_a_child_agent_that_sets_the_field_loads():
    load_flow(_def_call(child_llm="        llm_config: {model: m1}\n"))


def test_a_child_agent_that_does_not_inherit_loads():
    load_flow(_def_call(child_llm="        llm_config: {inherit: false, model: m1}\n"))


def test_a_root_input_value_above_a_child_loads():
    load_flow(_def_call(root_model="'${input.m}'"))


def test_a_def_input_value_above_its_own_nested_call_fails():
    text = """\
id: f
name: f
defs:
  inner:
    nodes:
      a: {kind: agent, prompt: Go.}
    output: ${a.output}
  outer:
    input: {m: !str }
    llm_config: {model: '${input.m}'}
    nodes:
      i: {kind: call, call: inner, input: {}}
    output: ${i.output}
nodes:
  o:
    kind: call
    call: outer
    input: {m: x}
output: ${o.output}
"""
    error = _error(text)
    message = error.diagnostic.message
    assert message.startswith("defs entry 'outer' llm_config.model (called by node 'o'): "), (
        message
    )
    assert "agent 'o/i/a'" in message and _FIX in message
    assert _line(error) == _line_of(text, "${input.m}")


_LIB = """\
id: lib
name: lib
input: {m: !str }
llm_config: {model: '${input.m}'}
defs:
  inner:
    nodes:
      a: {kind: agent, prompt: Go.}
    output: ${a.output}
nodes:
  c: {kind: call, call: inner, input: {}}
output: ${c.output}
"""

_IMPORTER = """\
id: parent
name: parent
uses:
  mylib: lib
nodes:
  n:
    kind: call
    call: mylib
    input: {m: x}
output: ${n.output}
"""


def test_an_imported_flows_input_value_above_its_nested_call_fails_at_the_import(
    tmp_path: Path,
):
    (tmp_path / "lib.yaml").write_text(_LIB)
    error = _error(_IMPORTER, search_paths=[tmp_path])
    message = error.diagnostic.message
    assert message.startswith(
        "llm_config.model of the flow imported through uses: 'mylib' (called by node 'n'): "
    ), message
    assert "agent 'n/c/a'" in message and _FIX in message
    frame = error.diagnostic.frames[-1]
    assert frame.label == "lib.yaml"
    assert frame.range.start_line == _line_of(_LIB, "${input.m}")
    # Loaded alone, the file is the run's root, so its input is known at run start.
    load_flow(_LIB, source_label="lib.yaml")


def test_an_inline_call_body_inherits_like_a_named_child():
    text = """\
id: f
name: f
llm_config: {model: '${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
    llm_config: {model: m0}
  c:
    kind: call
    input: {}
    nodes:
      a: {kind: agent, prompt: Go., output: str}
    output: ${a.output}
    depends_on: [pick]
output: ${c.output}
"""
    message = _error(text).diagnostic.message
    assert message.startswith("flow llm_config.model: "), message
    assert "agent 'c/a'" in message


def test_an_inline_loop_body_inherits_like_a_named_child():
    text = """\
id: f
name: f
llm_config: {model: '${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
    llm_config: {model: m0}
  repeat:
    kind: loop
    times: 2
    input: {t: start}
    nodes:
      a:
        kind: agent
        input: {t: '${input.t}'}
        prompt: Go ${t}.
        output: str
    output: {t: '${a.output}'}
    depends_on: [pick]
output: ${repeat.output}
"""
    message = _error(text).diagnostic.message
    assert message.startswith("flow llm_config.model: "), message
    assert "agent 'repeat/a'" in message


def test_a_flow_built_around_compose_hits_the_cascade_check():
    flow = load_flow(_def_call(root_model="m0")).compiled
    flow.flow_llm_config = ValueTemplate({"model": "${pick.output}"})
    with pytest.raises(ValueError, match=r"node 'c/a' would inherit llm_config\.model") as caught:
        resolve_llm_cascade(flow, {}, defaults={"provider": "openai", "model": "gpt-x"})
    assert _FIX in str(caught.value)


def test_a_root_input_value_or_a_literal_passes_the_cascade_check():
    defaults = {"provider": "openai", "model": "gpt-x"}
    flow = load_flow(_def_call(root_model="m0")).compiled
    assert resolve_llm_cascade(flow, {}, defaults=defaults)["c/a"]["model"] == "m0"
    flow.flow_llm_config = ValueTemplate({"model": "${input.m}"})
    run = resolve_run_llm(flow, {}, defaults=defaults, run_inputs={"m": "gpt-y"})
    assert run.effective["c/a"]["model"] == "gpt-y"


_TYPED_CHILD = """\
id: f
name: f
llm_config: {model: '${pick.output}'}
defs:
  child:
    input: {model: MODEL_TAG }
    llm_config: {model: '${input.model}'}
    nodes:
      a: {kind: agent, prompt: Go.}
    output: ${a.output}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
    llm_config: {model: m0}
  c:
    kind: call
    call: child
    input: {model: x}
    depends_on: [pick]
output: ${c.output}
"""


def test_a_child_flow_value_that_cannot_be_null_shields_the_outer_value():
    load_flow(_TYPED_CHILD.replace("MODEL_TAG", "!str"))


def test_a_child_flow_value_that_may_be_null_leaves_the_outer_value_reaching():
    message = _error(_TYPED_CHILD.replace("MODEL_TAG", "!Optional(str)")).diagnostic.message
    assert message.startswith("flow llm_config.model: "), message
    assert "agent 'c/a'" in message
