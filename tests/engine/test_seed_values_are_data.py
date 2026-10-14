"""CALL / MAP / LOOP hand their child resolved values, never templates.

A spawner binds its inputs before it grows, so a string that reaches the child START is data:
text that contains `${...}` (a shell script, a JS template literal, an LLM answer) arrives
unchanged and is never evaluated a second time one level down.
"""

import pytest

from yian.compile.expand import clone_child
from yian.compose import load_flow
from yian.run import run_flow


def _flow(source: str):
    return load_flow(source, source_label="flow.yaml")


_CALL = """
id: seed-call
name: seed_call
input:
  text: !str
nodes:
  probe:
    kind: code
    output: str
    code: |
      return "SECRET"
  echo:
    kind: call
    input:
      text: ${input.text}
    nodes:
      inner:
        kind: code
        input:
          text: ${input.text}
        output: str
        code: |
          return inputs["text"]
    output: ${inner.output}
output: ${echo.output}
"""


@pytest.mark.parametrize("text", [
    "${probe.output}",
    "hi ${probe.output}",
    "run=${system.run_id}",
    "costs ${5",
    "${input.text}",
    "${nope.output}",
    "$$ and $",
])
def test_call_passes_dollar_brace_text_through_unchanged(text):
    result = run_flow(_flow(_CALL), {"text": text})
    assert result.status == "succeeded", result.failure
    assert result.output == text


def test_call_passes_code_output_with_dollar_brace_through_unchanged():
    result = run_flow(_flow("""
id: seed-call-code
name: seed_call_code
nodes:
  probe:
    kind: code
    output: str
    code: |
      return "SECRET"
  mk:
    kind: code
    output: str
    code: |
      return "echo ${probe.output}"
  echo:
    kind: call
    input:
      text: ${mk.output}
    nodes:
      inner:
        kind: code
        input:
          text: ${input.text}
        output: str
        code: |
          return inputs["text"]
    output: ${inner.output}
output: ${echo.output}
"""), {})
    assert result.status == "succeeded", result.failure
    assert result.output == "echo ${probe.output}"


def test_map_passes_dollar_brace_items_through_unchanged():
    result = run_flow(_flow("""
id: seed-map
name: seed_map
nodes:
  probe:
    kind: code
    output: str
    code: |
      return "SECRET"
  mk:
    kind: code
    output: list(str)
    code: |
      return ["a ${probe.output}", "${probe.output}", "plain"]
  each:
    kind: map
    over: ${mk.output}
    input:
      text: ${item}
    nodes:
      inner:
        kind: code
        input:
          text: ${input.text}
        output: str
        code: |
          return inputs["text"]
    output: ${inner.output}
output: ${each.output}
"""), {})
    assert result.status == "succeeded", result.failure
    assert result.output == ["a ${probe.output}", "${probe.output}", "plain"]


def test_loop_carries_dollar_brace_text_through_unchanged():
    result = run_flow(_flow("""
id: seed-loop
name: seed_loop
nodes:
  probe:
    kind: code
    output: str
    code: |
      return "SECRET"
  mk:
    kind: code
    output: str
    code: |
      return "a ${probe.output}"
  lp:
    kind: loop
    times: 2
    input:
      text: ${mk.output}
    nodes:
      inner:
        kind: code
        input:
          text: ${input.text}
        output: str
        code: |
          return inputs["text"]
    output:
      text: ${inner.output}
output: ${lp.output}
"""), {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"text": "a ${probe.output}"}


def test_clone_child_mints_no_edge_for_a_dollar_brace_seed():
    child = _flow("""
id: seed-clone
name: seed_clone
input:
  text: !str
  other: !str
nodes:
  inner:
    kind: code
    input:
      text: ${input.text}
      other: ${input.other}
    output: str
    code: |
      return inputs["text"]
output: ${inner.output}
""").compiled
    cloned = clone_child(
        child, callsite="echo",
        record={"text": "${probe.output}", "other": "${probe.output}"},
    )
    assert not any(e.from_ == "probe" for e in cloned.edges)
