"""The run boundary accepts only declared input names and a mapping of inputs."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from yian import load_flow, run_flow

FLOW = """\
id: hello
name: hello
input:
  topic: !str "world"
nodes:
  greet:
    kind: code
    input:
      t: ${input.topic}
    output: str
    code: |
      return "hello " + inputs["t"]
output: ${greet.output}
"""


def test_undeclared_input_is_rejected_before_the_log(tmp_path) -> None:
    flow = load_flow(FLOW)
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match=r"unknown input 'topik'; declared inputs: topic"):
        run_flow(flow, {"topik": "Ada"}, log=log)
    assert not log.exists()


def test_flow_without_inputs_names_none_declared() -> None:
    flow = load_flow(FLOW.replace('input:\n  topic: !str "world"\n', "").replace(
        "${input.topic}", "x"
    ))
    with pytest.raises(ValueError, match=r"unknown inputs 'a', 'b'; declared inputs: \(none\)"):
        run_flow(flow, {"b": 1, "a": 2})


@pytest.mark.parametrize("bad", [None, [("topic", "x")], "topic"])
def test_non_mapping_inputs_raise_type_error(bad) -> None:
    with pytest.raises(TypeError, match="inputs must be a mapping"):
        run_flow(load_flow(FLOW), bad)


def test_declared_inputs_in_any_mapping_run() -> None:
    flow = load_flow(FLOW)
    assert run_flow(flow, MappingProxyType({"topic": "Ada"})).output == "hello Ada"
    assert run_flow(flow, {}).output == "hello world"
