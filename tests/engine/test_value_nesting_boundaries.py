"""A node output nesting deeper than 100 levels fails the node, naming the path."""

import pytest

from yian import load_flow, run_flow
from yian.run import resume_flow


def _code_flow(body: str, *, output_line: str = "", flow_output: str = "${deep.output}") -> str:
    """Return a flow whose `deep` CODE node runs the inline `body`."""
    indented = "\n".join(f"      {line}" for line in body.strip().splitlines())
    return f"""
id: nesting
name: nesting
nodes:
  deep:
    kind: code
{output_line}
    code: |
{indented}
output: {flow_output}
"""


def _tree_body(levels: int) -> str:
    """Return a CODE body returning `{"tree": v}` with `v` nested `levels` levels."""
    return f"""
v = "leaf"
for _ in range({levels}):
    v = {{"a": v}}
return {{"tree": v}}
"""


def _list_body(levels: int) -> str:
    """Return a CODE body returning a list chain `levels` levels deep."""
    return f"""
v = "leaf"
for _ in range({levels}):
    v = [v]
return v
"""


def _nest_list(levels: int, leaf: object = "leaf") -> object:
    value: object = leaf
    for _ in range(levels):
        value = [value]
    return value


def _nest_dict(levels: int) -> object:
    value: object = "leaf"
    for _ in range(levels):
        value = {"a": value}
    return value


def _assert_too_deep(result, node_id: str) -> None:
    """Assert the run failed `node_id`'s output contract on value depth."""
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == node_id
    assert "deeper than 100 levels" in result.failure.message
    assert "Circular" not in result.failure.message


_TREE_OUTPUT = "    output: {tree: object}"


def test_a_record_output_at_the_limit_succeeds():
    flow = load_flow(_code_flow(_tree_body(99), output_line=_TREE_OUTPUT))
    result = run_flow(flow, {})

    assert result.status == "succeeded"
    assert result.output == {"tree": _nest_dict(99)}


def test_a_record_output_over_the_limit_fails_at_the_field():
    flow = load_flow(_code_flow(_tree_body(100), output_line=_TREE_OUTPUT))
    result = run_flow(flow, {})

    _assert_too_deep(result, "deep")
    assert result.failure.message.startswith("output at tree.a")
    assert result.failure.trace[0].key == (
        "/s:root/s:node/s:deep/s:output/s:contract/s:path/s:tree"
    )


# Before the limit, 200 levels raised pydantic's "Invalid JSON: recursion limit exceeded" out
# of run_flow, and 1500 failed the inline CODE serializer with "Circular reference detected
# (depth exceeded)".
@pytest.mark.parametrize("levels", [200, 1500])
def test_a_far_too_deep_inline_output_fails_without_raising(levels):
    flow = load_flow(_code_flow(_tree_body(levels), output_line=_TREE_OUTPUT))
    result = run_flow(flow, {})

    _assert_too_deep(result, "deep")


def test_a_far_too_deep_reference_output_fails_on_depth():
    flow = load_flow("""
id: nesting
name: nesting
input:
  n: int
nodes:
  deep:
    kind: code
    input:
      n: ${input.n}
    code: tests.engine._compose_codefns:deep_tree
output: ${deep.output}
""")
    result = run_flow(flow, {"n": 1500})

    _assert_too_deep(result, "deep")
    assert "canonical data nesting" not in result.failure.message


def test_an_untyped_output_over_the_limit_fails():
    result = run_flow(load_flow(_code_flow(_list_body(101))), {})

    _assert_too_deep(result, "deep")


def test_an_untyped_output_at_the_limit_flows_out_whole():
    result = run_flow(load_flow(_code_flow(_list_body(100))), {})

    assert result.status == "succeeded"
    assert result.output == _nest_list(100)


def _map_flow(child_levels: int) -> str:
    """Return a flow whose `m` MAP collects children nested `child_levels` levels."""
    return f"""
id: nesting
name: nesting
nodes:
  m:
    kind: map
    over: "${{[1, 2]}}"
    input:
      y: int = ${{item}}
    nodes:
      work:
        kind: code
        input:
          y: ${{input.y}}
        code: |
          v = inputs["y"]
          for _ in range({child_levels}):
              v = [v]
          return v
    output: ${{work.output}}
output: ${{m.output}}
"""


def test_a_map_result_list_over_the_limit_fails_at_the_map_end():
    result = run_flow(load_flow(_map_flow(100)), {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "m/__end__"
    assert "output at [0]" in result.failure.message
    assert "100 levels" in result.failure.message


def test_a_map_result_list_at_the_limit_succeeds():
    result = run_flow(load_flow(_map_flow(99)), {})

    assert result.status == "succeeded"
    assert result.output == [_nest_list(99, 1), _nest_list(99, 2)]


def test_a_record_flow_output_around_a_value_at_the_limit_fails_at_end():
    flow = load_flow(_code_flow(_list_body(100), flow_output="\n  t: ${deep.output}"))
    result = run_flow(flow, {})

    _assert_too_deep(result, "__end__")


def test_a_run_input_record_may_carry_a_value_at_the_limit():
    flow = load_flow("""
id: nesting
name: nesting
input:
  x: object
nodes:
  echo:
    kind: code
    input:
      x: ${input.x}
    code: |
      return inputs["x"]
output: ${echo.output}
""")
    value = _nest_dict(100)
    result = run_flow(flow, {"x": value})

    assert result.status == "succeeded"
    assert result.output == value


def test_a_run_input_record_over_the_limit_fails_at_start():
    flow = load_flow("""
id: nesting
name: nesting
input:
  x: object
nodes: {}
output: ${input.x}
""")
    result = run_flow(flow, {"x": _nest_dict(101)})

    _assert_too_deep(result, "__start__")
    assert result.failure.message.startswith("output at x.a")


def test_replay_accepts_a_value_recorded_before_the_limit(monkeypatch, tmp_path):
    flow = load_flow(_code_flow(_list_body(120)))
    path = tmp_path / "r.jsonl"
    monkeypatch.setattr("yian.typesys.values.MAX_VALUE_NESTING", 500)
    recorded = run_flow(flow, {}, log=path)
    assert recorded.status == "succeeded"
    monkeypatch.undo()

    replayed = resume_flow(flow, log=path)

    assert replayed.status == "succeeded"
    assert replayed.failure is None
    assert replayed.output == recorded.output == _nest_list(120)
