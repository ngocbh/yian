"""A value nesting deeper than 100 levels fails where it enters a run, naming the path."""

import sys

import pytest

from yian import load_flow, run_flow
from yian.memory_host import build_memories
from yian.nodes.binding import BindingError, apply_defaults, bind_params
from yian.nodes.contracts import Parameter
from yian.run import resume_flow
from yian.typesys import TypeCheckError
from yian.typesys.pool import VariablePool


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


# START's committed record may sit one level over the limit, so run_flow's input check is
# what keeps a run input record from going further.
@pytest.mark.parametrize("levels", [101, 1500])
def test_a_run_input_over_the_limit_is_rejected_before_the_run_starts(tmp_path, levels):
    flow = load_flow("""
id: nesting
name: nesting
input:
  x: object
nodes: {}
output: ${input.x}
""")
    log = tmp_path / "r.jsonl"

    with pytest.raises(ValueError) as caught:
        run_flow(flow, {"x": _nest_dict(levels)}, log=log)

    assert isinstance(caught.value, TypeCheckError)
    assert str(caught.value).startswith("input 'x' at a.a")
    assert "deeper than 100 levels" in str(caught.value)
    assert len(caught.value.path) == 100
    assert not log.exists()


def _assert_binding_too_deep(result, node_id: str, prefix: str) -> None:
    """Assert `node_id` failed `binding.evaluate` on value depth with a `prefix` message."""
    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == node_id
    assert result.failure.message.startswith(prefix)
    assert "deeper than 100 levels" in result.failure.message


_WRAPPED_BINDING_FLOW = """
id: nesting
name: nesting
input:
  x: object
nodes:
  c:
    kind: code
    input:
      y: ${[input.x]}
    code: |
      return 1
output: ${c.output}
"""


def test_a_binding_around_a_value_at_the_limit_fails_its_node():
    result = run_flow(load_flow(_WRAPPED_BINDING_FLOW), {"x": _nest_dict(100)})

    _assert_binding_too_deep(result, "c", "input 'y' at [0].a")
    assert result.failure.trace[0].key == "/s:root/s:node/s:c/s:input/s:y/s:binding"


def _call_flow(binding: str, declared: str) -> str:
    """Return a flow whose `k` CALL node binds the def input `y: declared` to `binding`."""
    return f"""
id: nesting
name: nesting
input:
  x: object
defs:
  body:
    input:
      y: {declared}
    nodes:
      inner:
        kind: code
        input:
          y: ${{input.y}}
        code: |
          return 1
    output: ${{inner.output}}
nodes:
  k:
    kind: call
    call: body
    input:
      y: {binding}
output: ${{k.output}}
"""


def test_a_call_bound_to_a_value_at_the_limit_succeeds():
    result = run_flow(load_flow(_call_flow("${input.x}", "object")), {"x": _nest_dict(100)})

    assert result.status == "succeeded"
    assert result.output == 1


def test_a_call_bound_over_the_limit_fails_the_call_not_its_start():
    result = run_flow(load_flow(_call_flow("${[input.x]}", "list[object]")), {"x": _nest_dict(100)})

    _assert_binding_too_deep(result, "k", "input 'y' at [0].a")


def test_a_map_over_list_around_a_value_at_the_limit_fails_the_map():
    flow = load_flow("""
id: nesting
name: nesting
input:
  x: object
nodes:
  m:
    kind: map
    over: ${[input.x]}
    input:
      y: object = ${item}
    nodes:
      work:
        kind: code
        code: |
          return 1
    output: ${work.output}
output: ${m.output}
""")
    result = run_flow(flow, {"x": _nest_dict(100)})

    _assert_binding_too_deep(result, "m", "MAP node 'm': `over`")
    assert result.failure.trace[0].key == "/s:root/s:node/s:m/s:reserved/s:over/s:binding"


def test_apply_defaults_rejects_a_default_over_the_limit():
    with pytest.raises(BindingError) as caught:
        apply_defaults([Parameter("p", has_default=True, default=_nest_dict(101))], {})

    assert caught.value.failure_code == "binding.evaluate"
    assert str(caught.value).startswith("default for 'p' at a.a")
    assert "deeper than 100 levels" in str(caught.value)


def test_bind_params_rejects_a_filled_default_over_the_limit():
    parameter = Parameter("p", has_default=True, default=_nest_dict(101))

    with pytest.raises(BindingError) as caught:
        bind_params([parameter], {}, VariablePool())

    assert caught.value.failure_code == "binding.evaluate"
    assert str(caught.value).startswith("default for 'p' at a.a")
    assert "deeper than 100 levels" in str(caught.value)


def test_backward_rejects_feedback_over_the_limit_and_stays_usable():
    flow = load_flow(_code_flow(_list_body(1)))
    result = run_flow(flow, {}, learnable=True)
    assert result.status == "succeeded"

    with pytest.raises(ValueError) as caught:
        result.backward(_nest_list(1500, "bad"))

    assert str(caught.value).startswith("feedback at [0][0]")
    assert "deeper than 100 levels" in str(caught.value)
    assert result.backward_outcome is None
    outcome = result.backward("good")
    assert outcome.status == "succeeded"


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


def _memory_flow(text: str):
    """Return `text` loaded with episodic long-term memory declared on the flow."""
    return load_flow(text.replace("name: nesting\n", "name: nesting\nlong_term_memory: episodic\n", 1))


def _backward_with_less_stack(result, feedback):
    """Run `result.backward(feedback)` with 300 fewer frames of recursion headroom.

    A caller may already sit deep in its own stack, so learning from a value at the
    limit must not need the remaining headroom of a shallow test frame.
    """
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(limit - 300)
    try:
        return result.backward(feedback)
    finally:
        sys.setrecursionlimit(limit)


def _assert_learns(flow, inputs, feedback) -> None:
    """Assert a memory run on `inputs` succeeds and learns from `feedback`."""
    result = run_flow(flow, inputs, memory=build_memories(flow), learnable=True)
    assert result.status == "succeeded"

    outcome = _backward_with_less_stack(result, feedback)

    assert outcome.status == "succeeded"
    assert outcome.learning.status == "succeeded"


def test_value_at_the_limit_learns_from_an_output():
    _assert_learns(_memory_flow(_code_flow(_list_body(100))), {}, "good")


def test_value_at_the_limit_learns_from_an_input():
    flow = _memory_flow("""
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
    _assert_learns(flow, {"x": _nest_dict(100)}, "good")


def test_value_at_the_limit_learns_from_a_record_output():
    flow = _memory_flow(_code_flow(
        _tree_body(99), output_line=_TREE_OUTPUT, flow_output="${deep.output.tree}",
    ))
    _assert_learns(flow, {}, {"a": "bad"})
