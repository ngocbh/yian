"""An int over 4300 digits fails at the first binding or output boundary it crosses."""

import re

import pytest

from yian import load_flow, run_flow
from yian.compile.model import CompiledFlow, Edge
from yian.nodes.binding import BindingError, apply_defaults
from yian.nodes.contracts import Parameter, PartialRecordContract, ValueContract
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.run import resume_flow
from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
from yian.typesys import Type, ValueKind
from tests.engine._fakes import FuncNode
from tests.engine.test_run_state_outcomes import _commit_start_success, _initialized_and_started


BIG = 10**4999 + 12345  # 5000 digits
LIMIT_MSG = "integer with 5000 digits is larger than yian allows (4300 digits)"


def _assert_clean(message: str) -> None:
    """Check a message shows no digits of the long int and no Python limit hint."""
    assert "set_int_max_str_digits" not in message
    assert re.search(r"[0-9]{600}", message) is None


def _run(text: str, tmp_path):
    """Run the flow `text` with `n = 4999`, check its log replays, and return the result."""
    flow = load_flow(text)
    path = tmp_path / "run.jsonl"
    result = run_flow(flow, {"n": 4999}, log=path)
    replayed = resume_flow(flow, log=path)
    assert replayed.status == result.status
    if result.failure is not None:
        assert replayed.failure.message == result.failure.message
    return result


def _assert_fails(result, code: str, node_id: str, message: str) -> None:
    """Assert the run failed `node_id` with `code` and exactly `message`, without digits."""
    assert result.status == "failed"
    assert result.failure.code == code
    assert result.failure.node_id == node_id
    assert result.failure.message == message
    _assert_clean(result.failure.message)


def _code_input_flow(binding: str) -> str:
    """Return a flow whose `c` CODE node binds its input `x` to `binding`."""
    return f"""
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: {binding}
    code: |
      return 1
output: ${{c.output}}
"""


@pytest.mark.parametrize("binding, message", [
    ("${10 ** input.n}", "input 'x': " + LIMIT_MSG),
    ("!str ${10 ** input.n}", "input 'x': " + LIMIT_MSG),
    ("${[1, 10 ** input.n]}", "input 'x' at [1]: " + LIMIT_MSG),
    ("{a: !str '${10 ** input.n}'}", "binding 'x.a': " + LIMIT_MSG),
], ids=["whole", "str-input", "list-item", "nested-typed"])
def test_a_long_int_binding_fails_its_node(binding, message, tmp_path):
    result = _run(_code_input_flow(binding), tmp_path)

    _assert_fails(result, "binding.evaluate", "c", message)


def test_a_long_int_default_is_rejected():
    with pytest.raises(BindingError) as caught:
        apply_defaults([Parameter("x", has_default=True, default=BIG)], {})

    assert caught.value.failure_code == "binding.evaluate"
    assert str(caught.value) == "default for 'x': " + LIMIT_MSG
    _assert_clean(str(caught.value))


def test_a_long_int_default_is_rejected_before_its_contract():
    parameter = Parameter(
        "x", ValueContract(Type(ValueKind.STRING)), has_default=True, default=BIG
    )

    with pytest.raises(BindingError) as caught:
        apply_defaults([parameter], {})

    assert str(caught.value) == "default for 'x': " + LIMIT_MSG


def _code_output_flow(value: str, output_line: str = "") -> str:
    """Return a flow whose `c` CODE node returns `value`."""
    return f"""
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
{output_line}
    code: |
      return {value}
output: ${{c.output}}
"""


@pytest.mark.parametrize("value, output_line, message", [
    ("10 ** 4999", "", "output: " + LIMIT_MSG),
    ("10 ** 4999", "    output: int", "output: " + LIMIT_MSG),
    ("10 ** 4999", "    output: str", "output: " + LIMIT_MSG),
    ('{"v": [10 ** 4999]}', "", "output at v[0]: " + LIMIT_MSG),
    ('{"v": [10 ** 4999]}', "    output: {v: list(int)}", "output at v[0]: " + LIMIT_MSG),
], ids=["untyped", "int", "str", "record-untyped", "record-typed"])
def test_a_long_int_node_output_fails_the_node(value, output_line, message, tmp_path):
    result = _run(_code_output_flow(value, output_line), tmp_path)

    _assert_fails(result, "contract.output", "c", message)


def _partial_record_flow() -> CompiledFlow:
    """Return START -> `work` -> END where `work` declares a partial record output."""
    work = FuncNode("work", lambda inputs: inputs)
    work.output_contract = PartialRecordContract(
        fields=(
            ("items", ValueContract(Type(kind=ValueKind.LIST_INTEGER))),
            ("label", ValueContract(Type(kind=ValueKind.STRING))),
            ("note", None),
        )
    )
    return CompiledFlow.from_parts(
        nodes={
            "__start__": StartNode("__start__"),
            "work": work,
            "__end__": EndNode.value("__end__", output_name=None),
        },
        edges=[
            Edge(id="start-work", from_="__start__", to="work"),
            Edge(id="work-end", from_="work", to="__end__"),
        ],
    )


@pytest.mark.parametrize("output, path", [
    ({"items": [1, BIG], "label": "a", "note": "x"}, "items[1]"),
    ({"items": [1], "label": BIG, "note": "x"}, "label"),
    ({"items": [1], "label": "a", "note": {"deep": BIG}}, "note.deep"),
], ids=["declared-field", "mistyped-field", "opaque-field"])
def test_a_long_int_in_a_partial_record_output_names_the_limit(output, path):
    flow = _partial_record_flow()
    state, log = _initialized_and_started(flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "work"))

    with pytest.raises(ValueError) as caught:
        prepare_node_succeeded(state, "work", output)

    assert str(caught.value) == f"output at {path}: " + LIMIT_MSG
    _assert_clean(str(caught.value))


def test_a_long_int_flow_output_fails_the_end_binding(tmp_path):
    result = _run("""
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    code: |
      return 1
output: ${10 ** input.n}
""", tmp_path)

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.message == "flow output: " + LIMIT_MSG
    _assert_clean(result.failure.message)


def test_a_long_int_value_entry_fails_the_entry(tmp_path):
    result = _run("""
id: long
name: long
input:
  n: !int
nodes:
  v: !int ${10 ** input.n}
output: ${v.output}
""", tmp_path)

    _assert_fails(result, "binding.evaluate", "v", "value entry 'v': " + LIMIT_MSG)


def test_a_long_int_map_over_fails_the_map(tmp_path):
    result = _run("""
id: long
name: long
input:
  n: !int
nodes:
  m:
    kind: map
    over: ${[10 ** input.n]}
    input:
      y: !int ${item}
    nodes:
      work:
        kind: code
        code: |
          return 1
    output: ${work.output}
output: ${m.output}
""", tmp_path)

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "m"
    assert "`over`" in result.failure.message
    assert result.failure.message.endswith("at [0]: " + LIMIT_MSG)
    _assert_clean(result.failure.message)


def test_a_long_int_loop_seed_fails_the_loop(tmp_path):
    result = _run("""
id: long
name: long
input:
  n: !int
defs:
  bump:
    input:
      n: !int
      exited: !bool
    nodes: {}
    output:
      n: ${input.n}
      exited: ${input.exited}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: ${10 ** input.n}
      exited: false
    times: 1
output: ${loop.output}
""", tmp_path)

    _assert_fails(result, "binding.evaluate", "loop", "input 'n': " + LIMIT_MSG)


@pytest.mark.parametrize("expression, expected", [
    ("${10 ** input.n > 0}", True),
    ("${isinstance(10 ** input.n, str)}", False),
    ("${isinstance(10 ** input.n, int)}", True),
], ids=["compare", "isinstance-str", "isinstance-int"])
def test_a_long_int_inside_one_expression_is_allowed(expression, expected, tmp_path):
    result = _run(f"""
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: {expression}
    code: |
      return inputs["x"]
output: ${{c.output}}
""", tmp_path)

    assert result.status == "succeeded", result.failure
    assert result.output == expected


def test_a_constructor_mismatch_on_a_long_int_is_a_located_type_error(tmp_path):
    result = _run("""
id: long
name: long
typedefs:
  Pt:
    x: str
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      p: ${Pt(x=10 ** input.n)}
    code: |
      return 1
output: ${c.output}
""", tmp_path)

    _assert_fails(
        result,
        "binding.type",
        "c",
        "Pt(...): <int with 5000 digits> does not match declared type str",
    )
    assert result.failure.trace[0].key == "/s:root/s:node/s:c/s:input/s:p/s:binding"
