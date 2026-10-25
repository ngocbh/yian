"""A condition may call only a builtin, with arguments it takes, or a typedef constructor."""

import pytest

from yian.compose import LoadError, load_flow
from yian.expr import ExpressionError
from yian.expr.builtins import TEMPLATE_FNS
from yian.expr.expressions import parse_condition
from yian.nodes.contracts import Assertion
from yian.run import run_flow
from yian.typesys.types import read_typedefs, type_for


@pytest.mark.parametrize(
    "condition",
    ["nosuchfn(n) > 0", "${nosuchfn(n) > 0}", "true or nosuchfn(n)", "upper(nosuchfn(n)) == 'A'"],
)
def test_unknown_function_does_not_parse_as_a_condition(condition):
    with pytest.raises(ExpressionError, match=r"^unknown expression function 'nosuchfn'$"):
        parse_condition(condition, {})


def test_a_constructor_parses_when_its_type_is_known_or_left_to_compose():
    buy = type_for("Buy", read_typedefs({"Buy": {"qty": "int"}}))
    parse_condition("Buy(d).qty > 0", {"Buy": buy})
    # Without a type table, compose resolves the name against the flow's typedefs.
    parse_condition("Buy(d).qty > 0")
    with pytest.raises(ExpressionError, match=r"^unknown expression function 'Buy'$"):
        parse_condition("Buy(d).qty > 0", {})


def test_a_builtin_type_name_called_in_a_condition_gets_the_conversion_hint():
    with pytest.raises(ExpressionError, match=r"`int\(\.\.\.\)` is not a conversion"):
        parse_condition("int(x) > 0", {})


@pytest.mark.parametrize(
    "condition, message",
    [
        (
            "upper(a, b) == 'A'",
            r"^expression function 'upper' takes \(s\): too many positional arguments$",
        ),
        (
            "upper() == 'A'",
            r"^expression function 'upper' takes \(s\): missing a required argument: 's'$",
        ),
        (
            "render_as_json(a, width=4) != ''",
            r"^expression function 'render_as_json' takes \(value, indent=2\): "
            r"got an unexpected keyword argument 'width'$",
        ),
    ],
)
def test_builtin_with_arguments_it_does_not_take_does_not_parse(condition, message):
    with pytest.raises(ExpressionError, match=message):
        parse_condition(condition)


@pytest.mark.parametrize(
    "condition",
    [
        "upper(s) == 'A'",
        "render_as_json(a, indent=4) != ''",
        "join(xs) == 'a'",
        "join(xs, sep=',') == 'a'",
        "isinstance(x, int)",
        "lower(upper(s)) == 'a'",
    ],
)
def test_builtin_calls_that_bind_parse(condition):
    parse_condition(condition)


def test_a_registered_builtin_is_accepted(monkeypatch):
    monkeypatch.setitem(TEMPLATE_FNS, "shout", lambda s, times=1: s.upper() * times)
    parse_condition("shout(s, times=2) == 'AA'")
    with pytest.raises(ExpressionError, match=r"takes \(s, times=1\): too many positional"):
        parse_condition("shout(s, 2, 3) == 'AA'")


def test_a_repeated_keyword_does_not_parse():
    with pytest.raises(
        ExpressionError, match=r"^upper\(\.\.\.\) repeats the keyword 's'$"
    ):
        parse_condition("upper(s=a, s=b) == 'A'")


def test_a_builtin_without_a_readable_signature_is_not_arity_checked(monkeypatch):
    monkeypatch.setitem(TEMPLATE_FNS, "biggest", max)  # inspect.signature(max) raises ValueError
    parse_condition("biggest(a, b, c) > 0")


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert caught.value.diagnostic.frames[-1].range is not None
    return caught.value


_NODE_ASSERT = """
id: f
name: f
input:
  v: !int
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.v}
    output: int
    asserts:
      - "<cond>"
output: ${x.output}
"""

_FLOW_ASSERT = """
id: f
name: f
input:
  v: !int
asserts:
  - "<cond>"
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.v}
    output: int
output: ${x.output}
"""

_CASE = """
id: f
name: f
input:
  v: !int
nodes:
  gate:
    kind: case
    cases:
      - when: "<cond>"
        then: hot
    else: cold
  hot: {kind: code, code: "return 'hot'", output: str}
  cold: {kind: code, code: "return 'cold'", output: str}
output: ${hot.output | cold.output}
"""

_LOOP = """
id: f
name: f
defs:
  step:
    input: {n: !int , exited: !bool }
    nodes:
      s:
        kind: code
        code: "return {'n': inputs['n'] + 1, 'exited': inputs['n'] >= 2}"
        input:
          n: ${input.n}
        output:
          n: int
          exited: bool
    output: ${s.output}
nodes:
  repeat:
    kind: loop
    call: step
    input:
      n: 0
      exited: false
    <keyword>: "<cond>"
    max: 5
output: ${repeat.output}
"""


@pytest.mark.parametrize(
    "text",
    [
        _NODE_ASSERT.replace("<cond>", "nosuchfn(n) > 0"),
        _NODE_ASSERT.replace("<cond>", "nosuchfn(output) > 0"),
        _FLOW_ASSERT.replace("<cond>", "nosuchfn(input.v) > 0"),
        _FLOW_ASSERT.replace("<cond>", "nosuchfn(x.output) > 0"),
        _CASE.replace("<cond>", "nosuchfn(input.v) > 0"),
        _CASE.replace("<cond>", "true or nosuchfn(input.v)"),
        _LOOP.replace("<keyword>", "while").replace("<cond>", "not nosuchfn(exited)"),
        _LOOP.replace("<keyword>", "until").replace("<cond>", "nosuchfn(exited)"),
    ],
    ids=["node-pre", "node-post", "flow-boundary", "flow-post", "case", "case-short-circuit",
         "loop-while", "loop-until"],
)
def test_unknown_function_in_a_condition_fails_to_load(text):
    assert "unknown expression function 'nosuchfn'" in str(_load_error(text))


@pytest.mark.parametrize(
    "text",
    [
        _NODE_ASSERT.replace("<cond>", "upper(n, n) == 'A'"),
        _FLOW_ASSERT.replace("<cond>", "upper(input.v, 1) == 'A'"),
        _CASE.replace("<cond>", "upper('a', 'b') == 'A'"),
        _LOOP.replace("<keyword>", "while").replace("<cond>", "upper(exited, 1) == 'A'"),
    ],
    ids=["node", "flow", "case", "loop"],
)
def test_builtin_with_the_wrong_arguments_in_a_condition_fails_to_load(text):
    assert "expression function 'upper' takes (s): too many positional arguments" in str(
        _load_error(text)
    )


def test_whole_span_unknown_call_in_an_assert_keeps_the_inline_call_message():
    node = _load_error(_NODE_ASSERT.replace("<cond>", "${nosuchfn(n) > 0}"))
    assert "an inline call is not allowed in a node-local assert" in str(node)
    flow = _load_error(_FLOW_ASSERT.replace("<cond>", "${nosuchfn(input.v) > 0}"))
    assert "a flow call 'nosuchfn' inside ${...} is not supported" in str(flow)


@pytest.mark.parametrize("cond", ["${n >}", "${n > (1}"], ids=["dangling-operator", "unclosed-paren"])
def test_a_malformed_span_in_a_node_assert_reports_the_parse_error(cond):
    message = str(_load_error(_NODE_ASSERT.replace("<cond>", cond)))
    assert "not a valid boolean expression" in message
    assert "inline call" not in message


def test_a_condition_calling_a_builtin_correctly_loads_and_runs():
    flow = load_flow(_CASE.replace("<cond>", "upper(render_as_json(input.v, indent=0)) == '2'"))
    assert run_flow(flow, {"v": 2}).output == "hot"


def test_python_built_assert_calling_an_unknown_function_fails_before_the_run(tmp_path):
    flow = load_flow(_NODE_ASSERT.replace("<cond>", "n > 0"))
    flow.compiled.nodes["x"].pre_asserts = [Assertion("nosuchfn(n) > 0", "nosuchfn(n) > 0")]
    path = tmp_path / "run.jsonl"

    with pytest.raises(
        ExpressionError,
        match=r"^node 'x' pre-assert 'nosuchfn\(n\) > 0' is not a valid boolean expression: "
        r"unknown expression function 'nosuchfn'$",
    ):
        run_flow(flow, {"v": 1}, log=path)
    assert not path.exists()


def test_python_built_case_when_with_the_wrong_arguments_fails_before_the_run(tmp_path):
    flow = load_flow(_CASE.replace("<cond>", "input.v > 0"))
    flow.compiled.nodes["gate"].cases[0].when = "upper(__r0, 1) == 'A'"
    path = tmp_path / "run.jsonl"

    with pytest.raises(
        ExpressionError,
        match=r"^node 'gate' case 'hot' `when:` .* is not a valid boolean expression: "
        r"expression function 'upper' takes \(s\)",
    ):
        run_flow(flow, {"v": 1}, log=path)
    assert not path.exists()


_TYPEDEF_CASE = _CASE.replace("name: f\n", "name: f\ntypedefs:\n  Small: int(0..9)\n")


def test_a_typedef_constructor_in_a_condition_loads_and_runs():
    flow = load_flow(_TYPEDEF_CASE.replace("<cond>", "Small(input.v) > 4"))
    assert flow.compiled.nodes["gate"].types == {"Small": type_for("int(0..9)", {})}
    assert run_flow(flow, {"v": 7}).output == "hot"
    assert run_flow(flow, {"v": 2}).output == "cold"


def test_a_constructor_naming_no_typedef_in_a_condition_fails_to_load():
    error = _load_error(_TYPEDEF_CASE.replace("<cond>", "Nope(input.v) > 4"))
    assert "unknown expression function 'Nope'" in str(error)
