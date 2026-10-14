"""A `${...}` marks an expression inside a YAML string; it can't appear inside one.

Covers the quote-aware whole-span test that conditions share, and the load error that a
`${...}` nested inside an expression raises (with the rewrite it shows).
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.expr import ExpressionError, condition_refs, rewrite_condition_refs, scan_template
from yian.expr.expressions import parse_condition
from yian.expr.grammar import parse_expr
from yian.run import run_flow


# --------------------------------------------------------------------------- #
# Whole-span conditions: one `${...}` around the whole value
# --------------------------------------------------------------------------- #


def test_whole_span_condition_matches_its_bare_spelling():
    assert parse_condition("${a > 5}") == parse_condition("a > 5")


def test_whole_span_condition_with_a_brace_in_a_string_literal():
    assert condition_refs('${x == "}"}') == ["x"]


def test_whole_span_rewrite_with_a_brace_in_a_string_literal():
    rename = {"x": "y"}.get
    assert rewrite_condition_refs('${x == "}"}', rename) == '${y == "}"}'


# --------------------------------------------------------------------------- #
# A `${...}` inside an expression is an error that shows the rewrite
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, rewrite",
    [
        ("input.a :- ${input.b}", "input.a :- input.b"),
        ("${a.output} > 5", "a.output > 5"),
        ("${a.output:-0} >= 1", "(a.output:-0) >= 1"),
        ("x :- ${ y }", "x :- y"),
        ("x :- ${y + 1}", "x :- (y + 1)"),
        ("a :- ${b :- ${c}}", "a :- (b :- c)"),
        ("isinstance(x, ${Buy})", "isinstance(x, Buy)"),
        ("render_as_json(${briefs}, 4)", "render_as_json(briefs, 4)"),
        ("not${x}", "not x"),
        ("x :- ${and}", "x :- and"),
    ],
)
def test_nested_span_shows_the_rewrite(text, rewrite):
    with pytest.raises(ExpressionError) as caught:
        parse_expr(text)
    assert str(caught.value) == (
        f"`${{...}}` can't appear inside an expression; write `{rewrite}`"
    )


@pytest.mark.parametrize("text", ["a :- ${b", "a :- ${b +}", "${}"])
def test_nested_span_without_a_parsable_rewrite_shows_none(text):
    with pytest.raises(ExpressionError, match=r"can't appear inside an expression") as caught:
        parse_expr(text)
    assert "write" not in str(caught.value)


@pytest.mark.parametrize("text", ['x == "${y}"', "x == '${y}'"])
def test_a_span_inside_a_string_literal_is_text(text):
    parse_expr(text)


def test_a_whole_span_condition_holding_a_nested_span_shows_the_bare_rewrite():
    with pytest.raises(ExpressionError, match=r"write `a > b`"):
        parse_condition("${a > ${b}}")


@pytest.mark.parametrize(
    "template, rewrite",
    [
        ("${input.as_of:-${system.today}}", "${input.as_of:-system.today}"),
        ("pre ${upper(${name})} post", "${upper(name)}"),
    ],
)
def test_template_span_shows_the_whole_fixed_span(template, rewrite):
    with pytest.raises(ExpressionError) as caught:
        scan_template(template)
    assert str(caught.value) == (
        f"`${{...}}` can't appear inside an expression; write `{rewrite}`"
    )


def test_template_span_with_a_quoted_span_is_text():
    scan_template('${x == "${y}"}')


# --------------------------------------------------------------------------- #
# Load errors name the location and show the rewrite
# --------------------------------------------------------------------------- #


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert caught.value.diagnostic.frames[-1].range is not None
    return caught.value


_BINDING = """
id: f
name: f
input:
  a: Optional[str] = null
  b: Optional[str] = null
nodes:
  n:
    kind: code
    code: "return inputs['v']"
    input:
      v: <value>
    output: Optional[str]
output: ${n.output}
"""


def test_binding_with_a_nested_span_fails_to_load_with_the_rewrite():
    error = _load_error(_BINDING.replace("<value>", "${input.a :- ${input.b}}"))
    assert "write `${input.a :- input.b}`" in str(error)


@pytest.mark.parametrize(
    "value, result",
    [
        ('${input.a :- (input.b :- "lit")}', "lit"),
        ('${input.a | input.b | "lit"}', "lit"),
    ],
)
def test_parentheses_and_coalesce_cover_a_default_chain(value, result):
    out = run_flow(load_flow(_BINDING.replace("<value>", value)), {})
    assert out.status == "succeeded"
    assert out.output == result


_SEARCHED = """
id: f
name: f
input:
  n: Optional[int] = null
  approve: bool = false
  label: str = none
nodes:
  score:
    kind: code
    code: "return inputs['n']"
    input: {n: '${input.n}'}
    output: Optional[int]
  gate:
    kind: case
    cases:
      - when: <when>
        then: hot
    else: cold
  hot: {kind: code, code: "return 'hot'", output: str}
  cold: {kind: code, code: "return 'cold'", output: str}
output: ${hot.output | cold.output}
"""


@pytest.mark.parametrize(
    "when, rewrite",
    [
        ('"${score.output} > 5"', "score.output > 5"),
        ('"${score.output:-0} >= 1"', "(score.output:-0) >= 1"),
    ],
)
def test_case_condition_with_a_nested_span_fails_to_load_with_the_rewrite(when, rewrite):
    error = _load_error(_SEARCHED.replace("<when>", when))
    assert f"write `{rewrite}`" in str(error)


@pytest.mark.parametrize(
    "when, inputs, taken",
    [
        ('"score.output > 0 and score.output < 10"', {"n": 5}, "hot"),
        ('"score.output > 0 and score.output < 10"', {"n": 50}, "cold"),
        ('"${score.output > 5}"', {"n": 6}, "hot"),
        ("${input.approve}", {"approve": True}, "hot"),
        ("${input.approve}", {}, "cold"),
        ('"(input.n :- 9) >= 1"', {}, "hot"),
    ],
)
def test_documented_case_conditions_route(when, inputs, taken):
    out = run_flow(load_flow(_SEARCHED.replace("<when>", when)), inputs)
    assert out.status == "succeeded"
    assert out.output == taken


def test_case_on_a_whole_span_routes():
    text = _SEARCHED.replace(
        "    cases:\n      - when: <when>",
        "    on: ${input.label}\n    cases:\n      - when: go",
    )
    assert run_flow(load_flow(text), {"label": "go"}).output == "hot"
    assert run_flow(load_flow(text), {}).output == "cold"


_ASSERTS = """
id: f
name: f
input: {n: int}
nodes:
  n:
    kind: code
    code: "return inputs['n']"
    input: {n: '${input.n}'}
    output: int
    asserts:
      - <node_assert>
output: ${n.output}
asserts:
  - <flow_assert>
"""


def test_flow_assert_with_a_nested_span_fails_to_load_with_the_rewrite():
    text = _ASSERTS.replace("<node_assert>", "output >= 0").replace(
        "<flow_assert>", "${input.n} > 0"
    )
    assert "write `input.n > 0`" in str(_load_error(text))


def test_node_assert_with_a_nested_span_fails_to_load_with_the_rewrite():
    text = _ASSERTS.replace("<node_assert>", "${output} < 0").replace(
        "<flow_assert>", "input.n > 0"
    )
    assert "write `output < 0`" in str(_load_error(text))


_LOOP = """
id: f
name: f
defs:
  step:
    input: {n: int, exited: bool}
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
    while: <predicate>
    max: 5
output: ${repeat.output}
"""


def test_loop_predicate_with_a_nested_span_fails_to_load_with_the_rewrite():
    error = _load_error(_LOOP.replace("<predicate>", '"not ${exited}"'))
    assert "write `not exited`" in str(error)


def test_loop_predicate_in_the_bare_spelling_runs():
    out = run_flow(load_flow(_LOOP.replace("<predicate>", "not exited")), {})
    assert out.status == "succeeded"
    assert out.output == {"n": 3, "exited": True}


_PROMPT = """
id: f
name: f
input: {name: str}
nodes:
  brief:
    kind: agent
    input: {name: '${input.name}'}
    output: str
    prompt: "Hello ${upper(${name})}"
output: ${brief.output}
"""


def test_prompt_with_a_nested_span_fails_to_load_with_the_rewrite():
    assert "write `${upper(name)}`" in str(_load_error(_PROMPT))
