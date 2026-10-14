"""CASE `on:` form: each `when:` value is a typed literal compared exactly with the `on:` value.

Ints and floats compare numerically (`when: 2` matches `2.0`), a bool never equals a number
(Python's `True == 1` is guarded with `isinstance`), quoted values and unquoted dates are
text, and an untyped `on:` gets the same exact comparison at runtime. A non-finite number,
a list or record, and an unquoted datetime are located load errors. Where the `on:` type is
known, a `when:` value that can never be a member of it is a located load error, even next
to an `else:`, naming the value, the type, and the fix.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.case import _match_condition
from yian.expr.expressions import evaluate_when_record
from yian.run import run_flow
from yian.typesys.types import type_for
from yian.typesys.values import canonical_type_label


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _error_line(error: LoadError) -> int:
    """Return the start line of the innermost frame of a load error."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


def _branch(name: str) -> str:
    """Return a CODE node `name` that returns its own name."""
    return (
        f"  {name}:\n"
        "    kind: code\n"
        "    code: |\n"
        f'      return "{name}"\n'
        "    output: str\n"
    )


def _flow(
    whens: list[tuple[str, str]],
    *,
    on_type: str | None = "int",
    else_: str | None = "other",
    src_code: str = 'return True if inputs["p"] == "t" else 1',
    src_output: str | None = None,
) -> str:
    """Return a flow whose `route` case matches `on:` against `whens` (`(when, then)` pairs).

    With `on_type`, `on:` reads the flow input `n: <on_type>`; with `on_type=None` it reads an
    CODE producer `src` that runs `src_code` over the input `pick`, typed by `src_output`
    (untyped when `None`).
    """
    targets = list(dict.fromkeys(then for _, then in whens))
    if else_ is not None:
        targets.append(else_)
    if on_type is not None:
        text = f"id: f\nname: f\ninput:\n  n: {on_type}\nnodes:\n"
        on = "${input.n}"
    else:
        text = (
            "id: f\nname: f\ninput:\n  pick: str\nnodes:\n"
            "  src:\n"
            "    kind: code\n"
            "    input:\n"
            "      p: ${input.pick}\n"
            "    code: |\n"
            f"      {src_code}\n"
        )
        if src_output is not None:
            text += f"    output: {src_output}\n"
        on = "${src.output}"
    text += f"  route:\n    kind: case\n    on: {on}\n    cases:\n"
    for when, then in whens:
        text += f"      - when: {when}\n        then: {then}\n"
    if else_ is not None:
        text += f"    else: {else_}\n"
    text += "".join(_branch(name) for name in targets)
    text += "output: ${" + " | ".join(f"{name}.output" for name in targets) + "}\n"
    return text


def _route(text: str, inputs: dict) -> str:
    """Load and run `text` with `inputs`; return the flow output."""
    result = run_flow(load_flow(text), inputs)
    assert result.status == "succeeded", result.failure
    return result.output


def _load_error(text: str) -> LoadError:
    """Return the load error `text` raises."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def test_int_on_matches_int_when():
    text = _flow([("1", "one"), ("2", "two")])
    assert _route(text, {"n": 1}) == "one"
    assert _route(text, {"n": 2}) == "two"
    assert _route(text, {"n": 3}) == "other"


def test_negative_int_when():
    text = _flow([("-1", "one")])
    assert _route(text, {"n": -1}) == "one"
    assert _route(text, {"n": 1}) == "other"


def test_float_on_matches_numerically():
    text = _flow([("1.5", "one"), ("2", "two")], on_type="float")
    assert _route(text, {"n": 1.5}) == "one"
    assert _route(text, {"n": 2.0}) == "two"
    assert _route(text, {"n": 3.0}) == "other"


def test_bool_on_matches_bool_when():
    text = _flow([("true", "one"), ("false", "two")], on_type="bool")
    assert _route(text, {"n": True}) == "one"
    assert _route(text, {"n": False}) == "two"


def test_quoted_when_is_text():
    text = _flow([('"1"', "one")], on_type="str")
    assert _route(text, {"n": "1"}) == "one"


def test_no_else_int_on_routes_at_runtime():
    text = _flow([("1", "one"), ("2", "two")], else_=None)
    assert _route(text, {"n": 1}) == "one"
    assert _route(text, {"n": 2}) == "two"


def test_bool_never_equals_a_number_on_an_untyped_on():
    text = _flow([("1", "one"), ("true", "two")], on_type=None)
    assert _route(text, {"pick": "t"}) == "two"
    assert _route(text, {"pick": "x"}) == "one"
    floats = _flow([("1", "one")], on_type=None, src_code="return 1.0")
    assert _route(floats, {"pick": "x"}) == "one"


def test_lowered_conditions():
    text = _flow(
        [("1", "one"), ("1.5", "two"), ("true", "three"), ('"x"', "four")],
        on_type=None,
    )
    cases = load_flow(text).compiled.nodes["route"].cases
    assert [case.when for case in cases] == [
        "${__on} == 1 and not isinstance(${__on}, bool)",
        "${__on} == 1.5 and not isinstance(${__on}, bool)",
        "${__on} == true and isinstance(${__on}, bool)",
        '${__on} == "x"',
    ]


@pytest.mark.parametrize("value", [1e20, 1.5e-7, -2.5, -0.0, 5e-324])
def test_large_and_small_floats_lower_exactly(value):
    condition = _match_condition(
        value, LoadDiagnostics(), node_id="route", index=0, range=None
    )
    types = {"bool": type_for("bool", {})}
    assert evaluate_when_record(condition, {"__on": value}, types) is True
    if value * 2 != value:
        assert evaluate_when_record(condition, {"__on": value * 2}, types) is False


def test_date_when_is_iso_text():
    text = _flow([("2026-01-02", "one")], on_type=None, src_code='return "2026-01-02"')
    assert _route(text, {"pick": "x"}) == "one"


@pytest.mark.parametrize(
    ("authored", "shown"), [(".inf", "inf"), ("-.inf", "-inf"), (".nan", "nan")]
)
def test_non_finite_when_is_a_load_error(authored, shown):
    text = _flow([(authored, "one")], on_type="float")
    error = _load_error(text)
    assert str(error) == (
        f"case node 'route' cases[0].when: non-finite number {shown} is not a yian value"
    )
    assert _error_line(error) == _line_of(text, f"when: {authored}")


@pytest.mark.parametrize(("authored", "kind"), [("[a]", "a list"), ("{a: 1}", "a record")])
def test_list_or_record_when_is_a_load_error(authored, kind):
    text = _flow([(authored, "one")])
    error = _load_error(text)
    assert (
        "case node 'route': an on: case's when: takes text, a number, or a bool, "
        f"got {kind}"
    ) in str(error)
    assert _error_line(error) == _line_of(text, f"when: {authored}")


def test_unquoted_datetime_when_is_a_load_error():
    text = _flow([("2026-01-02T10:00:00Z", "one")], on_type="str")
    error = _load_error(text)
    assert str(error) == (
        "case node 'route': quote a datetime when: to match its text exactly: "
        '`when: "2026-01-02T10:00:00Z"`'
    )
    assert _error_line(error) == _line_of(text, "when: 2026-01-02T10")


def test_untyped_on_compares_exactly():
    unquoted = _flow([("1", "one")], on_type=None, src_code='return "1"')
    assert _route(unquoted, {"pick": "x"}) == "other"
    quoted = _flow([('"1"', "one")], on_type=None, src_code='return "1"')
    assert _route(quoted, {"pick": "x"}) == "one"


def _when_error(text: str, when: str) -> str:
    """Return the message of the load error `text` raises, asserting it is at `when: <when>`."""
    error = _load_error(text)
    assert _error_line(error) == _line_of(text, f"when: {when}")
    return str(error)


def test_int_when_on_str_is_a_migration_error():
    text = _flow([("1", "one")], on_type="str")
    assert _when_error(text, "1") == (
        "case node 'route': `when: 1` is an int but `on: ${input.n}` is `str`; "
        'quote it: `when: "1"`'
    )


def test_text_when_on_int_is_an_error():
    text = _flow([('"1"', "one")])
    assert _when_error(text, '"1"') == (
        "case node 'route': `when: \"1\"` is text but `on: ${input.n}` is `int`; "
        "drop the quotes: `when: 1`"
    )


def test_bool_when_on_int_is_an_error():
    text = _flow([("true", "one")])
    assert _when_error(text, "true") == (
        "case node 'route': `when: true` is a bool but `on: ${input.n}` is `int`"
    )


def test_int_when_on_bool_is_an_error():
    text = _flow([("1", "one")], on_type="bool")
    assert _when_error(text, "1") == (
        "case node 'route': `when: 1` is an int but `on: ${input.n}` is `bool`"
    )


def test_float_when_on_int_is_an_error():
    text = _flow([("1.5", "one")])
    assert _when_error(text, "1.5") == (
        "case node 'route': `when: 1.5` is a float but `on: ${input.n}` is `int`"
    )


def test_out_of_bounds_int_when():
    text = _flow([("1", "one"), ("13", "two")], on_type="int[1, 12]")
    message = _when_error(text, "13")
    assert message.startswith(
        "case node 'route': `when: 13` is an int but `on: ${input.n}` is `int[1,12]` ("
    )
    assert message.endswith(")")


def test_typo_next_to_else_on_a_literal_node_output():
    text = _flow(
        [("go", "one"), ("nogo", "two")],
        on_type=None,
        src_code='return "go"',
        src_output="Literal[go, no_go]",
    )
    label = canonical_type_label(type_for("Literal[go, no_go]", {}))
    assert _when_error(text, "nogo") == (
        f"case node 'route': `when: nogo` is not a tag of `on: ${{src.output}}` (`{label}`); "
        "use one of: go, no_go"
    )


def test_typo_on_a_literal_flow_input():
    text = _flow([("a", "one"), ("c", "two")], on_type="Literal[a, b]")
    label = canonical_type_label(type_for("Literal[a, b]", {}))
    assert _when_error(text, "c") == (
        f"case node 'route': `when: c` is not a tag of `on: ${{input.n}}` (`{label}`); "
        "use one of: a, b"
    )


def test_int_when_on_a_literal_is_an_error():
    text = _flow([("1", "one")], on_type="Literal[a, b]")
    label = canonical_type_label(type_for("Literal[a, b]", {}))
    assert _when_error(text, "1") == (
        f"case node 'route': `when: 1` is an int but `on: ${{input.n}}` is `{label}`"
    )


def test_quote_fix_on_a_literal_with_a_numeric_tag():
    text = _flow([("1", "one")], on_type='Literal["1", "2"]')
    label = canonical_type_label(type_for('Literal["1", "2"]', {}))
    assert _when_error(text, "1") == (
        f"case node 'route': `when: 1` is an int but `on: ${{input.n}}` is `{label}`; "
        'quote it: `when: "1"`'
    )


def test_tags_that_read_as_other_values_are_listed_quoted():
    text = _flow([('"3"', "one")], on_type='Literal["1", "2"]')
    label = canonical_type_label(type_for('Literal["1", "2"]', {}))
    assert _when_error(text, '"3"') == (
        f"case node 'route': `when: \"3\"` is not a tag of `on: ${{input.n}}` (`{label}`); "
        'use one of: "1", "2"'
    )


_TEXT_ON_INT = (
    "case node 'route': `when: \"1\"` is text but `on: ${input.%s}` is `int`; "
    "drop the quotes: `when: 1`"
)


def _branch_body(name: str, output: str, value: str) -> str:
    """Return an inline-body CODE node `name` returning `value` as `output`, indented six."""
    return (
        f"      {name}:\n"
        "        kind: code\n"
        "        code: |\n"
        f"          return {value}\n"
        f"        output: {output}\n"
    )


def test_loop_body_carried_field():
    text = (
        "id: f\nname: f\ninput:\n  s: int\nnodes:\n"
        "  rep:\n"
        "    kind: loop\n"
        "    times: 2\n"
        "    input:\n"
        "      n: int = 0\n"
        "    nodes:\n"
        "      route:\n"
        "        kind: case\n"
        "        on: ${input.n}\n"
        "        cases:\n"
        '          - when: "1"\n'
        "            then: one\n"
        "        else: other\n"
        + _branch_body("one", "int", "2")
        + _branch_body("other", "int", "1")
        + "    output:\n"
        "      n: ${one.output | other.output}\n"
        "output: ${rep.output}\n"
    )
    assert _when_error(text, '"1"').endswith(_TEXT_ON_INT % "n")


def test_map_child_input():
    text = (
        "id: f\nname: f\ninput:\n  xs: list[int]\nnodes:\n"
        "  each:\n"
        "    kind: map\n"
        "    over: ${input.xs}\n"
        "    input:\n"
        "      x: ${item}\n"
        "    nodes:\n"
        "      route:\n"
        "        kind: case\n"
        "        on: ${input.x}\n"
        "        cases:\n"
        '          - when: "1"\n'
        "            then: one\n"
        "        else: other\n"
        + _branch_body("one", "str", '"one"')
        + _branch_body("other", "str", '"other"')
        + "    output: ${one.output | other.output}\n"
        "output: ${each.output}\n"
    )
    assert _when_error(text, '"1"').endswith(_TEXT_ON_INT % "x")


def test_unknown_on_type_is_not_checked():
    text = _flow([("1", "one"), ('"a"', "two"), ("true", "three")], on_type=None)
    load_flow(text)


def test_object_on_type_is_not_checked():
    load_flow(_flow([("1", "one")], on_type="object"))


def test_optional_on_type_checks_the_non_null_part():
    load_flow(_flow([("1", "one")], on_type="Optional[int]"))
    text = _flow([('"1"', "one")], on_type="Optional[int]")
    assert "`Optional[int]`" in _when_error(text, '"1"')


def test_date_when_on_a_date_input():
    text = _flow([("2026-01-02", "one"), ('"2026-01-03"', "two")], on_type="date")
    assert _route(text, {"n": "2026-01-02"}) == "one"
    assert _route(text, {"n": "2026-01-03"}) == "two"


def test_bool_when_on_str_quote_fix_keeps_spelling():
    title = _flow([("True", "one")], on_type="str")
    assert _when_error(title, "True").endswith('; quote it: `when: "True"`')
    lower = _flow([("true", "one")], on_type="str")
    assert _when_error(lower, "true").endswith('; quote it: `when: "true"`')


def test_integral_float_when_on_int():
    text = _flow([("1.0", "one")])
    assert _when_error(text, "1.0") == (
        "case node 'route': `when: 1.0` is a float but `on: ${input.n}` is `int`; "
        "write it as an int: `when: 1`"
    )


def test_no_drop_quotes_hint_for_text_yaml_reads_as_text():
    text = _flow([("1e3", "one")], on_type="float")
    try:
        load_flow(text)
    except LoadError as error:
        assert "is text but `on: ${input.n}` is `float`" in str(error)
        assert "drop the quotes: `when: 1e3`" not in str(error)


def test_invalid_date_text_hint_does_not_crash():
    text = _flow([('"2026-13-01"', "one")])
    assert _when_error(text, '"2026-13-01"') == (
        "case node 'route': `when: \"2026-13-01\"` is text but `on: ${input.n}` is `int`"
    )


@pytest.mark.parametrize(
    ("when", "on_type"),
    [('""', "Optional[int]"), ('"null"', "Optional[int]"), ('".inf"', "float")],
)
def test_no_drop_quotes_for_null_or_nonfinite_text(when, on_type):
    text = _flow([(when, "one")], on_type=on_type)
    message = _when_error(text, when)
    assert f"`when: {when}` is text but `on: ${{input.n}}` is `{on_type}`" in message
    assert "drop the quotes" not in message


def test_unquoted_date_on_int_is_called_a_date():
    text = _flow([("2026-01-02", "one")])
    assert _when_error(text, "2026-01-02") == (
        "case node 'route': `when: 2026-01-02` is a date but `on: ${input.n}` is `int`"
    )


def test_expression_on_is_typed():
    text = _flow([('"x"', "one")]).replace("on: ${input.n}", "on: ${input.n > 1}")
    assert _when_error(text, '"x"') == (
        "case node 'route': `when: \"x\"` is text but `on: ${input.n > 1}` is `bool`"
    )
    untyped = _flow([('"2"', "one")]).replace("on: ${input.n}", "on: ${input.n + 1}")
    load_flow(untyped)
