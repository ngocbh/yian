"""CASE `on:` form: each `when:` value is a typed literal compared exactly with the `on:` value.

Ints and floats compare numerically (`when: 2` matches `2.0`), a bool never equals a number
(Python's `True == 1` is guarded with `isinstance`), quoted values and unquoted dates are
text, and an untyped `on:` gets the same exact comparison at runtime. A non-finite number,
a list or record, and an unquoted datetime are located load errors.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.case import _match_condition
from yian.expr.expressions import evaluate_when_record
from yian.run import run_flow
from yian.typesys.types import type_for


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
) -> str:
    """Return a flow whose `route` case matches `on:` against `whens` (`(when, then)` pairs).

    With `on_type`, `on:` reads the flow input `n: <on_type>`; with `on_type=None` it reads an
    untyped CODE producer `src` (no `output:`) that runs `src_code` over the input `pick`.
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
