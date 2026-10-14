"""A condition may call only a builtin, with arguments the builtin takes."""

import pytest

from yian.expr import ExpressionError
from yian.expr.builtins import TEMPLATE_FNS
from yian.expr.expressions import parse_condition


@pytest.mark.parametrize(
    "condition",
    ["nosuchfn(n) > 0", "${nosuchfn(n) > 0}", "true or nosuchfn(n)", "upper(nosuchfn(n)) == 'A'"],
)
def test_unknown_function_does_not_parse_as_a_condition(condition):
    with pytest.raises(ExpressionError, match=r"^unknown expression function 'nosuchfn'$"):
        parse_condition(condition)


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
        ExpressionError, match=r"^expression function 'upper' got keyword 's' twice$"
    ):
        parse_condition("upper(s=a, s=b) == 'A'")


def test_a_builtin_without_a_readable_signature_is_not_arity_checked(monkeypatch):
    monkeypatch.setitem(TEMPLATE_FNS, "biggest", max)  # inspect.signature(max) raises ValueError
    parse_condition("biggest(a, b, c) > 0")
