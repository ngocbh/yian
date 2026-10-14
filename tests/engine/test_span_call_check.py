"""The span call check: every call in a `${...}` must name a builtin that takes its arguments."""

import pytest

from yian.expr import ExpressionError, check_template_calls, prompt_refs

_UPPER = r"expression function 'upper' takes \(s\): too many positional arguments"


@pytest.mark.parametrize(
    "text, message",
    [
        ("${upper(v, 1)}", _UPPER),
        ("hi ${upper(v, 1)} there", _UPPER),
        ("${v :- upper(v, 1)}", _UPPER),
        ("${upper()}", r"expression function 'upper' takes \(s\): missing a required argument: 's'"),
        (
            "${render_as_json(v, width=4)}",
            r"expression function 'render_as_json' takes \(value, indent=2\): "
            r"got an unexpected keyword argument 'width'",
        ),
        ("${upper(s=v, s=w)}", r"upper\(\.\.\.\) repeats the keyword 's'"),
    ],
)
def test_template_call_check_rejects_a_bad_call(text, message):
    with pytest.raises(ExpressionError, match=rf"^{message}$"):
        check_template_calls(text)


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        "${v}",
        "${upper(v)} and ${join(xs, sep=',')}",
        "${isinstance(v, int)}",
        "${render_as_json(v, indent=4)}",
        "$${upper(v, 1)}",
        "${Buy(v)}",
        "${Buy(action='buy', qty=upper(v))}",
    ],
)
def test_template_call_check_accepts_good_calls(text):
    check_template_calls(text)


def test_template_call_check_raises_a_parse_error():
    with pytest.raises(ExpressionError, match="could not parse"):
        check_template_calls("${v +}")


@pytest.mark.parametrize(
    "text, message",
    [
        ("Say ${upper(t, 1)}", _UPPER),
        ("${render_as_json(t, width=4)}", r"expression function 'render_as_json' takes"),
    ],
)
def test_prompt_with_a_mis_called_builtin_is_rejected(text, message):
    with pytest.raises(ExpressionError, match=message):
        prompt_refs(text)


def test_prompt_unknown_function_keeps_its_message():
    with pytest.raises(ExpressionError, match=r"^unknown prompt function 'nosuch';"):
        prompt_refs("${nosuch(t)}")
