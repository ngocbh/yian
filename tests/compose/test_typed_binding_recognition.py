"""Recognizing `name: Type = value` binding text by its YAML scalar style."""

from __future__ import annotations

import pytest

from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.typed_bindings import TypedBindingText, split_typed_binding
from yian.compose.source import parse_yaml_literal
from yian.diagnostics import SourceRange
from yian.typesys.types import read_typedefs

REGISTRY = read_typedefs(
    {
        "OrderResult": {"id": "str", "qty": "int"},
        "Size": "Literal[small, large]",
        "Buy": {"action": "Literal[buy]", "qty": "int"},
        "Sell": {"action": "Literal[sell]", "qty": "int"},
    }
)
DIAGNOSTICS = LoadDiagnostics(label="flow.yaml", text="inputs:\n  qty: a = b\n")
RANGE = SourceRange(2, 8, 2, 13)


def _split(value, *, style="plain", label="qty"):
    return split_typed_binding(
        value,
        style=style,
        registry=REGISTRY,
        binding_label=label,
        diagnostics=DIAGNOSTICS,
        range=RANGE,
    )


def _error(value, *, label="qty") -> LoadError:
    with pytest.raises(LoadError) as caught:
        _split(value, label=label)
    error = caught.value
    assert [frame.range for frame in error.diagnostic.frames] == [RANGE]
    return error


@pytest.mark.parametrize(
    ("text", "type_text", "value"),
    [
        ("int = ${x}", "int", "${x}"),
        ("Optional[OrderResult] = ${p.output}", "Optional[OrderResult]", "${p.output}"),
        ("list[str] = []", "list[str]", []),
        ("int = 0", "int", 0),
        ("Optional[int] = null", "Optional[int]", None),
        ("Literal[a, b] = a", "Literal[a, b]", "a"),
        ('str = "30"', "str", "30"),
        ("int = call(f, x=1)", "int", "call(f, x=1)"),
        ("str = Hi ${name}", "str", "Hi ${name}"),
        ("Size = ${s.output}", "Size", "${s.output}"),
        ("int =  ${x}", "int", "${x}"),
        ("bool = yes", "bool", True),
        ("Optional[int] = null", "Optional[int]", None),
        ('str = "${a} = 2"', "str", "${a} = 2"),
        ("str = '${a}'", "str", "${a}"),
        ('int = "call(f, x=1)"', "int", "call(f, x=1)"),
        ('list[str] = ["${a}"]', "list[str]", ["${a}"]),
        (
            'OrderResult = {"id": "${a}", "qty": 1}',
            "OrderResult",
            {"id": "${a}", "qty": 1},
        ),
    ],
)
@pytest.mark.parametrize("style", ["plain", None])
def test_plain_or_unknown_style_value_with_separator_is_typed(text, type_text, value, style):
    assert _split(text, style=style) == TypedBindingText(type_text=type_text, value=value)


@pytest.mark.parametrize("style", ["single", "double", "literal", "folded"])
def test_quoted_or_block_value_is_plain_text(style):
    assert _split("a = b", style=style) is None


@pytest.mark.parametrize("text", ["${a == b}", "x=${y}", "int=3", "hello"])
def test_plain_value_without_separator_is_plain_text(text):
    assert _split(text) is None


@pytest.mark.parametrize("value", [3, [1], {"a": 1}])
def test_non_string_value_is_not_a_typed_binding(value):
    assert _split(value) is None


def test_unknown_left_side_suggests_quoting():
    error = _error("a = b")
    assert str(error) == (
        '`a` in `qty: a = b` is not a type. If you meant the text "a = b", '
        'quote it: `qty: "a = b"`'
    )


def test_misspelled_builtin_type_gets_a_close_match():
    error = _error("itn = ${x}")
    assert str(error).startswith("`itn` in `qty: itn = ${x}` is not a type.")
    assert str(error).endswith(" Did you mean `int`?")


def test_misspelled_constructor_gets_a_close_match():
    error = _error("Optinal[int] = ${x}")
    assert str(error).endswith(" Did you mean `Optional`?")


def test_misspelled_typedef_gets_a_close_match():
    error = _error("OrderReslt = ${x}")
    assert str(error).endswith(" Did you mean `OrderResult`?")


@pytest.mark.parametrize("text", ["list[Strr] = []", "date[x] = 1", "OrderResult[int] = 1"])
def test_valid_head_with_a_bad_type_gets_no_close_match(text):
    assert "Did you mean" not in str(_error(text))


def test_union_of_records_is_a_type():
    assert _split("Union[Buy, Sell] = ${x}") == TypedBindingText(
        type_text="Union[Buy, Sell]", value="${x}"
    )


def test_invalid_union_explains_why():
    assert str(_error("Union[Buy, str] = 1")) == (
        "`Union[Buy, str]` in `qty: Union[Buy, str] = 1` is not a type "
        "(Union[Buy, str]: member 'str' is not a record). "
        'If you meant the text "Union[Buy, str] = 1", quote it: '
        '`qty: "Union[Buy, str] = 1"`'
    )


def test_pipe_union_keeps_the_resolver_hint():
    message = str(_error("Buy | Sell = ${x}"))
    assert "use Union[Buy, Sell]" in message
    assert message.startswith("`Buy | Sell` in `qty: Buy | Sell = ${x}` is not a type (")


def test_bare_union_gets_the_generic_message():
    message = str(_error("Union = 1"))
    assert message.startswith("`Union` in `qty: Union = 1` is not a type. ")
    assert "Did you mean" not in message


def test_misspelled_union_gets_a_close_match():
    assert str(_error("Unoin[Buy, Sell] = ${x}")).endswith(" Did you mean `Union`?")


def test_lowercase_any_is_a_type():
    assert _split("any = ${x}") == TypedBindingText(type_text="any", value="${x}")


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="unknown YAML scalar style 'flow'"):
        _split("int = 1", style="flow")


def test_text_with_double_quotes_gets_a_valid_quoting_suggestion():
    text = '${y:-"a = b"}'
    error = _error(text)
    message = str(error)
    assert message.startswith('`${y:-"a` in `qty: ' + text + "` is not a type.")
    suggestion = message.rsplit("quote it: `", 1)[1].rstrip("`")
    assert suggestion == "qty: '" + text + "'"
    assert parse_yaml_literal(suggestion) == {"qty": text}


def test_text_with_both_quote_kinds_gets_a_valid_quoting_suggestion():
    text = """a'b = "c\""""
    message = str(_error(text))
    suggestion = message.rsplit("quote it: `", 1)[1].rstrip("`")
    assert parse_yaml_literal(suggestion) == {"qty": text}


def test_text_with_backslash_gets_a_valid_quoting_suggestion():
    text = r"a\b = c"
    message = str(_error(text))
    suggestion = message.rsplit("quote it: `", 1)[1].rstrip("`")
    assert parse_yaml_literal(suggestion) == {"qty": text}


@pytest.mark.parametrize(
    "text",
    [
        "a\nb = c",
        'a\tb = "c\\"',
        "a = b\x85c",
        "a = \U0001F600\nb",
        "a = \U000E0001",
    ],
)
def test_text_with_control_characters_gets_a_valid_quoting_suggestion(text):
    message = str(_error(text))
    suggestion = message.rsplit("quote it: `", 1)[1].rstrip("`")
    assert suggestion.startswith('qty: "')
    assert parse_yaml_literal(suggestion) == {"qty": text}


def test_call_directive_with_spaced_keyword_is_not_a_type():
    error = _error("call(f, a = 1)")
    assert str(error).startswith("`call(f, a` in `qty: call(f, a = 1)` is not a type.")


@pytest.mark.parametrize("text", ["int =", "int = "])
def test_empty_right_side_names_the_binding(text):
    error = _error(text)
    assert str(error) == "binding 'qty': the value after `int =` is empty"


def test_not_a_type_wins_over_empty_right_side():
    error = _error("itn =")
    assert str(error).startswith("`itn` in `qty: itn =` is not a type.")


def test_malformed_yaml_right_side_names_the_binding():
    error = _error("str = 'x")
    assert str(error).startswith("binding 'qty': bad value \"'x\": ")


_QUOTE_HINT = "; quote it to keep it as text: "
_BOOLEAN_NOTE = " (YAML reads unquoted yes/no/on/off/true/false as booleans)"


def test_unquoted_no_for_str_suggests_quoting_and_explains_booleans():
    assert str(_error("str = no")) == (
        "binding 'qty': `no` is not a valid str"
        + _QUOTE_HINT
        + '`qty: str = "no"`'
        + _BOOLEAN_NOTE
    )


@pytest.mark.parametrize(
    ("text", "right", "type_text"),
    [
        ("str = 0x1F", "0x1F", "str"),
        ("str = 017", "017", "str"),
        ("str = 1_000", "1_000", "str"),
        ("str = 30", "30", "str"),
        ("date = 2026-01-01", "2026-01-01", "date"),
        (
            "Optional[datetime] = 2026-01-01T10:00:00",
            "2026-01-01T10:00:00",
            "Optional[datetime]",
        ),
    ],
)
def test_literal_that_fits_as_text_suggests_quoting_the_authored_text(text, right, type_text):
    assert str(_error(text)) == (
        f"binding 'qty': `{right}` is not a valid {type_text}"
        + _QUOTE_HINT
        + f'`qty: {type_text} = "{right}"`'
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("datetime = 2026-01-01", "binding 'qty': `2026-01-01` is not a valid datetime"),
        ("Literal[a, b] = 30", "binding 'qty': `30` is not a valid Literal[a, b]"),
        ("int = abc", "binding 'qty': `abc` is not a valid int"),
        ("int = null", "binding 'qty': `null` is not a valid int"),
        ('int = "5"', "binding 'qty': `\"5\"` is not a valid int"),
    ],
)
def test_literal_that_does_not_fit_as_text_gets_no_quoting_hint(text, message):
    assert str(_error(text)) == message


def test_nested_literal_mismatch_explains_the_element():
    message = str(_error("list[int] = [1, a]"))
    assert message.startswith("binding 'qty': `[1, a]` is not a valid list[int] (")
    assert "quote" not in message


def test_unlocated_diagnostics_still_raise():
    with pytest.raises(LoadError, match="is not a type"):
        split_typed_binding(
            "a = b",
            style=None,
            registry=REGISTRY,
            binding_label="qty",
            diagnostics=LoadDiagnostics(),
        )
