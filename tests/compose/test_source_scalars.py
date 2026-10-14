"""Plain scalars read with YAML 1.2 core rules, keeping timestamps and merge keys.

Both YAML entry points, ``SourceDocument.parse`` and ``parse_yaml_literal``, share
one loader table; PyYAML's global SafeLoader keeps its YAML 1.1 rules.
"""

import math
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import pytest
import yaml

from yian.compose.source import (
    SourceDocument,
    SourceSyntaxError,
    parse_yaml_literal,
    read_plain_scalar,
)
from yian.compose.source.scalars import _FlowLoader


def _document_value(text: str):
    return SourceDocument.parse(text).root.value


ENTRY_POINTS = pytest.mark.parametrize(
    "read",
    [_document_value, parse_yaml_literal],
    ids=["document", "literal"],
)

_WORDS = [
    word
    for base in ("on", "off", "yes", "no")
    for word in (base, base.title(), base.upper())
]

_ROWS = [
    *[(word, word) for word in _WORDS],
    *[(word, word) for word in ("y", "n", "Y", "N", "oN")],
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("false", False),
    ("False", False),
    ("FALSE", False),
    ("1:30", "1:30"),
    ("017", 17),
    ("0o17", 15),
    ("0x1F", 31),
    ("1_000", "1_000"),
    ("0b101", "0b101"),
    ("1e3", 1000.0),
    ("-1.5E-2", -0.015),
    (".5", 0.5),
    ("+12", 12),
    ("2026-10-14", date(2026, 10, 14)),
    ("2026-10-14T10:00:00Z", datetime(2026, 10, 14, 10, tzinfo=timezone.utc)),
    ("~", None),
    ("null", None),
    ("NULL", None),
    ("=", "="),
    (".inf", float("inf")),
    ("-.inf", float("-inf")),
    ("+.Inf", float("inf")),
    ("-.nan", "-.nan"),
]


def _assert_same(actual, expected):
    assert actual == expected
    assert type(actual) is type(expected)
    if isinstance(expected, datetime):
        assert actual.utcoffset() == timedelta(0)


@ENTRY_POINTS
@pytest.mark.parametrize(("text", "expected"), _ROWS)
def test_plain_scalar_value(read, text, expected):
    _assert_same(read(f"value: {text}\n")["value"], expected)


@ENTRY_POINTS
@pytest.mark.parametrize(("text", "expected"), [row for row in _ROWS if row[1] is not None])
def test_plain_scalar_key(read, text, expected):
    (key,) = read(f"{text}: value\n")
    _assert_same(key, expected)


@ENTRY_POINTS
@pytest.mark.parametrize("text", ["~", "null", "NULL", ""])
def test_null_forms_as_a_key(read, text):
    assert read(f"? {text}\n: value\n") == {None: "value"}


@ENTRY_POINTS
def test_empty_plain_scalar_is_null(read):
    assert read("value:\n") == {"value": None}
    assert read("") is None


@ENTRY_POINTS
@pytest.mark.parametrize("text", [".nan", ".NaN"])
def test_nan_is_a_float(read, text):
    assert math.isnan(read(f"value: {text}\n")["value"])


@ENTRY_POINTS
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("'yes'", "yes"),
        ('"017"', "017"),
        ("|\n  yes\n", "yes\n"),
        (">\n  017\n", "017\n"),
    ],
    ids=["single", "double", "literal", "folded"],
)
def test_quoted_and_block_scalars_stay_text(read, text, expected):
    assert read(f"value: {text}") == {"value": expected}


@ENTRY_POINTS
def test_nested_collections_read_the_same(read):
    text = "outer:\n  inner: [yes, 017, on]\n  map: {no: 0x1F, true: 1:30}\n"

    assert read(text) == {
        "outer": {"inner": ["yes", 17, "on"], "map": {"no": 31, True: "1:30"}}
    }


@ENTRY_POINTS
def test_merge_keys_still_merge_and_override(read):
    text = (
        "base: &base\n"
        "  on: x\n"
        "  yes: 1\n"
        "  017: a\n"
        "target:\n"
        "  <<: *base\n"
        "  yes: 2\n"
    )

    value = read(text)

    assert value["base"] == {"on": "x", "yes": 1, 17: "a"}
    assert value["target"] == {"on": "x", "yes": 2, 17: "a"}


@ENTRY_POINTS
@pytest.mark.parametrize(
    ("text", "expected"),
    [("!!int 017", 17), ("!!float 1", 1.0), ("!!str 1", "1")],
)
def test_explicit_tag_reads_a_core_form(read, text, expected):
    _assert_same(read(f"value: {text}\n")["value"], expected)


@ENTRY_POINTS
@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("!!int 1_000", "1_000"),
        ("!!int 1.0", "1.0"),
        ("!!bool yes", "yes"),
        ("!!bool |\n  true\n", "true\n"),
        ('!!float ".inf\\n"', ".inf\n"),
    ],
    ids=["int-underscore", "int-float", "bool-yes", "bool-block-newline", "float-newline"],
)
def test_explicit_tag_rejects_a_non_core_form(read, text, found):
    with pytest.raises(SourceSyntaxError) as caught:
        read(f"value: {text}\n")

    assert repr(found) in str(caught.value)
    if read is _document_value:
        assert caught.value.range is not None
        assert caught.value.range.start_line == 1


@ENTRY_POINTS
def test_an_integer_too_long_to_convert_is_a_located_syntax_error(read):
    message = "an integer with 5000 digits is too long"
    with pytest.raises(SourceSyntaxError, match=message) as caught:
        read("value: " + "1" * 5000 + "\n")

    if read is _document_value:
        assert caught.value.range is not None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        *_ROWS,
        ("", None),
        ("None", "None"),
        ("none", "none"),
        ("NONE", "NONE"),
        ("tRUE", "tRUE"),
        ("<<", "<<"),
    ],
)
def test_read_plain_scalar_follows_the_table(text, expected):
    _assert_same(read_plain_scalar(text), expected)


@pytest.mark.parametrize("text", [".nan", ".NaN"])
def test_read_plain_scalar_nan(text):
    assert math.isnan(read_plain_scalar(text))


@pytest.mark.parametrize(
    "text",
    ["a: b", "[1]", "{a: 1}", "- x", "a #b", "&a", "*a", "!x", "|", "hello world", "1\n"],
)
def test_read_plain_scalar_keeps_yaml_structure_as_text(text):
    assert read_plain_scalar(text) == text


def test_read_plain_scalar_rejects_an_unbuildable_form():
    with pytest.raises(SourceSyntaxError) as caught:
        read_plain_scalar("2026-13-45")

    # The inline-call error adds the quoting fix and must not echo the value.
    assert str(caught.value) == "not a valid date: month must be in 1..12"


_RESOLVER_TAGS = (
    "import yaml; "
    "print(repr([sorted((key, [tag for tag, _ in value]) "
    "for key, value in loader.yaml_implicit_resolvers.items()) "
    "for loader in (yaml.SafeLoader, yaml.Loader, yaml.FullLoader)]))"
)


def test_pyyaml_globals_are_untouched():
    SourceDocument.parse("a: yes\nb: 017\n")
    parse_yaml_literal("c: on\n")

    # A fresh interpreter that never imports yian gives PyYAML's own tables.
    fresh = subprocess.run(
        [sys.executable, "-c", _RESOLVER_TAGS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    in_process = repr(
        [
            sorted(
                (key, [tag for tag, _ in value])
                for key, value in loader.yaml_implicit_resolvers.items()
            )
            for loader in (yaml.SafeLoader, yaml.Loader, yaml.FullLoader)
        ]
    )

    assert in_process == fresh
    assert yaml.safe_load("yes") is True
    assert yaml.safe_load("1:30") == 90
    assert yaml.safe_load("017") == 15
    assert yaml.safe_load("!!bool yes") is True
    assert _FlowLoader.yaml_implicit_resolvers is not yaml.SafeLoader.yaml_implicit_resolvers
    assert "yaml_implicit_resolvers" in _FlowLoader.__dict__
    assert "yaml_constructors" in _FlowLoader.__dict__
    assert (
        yaml.SafeLoader.yaml_constructors["tag:yaml.org,2002:int"]
        is yaml.constructor.SafeConstructor.construct_yaml_int
    )
    assert (
        yaml.SafeLoader.yaml_constructors["tag:yaml.org,2002:timestamp"]
        is yaml.constructor.SafeConstructor.construct_yaml_timestamp
    )


_BAD_DATES = [
    ("2026-13-45", "month must be in 1..12"),
    ("2026-02-30", "day is out of range for month"),
    ("0000-01-01", "year 0 is out of range"),
    ("2026-01-01T25:00:00", "hour must be in 0..23"),
    ("2026-01-01 10:00:00 +99:00", "UTC offset must be under 24:00"),
]


@ENTRY_POINTS
@pytest.mark.parametrize(("text", "reason"), _BAD_DATES)
def test_an_unbuildable_date_is_a_located_syntax_error_with_the_fix(read, text, reason):
    with pytest.raises(SourceSyntaxError) as caught:
        read(f"value: {text}\n")

    message = str(caught.value)
    kind = "datetime" if ":" in text else "date"
    assert f"not a valid {kind}: {reason}" in message
    assert "quote it to keep it as text" in message
    if read is _document_value:
        assert caught.value.range is not None
        assert (caught.value.range.start_line, caught.value.range.start_col) == (1, 8)


@pytest.mark.parametrize(
    ("text", "line", "column"),
    [("[1, 2026-13-45]\n", 1, 5), ("a:\n  2026-13-45: 1\n", 2, 3)],
    ids=["in-a-list", "as-a-key"],
)
def test_an_unbuildable_date_in_a_list_or_key_is_located(text, line, column):
    with pytest.raises(SourceSyntaxError, match="not a valid date") as caught:
        SourceDocument.parse(text)

    assert (caught.value.range.start_line, caught.value.range.start_col) == (line, column)


@ENTRY_POINTS
def test_an_explicit_timestamp_tag_on_non_date_text_names_the_text(read):
    with pytest.raises(SourceSyntaxError) as caught:
        read("value: !!timestamp hello\n")

    message = str(caught.value)
    assert "expected a date or timestamp, but found 'hello'" in message
    assert "quote it" not in message


@ENTRY_POINTS
def test_a_quoted_unbuildable_date_under_an_explicit_tag_gets_no_quoting_fix(read):
    with pytest.raises(SourceSyntaxError) as caught:
        read('value: !!timestamp "2026-13-45"\n')

    message = str(caught.value)
    assert "not a valid date: month must be in 1..12" in message
    assert "quote it" not in message


@pytest.mark.parametrize(
    "tag", sorted(tag for tag in _FlowLoader.yaml_constructors if tag is not None)
)
@pytest.mark.parametrize("text", ["hello", "2026-13-45", "''"])
def test_no_constructor_raises_a_non_yaml_error(tag, text):
    # Consumers turn only YAML errors into located load errors.
    try:
        yaml.load(f"value: !<{tag}> {text}\n", Loader=_FlowLoader)
    except yaml.YAMLError:
        pass
