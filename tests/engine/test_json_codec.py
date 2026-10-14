"""Contracts for the JSON codec that reads and writes ints of any size."""

from __future__ import annotations

import ast
from decimal import Decimal
import enum
import itertools
import json
from pathlib import Path
import sys

import pytest

import yian.json_codec as json_codec


BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None:
        yield  # this Python has no limit
        return
    before = get()
    sys.set_int_max_str_digits(4300)
    try:
        yield
        after = get()
    finally:
        sys.set_int_max_str_digits(before)
    assert after == 4300, "yian code changed the global digit limit"


class _Level(enum.IntEnum):
    HIGH = 7


@pytest.mark.parametrize(
    "number",
    [0, -1, 2**63, BIG, -BIG, 10**600, 10**601, -(10**601)],
    # pytest's default ids call str(), which fails on a 5000-digit int.
    ids=["0", "-1", "2**63", "BIG", "-BIG", "10**600", "10**601", "-(10**601)"],
)
def test_int_text_and_parse_int_round_trip(number: int) -> None:
    """Every int converts to decimal text and back unchanged."""
    assert json_codec.parse_int(json_codec.int_text(number)) == number


@pytest.mark.parametrize("number", [0, -1, 2**63, 10**600])
def test_int_text_matches_str_for_small_ints(number: int) -> None:
    """Ints under the limit get Python's own spelling."""
    assert json_codec.int_text(number) == str(number)


def test_int_text_of_a_long_int_matches_decimal() -> None:
    """A 5000-digit int is written in full, without changing the limit."""
    assert json_codec.int_text(BIG) == format(Decimal(BIG), "f")
    assert json_codec.int_text(-BIG) == "-" + format(Decimal(BIG), "f")


def test_int_text_of_an_int_subclass_is_its_value() -> None:
    """An `IntEnum` member is written as its int value, not its name."""
    assert json_codec.int_text(_Level.HIGH) == "7"


def test_parse_int_follows_the_int_grammar() -> None:
    """Signs, underscore groups and surrounding whitespace are accepted."""
    assert json_codec.parse_int(" +1_000 ") == 1000
    assert json_codec.parse_int("1" * 5000) == int(Decimal("1" * 5000))
    assert json_codec.parse_int(" \n" + "1" * 700 + "\t") == int(Decimal("1" * 700))


def test_parse_int_reads_short_text_as_int_does() -> None:
    """Short text accepts whatever `int()` accepts, including non-ASCII digits."""
    assert json_codec.parse_int("\u0661\u0662\u0663") == 123
    assert json_codec.parse_int("\u00a0-12\u2003") == -12


@pytest.mark.parametrize("text", ["12.5", "1__0", "", " "])
def test_parse_int_rejects_short_malformed_text(text: str) -> None:
    """Short text that `int()` rejects fails with its error."""
    with pytest.raises(ValueError, match="invalid literal for int"):
        json_codec.parse_int(text)


@pytest.mark.parametrize(
    "text", ["1.5" + "0" * 700, "1e" + "1" * 700, "1__0" + "0" * 700]
)
def test_parse_int_rejects_long_non_int_text(text: str) -> None:
    """Long text that `int()` would reject fails without echoing the text."""
    with pytest.raises(ValueError, match="invalid literal for int") as caught:
        json_codec.parse_int(text)
    assert len(str(caught.value)) < 100


def test_dumps_and_loads_round_trip_a_long_int() -> None:
    """A document holding long ints is written in full and read back equal."""
    value = {"a": [BIG, -BIG, 1.5, "x", None, True]}
    digits = json_codec.int_text(BIG)

    text = json_codec.dumps(value, sort_keys=True, separators=(",", ":"))

    assert text == '{"a":[' + digits + ",-" + digits + ',1.5,"x",null,true]}'
    assert json_codec.loads(text) == value
    assert json_codec.loads(text.encode()) == value


_SHARED = [1]

_VALUES = [
    [_SHARED, _SHARED, {"a": _SHARED}],
    {"b": [1, 2.5, {"c": None}], "a": {"d": [True, False, []]}},
    [1, (2, 3), ("x", ["y", ()])],
    {"text": "héllo ☃ \U0001f600 \"quoted\" \\ \n\t"},
    [[], {}, (), [{}], {"e": []}],
    [1.0, 1e300, -0.0, 0.1, -2.5e-10],
    [True, False, None, 0, -7],
    {1: "int key", 2.5: "float key", None: "none key", True: "bool key"},
    {"z": 1, "y": {"x": 2, "w": [3, {"v": 4}]}},
    {"obj": [object(), 1]},
    "a scalar é",
    3,
    [],
    {},
]

_OPTIONS = [
    {},
    {"sort_keys": True},
    {"separators": (",", ":")},
    {"indent": 2},
    {"indent": "\t"},
    {"indent": 2, "sort_keys": True, "separators": (", ", " = ")},
    {"ensure_ascii": False},
    {"ensure_ascii": False, "indent": 2},
    {"default": str},
    {"default": str, "indent": 2, "sort_keys": True},
]


def _outcome(encode, value, options):
    try:
        return "ok", encode(value, **options)
    except (TypeError, ValueError) as exc:
        return type(exc), str(exc)


@pytest.mark.parametrize(
    ("value", "options"),
    list(itertools.product(_VALUES, _OPTIONS)),
)
def test_slow_path_encoder_matches_json_dumps(value, options) -> None:
    """The long-int encoder writes the same bytes, or raises the same error, as `json.dumps`."""
    assert _outcome(json_codec._encode, value, options) == _outcome(json.dumps, value, options)


@pytest.mark.parametrize("options", _OPTIONS)
def test_dumps_of_long_ints_matches_json_dumps_without_the_limit(options) -> None:
    """Long ints as values, nested values and keys, and an `IntEnum`, match unlimited json."""
    value = {BIG: [BIG, {"k": -BIG}], 3: _Level.HIGH}
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        expected = json.dumps(value, **options)
    finally:
        sys.set_int_max_str_digits(before)

    assert json_codec.dumps(value, **options) == expected


def test_slow_path_comparison_covers_successful_encodings() -> None:
    """Most value/option pairs encode, so the comparison checks real output."""
    successes = sum(
        _outcome(json.dumps, value, options)[0] == "ok"
        for value, options in itertools.product(_VALUES, _OPTIONS)
    )
    assert successes >= len(_VALUES) * len(_OPTIONS) // 2


def test_dumps_raises_value_error_for_nan_and_cycles() -> None:
    """Non-digit-limit `ValueError`s are raised again, with or without a long int."""
    cycle: list = []
    cycle.append(cycle)

    with pytest.raises(ValueError):
        json_codec.dumps(float("nan"), allow_nan=False)
    with pytest.raises(ValueError, match="Circular reference detected"):
        json_codec.dumps(cycle)
    with pytest.raises(ValueError, match="Circular reference detected"):
        json_codec.dumps([BIG, cycle])
    with pytest.raises(ValueError):
        json_codec.dumps([BIG, float("nan")], allow_nan=False)


def test_dumps_raises_type_error_for_unserializable_objects() -> None:
    """Objects json can't encode fail as in `json.dumps`, with or without a long int."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_codec.dumps(object())
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_codec.dumps([BIG, object()])
    with pytest.raises(TypeError, match="keys must be"):
        json_codec.dumps({(1,): BIG})


def test_dumps_writes_long_int_keys_and_default_results() -> None:
    """Long int keys are quoted digits, and `default` results may hold long ints."""
    digits = json_codec.int_text(BIG)

    assert json_codec.dumps({BIG: 1}) == '{"' + digits + '": 1}'
    assert json_codec.dumps([object()], default=lambda _obj: BIG) == "[" + digits + "]"


def test_loads_forwards_hooks_on_a_document_with_a_long_int() -> None:
    """Caller hooks still fire when the document holds a long int."""
    digits = json_codec.int_text(BIG)

    def reject_duplicates(pairs):
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(pairs)

    def reject_constant(name):
        raise ValueError(f"constant {name}")

    with pytest.raises(ValueError, match="duplicate key"):
        json_codec.loads('{"a": ' + digits + ', "a": 1}', object_pairs_hook=reject_duplicates)
    with pytest.raises(ValueError, match="constant NaN"):
        json_codec.loads("[" + digits + ", NaN]", parse_constant=reject_constant)


def test_loads_reads_a_long_int_after_the_fast_path_fails() -> None:
    """A document with a long int deep inside still loads, with hooks applied."""
    digits = json_codec.int_text(BIG)
    seen: list[str] = []

    def record_constant(name):
        seen.append(name)
        return name

    value = json_codec.loads(
        '{"b": [1, ' + digits + '], "a": NaN}',
        object_pairs_hook=lambda pairs: sorted(pairs),
        parse_constant=record_constant,
    )

    assert value == [("a", "NaN"), ("b", [1, BIG])]
    assert "NaN" in seen


def test_loads_uses_hooks_on_documents_without_long_ints() -> None:
    """The plain path forwards hooks too."""
    value = json_codec.loads(
        '{"b": 1, "a": NaN}',
        object_pairs_hook=lambda pairs: sorted(pairs),
        parse_constant=lambda name: name,
    )

    assert value == [("a", "NaN"), ("b", 1)]


def test_loads_parses_short_ints_without_the_codec(monkeypatch) -> None:
    """Documents without long ints take `json.loads` directly."""

    def fail(text):
        raise AssertionError("the long-int parser ran")

    monkeypatch.setattr(json_codec, "parse_int", fail)

    assert json_codec.loads('{"a": [1, -2, 3]}') == {"a": [1, -2, 3]}


def test_loads_accepts_an_explicit_parse_int() -> None:
    """A caller's `parse_int` replaces the default one."""
    assert json_codec.loads("[1, 2]", parse_int=lambda text: text) == ["1", "2"]


@pytest.mark.parametrize(
    "text", ['{"a": 1,', "[1, 2", "nope"], ids=["object", "list", "bare"]
)
def test_loads_raises_json_decode_error_for_malformed_documents(text: str) -> None:
    """Malformed JSON raises `JSONDecodeError`, with or without a long int."""
    with pytest.raises(json.JSONDecodeError):
        json_codec.loads(text)
    with pytest.raises(json.JSONDecodeError):
        json_codec.loads("[" + json_codec.int_text(BIG) + ", " + text)


def test_json_codec_module_imports_only_the_standard_library() -> None:
    """The codec sits below every project package."""
    path = Path(__file__).parents[2] / "src" / "yian" / "json_codec.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    relative_imports = [
        node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level
    ]
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert not relative_imports
    assert imports <= {"__future__", "decimal", "json", "re", "typing"}
