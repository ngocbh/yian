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

from yian.json_codec import _encode, dumps, int_text, loads, parse_int


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
    yield
    assert sys.get_int_max_str_digits() == 4300, "yian code changed the global digit limit"
    sys.set_int_max_str_digits(before)


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
    assert parse_int(int_text(number)) == number


@pytest.mark.parametrize("number", [0, -1, 2**63, 10**600])
def test_int_text_matches_str_for_small_ints(number: int) -> None:
    """Ints under the limit get Python's own spelling."""
    assert int_text(number) == str(number)


def test_int_text_of_a_long_int_matches_decimal() -> None:
    """A 5000-digit int is written in full, without changing the limit."""
    assert int_text(BIG) == format(Decimal(BIG), "f")
    assert int_text(-BIG) == "-" + format(Decimal(BIG), "f")


def test_int_text_of_an_int_subclass_is_its_value() -> None:
    """An `IntEnum` member is written as its int value, not its name."""
    assert int_text(_Level.HIGH) == "7"


def test_parse_int_follows_the_int_grammar() -> None:
    """Signs, underscore groups and surrounding whitespace are accepted."""
    assert parse_int(" +1_000 ") == 1000
    assert parse_int("1" * 5000) == int(Decimal("1" * 5000))


@pytest.mark.parametrize(
    "text", ["1.5" + "0" * 700, "1e" + "1" * 700, "1__0" + "0" * 700]
)
def test_parse_int_rejects_long_non_int_text(text: str) -> None:
    """Long text that `int()` would reject fails without echoing the text."""
    with pytest.raises(ValueError, match="invalid literal for int") as caught:
        parse_int(text)
    assert len(str(caught.value)) < 100


def test_dumps_and_loads_round_trip_a_long_int() -> None:
    """A document holding long ints is written in full and read back equal."""
    value = {"a": [BIG, -BIG, 1.5, "x", None, True]}
    digits = int_text(BIG)

    text = dumps(value, sort_keys=True, separators=(",", ":"))

    assert text == '{"a":[' + digits + ",-" + digits + ',1.5,"x",null,true]}'
    assert loads(text) == value
    assert loads(text.encode()) == value


_VALUES = [
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
    assert _outcome(_encode, value, options) == _outcome(json.dumps, value, options)


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
        dumps(float("nan"), allow_nan=False)
    with pytest.raises(ValueError, match="Circular reference detected"):
        dumps(cycle)
    with pytest.raises(ValueError, match="Circular reference detected"):
        dumps([BIG, cycle])
    with pytest.raises(ValueError):
        dumps([BIG, float("nan")], allow_nan=False)


def test_dumps_raises_type_error_for_unserializable_objects() -> None:
    """Objects json can't encode fail as in `json.dumps`, with or without a long int."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps(object())
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps([BIG, object()])
    with pytest.raises(TypeError, match="keys must be"):
        dumps({(1,): BIG})


def test_dumps_writes_long_int_keys_and_default_results() -> None:
    """Long int keys are quoted digits, and `default` results may hold long ints."""
    digits = int_text(BIG)

    assert dumps({BIG: 1}) == '{"' + digits + '": 1}'
    assert dumps([object()], default=lambda _obj: BIG) == "[" + digits + "]"


def test_loads_forwards_hooks_on_a_document_with_a_long_int() -> None:
    """Caller hooks still fire when the document holds a long int."""
    digits = int_text(BIG)

    def reject_duplicates(pairs):
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(pairs)

    def reject_constant(name):
        raise ValueError(f"constant {name}")

    with pytest.raises(ValueError, match="duplicate key"):
        loads('{"a": ' + digits + ', "a": 1}', object_pairs_hook=reject_duplicates)
    with pytest.raises(ValueError, match="constant NaN"):
        loads("[" + digits + ", NaN]", parse_constant=reject_constant)


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
