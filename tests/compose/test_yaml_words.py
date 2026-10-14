"""YAML 1.1 boolean spellings, which YAML 1.2 core rules read as text, and their fix; and
what YAML read a value as."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from yian.compose.helpers.yaml_words import former_boolean_fix, yaml_read_as
from yian.typesys import type_for
from yian.typesys.types import RecordDef, TypeRegistry


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("yes", (True, "`yes` is text; write `true`")),
        ("Yes", (True, "`Yes` is text; write `true`")),
        ("YES", (True, "`YES` is text; write `true`")),
        ("on", (True, "`on` is text; write `true`")),
        ("ON", (True, "`ON` is text; write `true`")),
        ("y", (True, "`y` is text; write `true`")),
        ("no", (False, "`no` is text; write `false`")),
        ("N", (False, "`N` is text; write `false`")),
        ("Off", (False, "`Off` is text; write `false`")),
        (" off ", (False, "`off` is text; write `false`")),
    ],
)
def test_former_boolean_spelling_maps_to_its_boolean_and_fix(value, expected):
    assert former_boolean_fix(value) == expected


@pytest.mark.parametrize("value", ["true", "false", "ture", "", "yes please", True, 1, None])
def test_other_values_have_no_fix(value):
    assert former_boolean_fix(value) is None


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("bool", (True, "`yes` is text; write `true`")),
        ("Optional(bool)", (True, "`yes` is text; write `true`")),
        ("int", None),
        ("Literal(a, b)", None),
    ],
)
def test_fix_is_offered_only_when_the_boolean_fits_the_declared_type(type_text, expected):
    declared = type_for(type_text, TypeRegistry())
    assert former_boolean_fix("yes", declared) == expected


@pytest.mark.parametrize(
    ("type_text", "value", "expected"),
    [
        ("list(bool)", ["yes", "no"], (True, "`yes` is text; write `true`")),
        ("list(bool)", [True, "Off"], (False, "`Off` is text; write `false`")),
        ("Rec", {"ok": "on", "name": "yes"}, (True, "`on` is text; write `true`")),
        ("list(str)", ["yes"], None),
        ("list(bool)", ["maybe"], None),
        ("list(bool)", [1], None),
        ("Rec", {"ok": True, "name": 3}, None),
    ],
)
def test_fix_finds_the_rejected_word_inside_a_list_or_record(type_text, value, expected):
    registry = {"Rec": RecordDef(fields={"ok": "bool", "name": "str"})}
    declared = type_for(type_text, registry)
    assert former_boolean_fix(value, declared) == expected


@pytest.mark.parametrize(
    ("value", "phrase"),
    [
        (True, "a boolean"),
        (5, "an integer"),
        (1.5, "a float"),
        (datetime(2026, 10, 14, 10, 0), "a date and time"),
        (date(2026, 10, 14), "a date"),
        (None, "null"),
        (b"x", "binary data"),
        ([1], "a list"),
        ({"x": 1}, "a mapping"),
        (object(), "a `object`"),
    ],
)
def test_yaml_read_as_names_the_value(value, phrase):
    assert yaml_read_as(value) == phrase
