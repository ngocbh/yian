"""YAML 1.1 boolean spellings, which YAML 1.2 core rules read as text, and their fix."""

from __future__ import annotations

import pytest

from yian.compose.helpers.yaml_words import former_boolean_fix
from yian.typesys import type_for
from yian.typesys.types import TypeRegistry


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
        ("Optional[bool]", (True, "`yes` is text; write `true`")),
        ("int", None),
        ("Literal[a, b]", None),
    ],
)
def test_fix_is_offered_only_when_the_boolean_fits_the_declared_type(type_text, expected):
    declared = type_for(type_text, TypeRegistry())
    assert former_boolean_fix("yes", declared) == expected
