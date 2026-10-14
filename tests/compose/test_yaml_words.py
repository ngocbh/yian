"""YAML 1.1 boolean spellings, which YAML 1.2 core rules read as text, and their fix."""

from __future__ import annotations

import pytest

from yian.compose.helpers.yaml_words import former_boolean_fix


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
