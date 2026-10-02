"""The long_term_memory parser: sugar, mapping, long form, and shape errors."""

from __future__ import annotations

import pytest

from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.compose.helpers.memory import parse_memory_declarations
from agent_composer.memory.contracts import DeclSite

_SITE = DeclSite((), ("root",), None)


def _parse(value):
    return parse_memory_declarations(
        value, diagnostics=LoadDiagnostics(), source_range=None, site=_SITE
    )


def _shape(decls):
    return [(d.name, d.type, d.config, d.site) for d in decls]


def test_sugar_is_one_memory_named_after_its_type() -> None:
    assert _shape(_parse("episodic")) == [("episodic", "episodic", {}, _SITE)]


def test_mapping_and_long_form_keep_authored_order() -> None:
    value = {
        "lessons": "reflective",
        "notes": {"type": "episodic", "config": {"max_episodes": 50}},
        "plain": {"type": "vector"},
    }
    assert _shape(_parse(value)) == [
        ("lessons", "reflective", {}, _SITE),
        ("notes", "episodic", {"max_episodes": 50}, _SITE),
        ("plain", "vector", {}, _SITE),
    ]
    assert all(decl.origin is None for decl in _parse(value))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "declares no memory"),
        (3, "type name or a mapping"),
        (["episodic"], "type name or a mapping"),
        ({"name": "reflective", "config": {"max_items": 3}}, "reserved"),
        ({"name": "reflective"}, "reserved"),
        ({"type": "episodic"}, "reserved"),
        ({"notes": {"type": "episodic", "extra": 1}}, "unknown key"),
        ({"notes": {"config": {}}}, "requires 'type'"),
        ({"notes": {"type": 3}}, "type must be a string"),
        ({"notes": {"type": "episodic", "config": []}}, "config must be a mapping"),
        ({"notes": 3}, "type name or a mapping"),
        ({3: "episodic"}, "must be a string"),
        ({"9notes": "episodic"}, "memory name"),
        ({"notes": "my.type"}, "memory type"),
        ("", "memory name"),
        ({"notes": {"type": "episodic", "config": {"path": "x.sqlite"}}}, "not allowed|forbidden"),
        ({"notes": {"type": "episodic", "config": {"max_items": 3}}}, "unknown key"),
    ],
)
def test_malformed_values_are_load_errors(value, message: str) -> None:
    with pytest.raises(LoadError, match=message):
        _parse(value)
