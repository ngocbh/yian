"""Per-declaration shape rules carried by MemoryDecl."""

from __future__ import annotations

import math

import pytest

from yian.diagnostics import OriginRef
import yian.memory as memory_package
from yian.memory.contracts import (
    _BUILTIN_MEMORY_CONFIG_RULES,
    BUILTIN_MEMORY_CONFIG_KEYS,
    MEMORY_IDENTIFIER_PATTERN,
    DeclSite,
    MemoryDecl,
    MemorySetupError,
    check_memory_declaration,
)
from yian.memory_host import _builtin

_SITE = DeclSite((), ("root",), None)


@pytest.mark.parametrize("name", ["notes", "_n", "a-b", "A9_x"])
def test_valid_memory_names_are_accepted(name: str) -> None:
    assert MemoryDecl(name, "episodic", {}, _SITE).name == name
    assert MEMORY_IDENTIFIER_PATTERN.fullmatch(name)


@pytest.mark.parametrize("name", ["", "9a", "a.b", "a b", "-a", "né"])
def test_invalid_memory_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="memory name"):
        MemoryDecl(name, "episodic", {}, _SITE)


@pytest.mark.parametrize("type_name", ["", "my.type", "1x"])
def test_invalid_memory_type_names_are_rejected(type_name: str) -> None:
    with pytest.raises(ValueError, match="memory type"):
        MemoryDecl("notes", type_name, {}, _SITE)


@pytest.mark.parametrize("key", ["storage", "store", "path"])
def test_location_keys_are_banned_for_every_type(key: str) -> None:
    with pytest.raises(ValueError, match="is not allowed"):
        MemoryDecl("notes", "vector", {key: "x"}, _SITE)


def test_api_key_anywhere_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden private key"):
        MemoryDecl("notes", "vector", {"nested": {"api_key": "x"}}, _SITE)


def test_builtin_allowlist_is_a_static_table() -> None:
    assert {k: set(v) for k, v in BUILTIN_MEMORY_CONFIG_KEYS.items()} == {
        "episodic": {"max_episodes", "recall_limit"},
        "reflective": {"max_items", "recall_limit"},
        "session": {"history_budget_chars", "recent_turns"},
        "profile": {"max_chars", "input_budget_chars"},
    }


def test_builtin_types_accept_only_allowlisted_keys() -> None:
    MemoryDecl("notes", "episodic", {"max_episodes": 5, "recall_limit": 2}, _SITE)
    with pytest.raises(
        ValueError, match=r"episodic memory config has unknown key\(s\) \['max_items'\]"
    ):
        MemoryDecl("notes", "episodic", {"max_items": 5}, _SITE)
    with pytest.raises(ValueError, match="allowed: history_budget_chars, recent_turns"):
        MemoryDecl("chat", "session", {"budget": 1}, _SITE)


@pytest.mark.parametrize(
    ("type_name", "config", "error", "message"),
    [
        ("episodic", {"max_episodes": 0}, ValueError, "max_episodes must be an integer of at least 1"),
        ("episodic", {"recall_limit": -3}, ValueError, "recall_limit must be an integer of at least 1"),
        ("reflective", {"max_items": "ten"}, TypeError, "max_items must be an integer or null"),
        ("reflective", {"max_items": 1.5}, TypeError, "max_items must be an integer or null"),
        ("reflective", {"recall_limit": True}, TypeError, "recall_limit must be an integer or null"),
        ("session", {"recent_turns": -1}, ValueError, "recent_turns must be an integer of at least 0"),
        ("session", {"history_budget_chars": 0}, ValueError, "at least 1"),
        ("session", {"recent_turns": None}, TypeError, "recent_turns must be an integer, got null"),
        ("profile", {"max_chars": 0}, ValueError, "max_chars must be an integer of at least 1"),
        ("profile", {"input_budget_chars": None}, TypeError, "input_budget_chars must be an integer"),
    ],
)
def test_builtin_config_values_are_checked(
    type_name: str, config: dict, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        check_memory_declaration("notes", type_name, config)


@pytest.mark.parametrize(
    ("type_name", "config"),
    [
        ("episodic", {"max_episodes": 1, "recall_limit": None}),
        ("reflective", {"max_items": None, "recall_limit": 1}),
        ("session", {"history_budget_chars": 1, "recent_turns": 0}),
        ("profile", {"max_chars": 1, "input_budget_chars": 1}),
    ],
)
def test_builtin_config_values_at_their_limits_load_and_build(
    type_name: str, config: dict
) -> None:
    check_memory_declaration("notes", type_name, config)
    _builtin(type_name, config, None, create_parent=False).close()


@pytest.mark.parametrize(
    ("type_name", "key", "minimum", "nullable"),
    [
        (type_name, key, minimum, nullable)
        for type_name, rules in _BUILTIN_MEMORY_CONFIG_RULES.items()
        for key, (minimum, nullable) in rules.items()
    ],
)
def test_builtin_config_rules_match_the_backends(
    type_name: str, key: str, minimum: int, nullable: bool
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _builtin(type_name, {key: minimum - 1}, None, create_parent=False)
    if not nullable:
        # A backend takes None as the yian default, but an authored null is refused.
        with pytest.raises((TypeError, ValueError)):
            check_memory_declaration("notes", type_name, {key: None})


def test_builtin_config_rules_cover_the_allowlist() -> None:
    assert {k: set(v) for k, v in _BUILTIN_MEMORY_CONFIG_RULES.items()} == {
        k: set(v) for k, v in BUILTIN_MEMORY_CONFIG_KEYS.items()
    }


def test_custom_types_accept_any_public_config() -> None:
    decl = MemoryDecl("notes", "vector", {"top_k": 3, "nested": {"x": 1}}, _SITE)
    assert decl.config == {"top_k": 3, "nested": {"x": 1}}


def test_non_finite_config_is_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryDecl("notes", "vector", {"x": math.nan}, _SITE)


def test_shape_check_is_public_and_exported() -> None:
    check_memory_declaration("notes", "episodic", {"recall_limit": 2})
    with pytest.raises(ValueError, match="memory name"):
        check_memory_declaration("a.b", "episodic", {})
    for name in ("BUILTIN_MEMORY_CONFIG_KEYS", "MEMORY_IDENTIFIER_PATTERN", "check_memory_declaration"):
        assert name in memory_package.__all__
        assert getattr(memory_package, name) is not None


def test_setup_error_carries_related_origins() -> None:
    first, second = OriginRef("a"), OriginRef("b")
    error = MemorySetupError("clash", origin=first, related=(second,))
    assert (error.origin, error.related) == (first, (second,))
    assert MemorySetupError("plain").related == ()
    with pytest.raises(TypeError):
        MemorySetupError("bad", related=[second])
    with pytest.raises(TypeError):
        MemorySetupError("bad", related=(None,))
