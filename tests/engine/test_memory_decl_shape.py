"""Per-declaration shape rules carried by MemoryDecl."""

from __future__ import annotations

import math

import pytest

from agent_composer.diagnostics import OriginRef
import agent_composer.memory as memory_package
from agent_composer.memory.contracts import (
    BUILTIN_MEMORY_CONFIG_KEYS,
    MEMORY_IDENTIFIER_PATTERN,
    DeclSite,
    MemoryDecl,
    MemorySetupError,
    check_memory_declaration,
)

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
    with pytest.raises(ValueError, match="not allowed|forbidden private key"):
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
