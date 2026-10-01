"""Contracts for memory declaration sites, declarations, specs, and stacks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent_composer.diagnostics import OriginRef
from agent_composer.memory import DeclSite, MemoryDecl, MemorySpec, MemoryStack
from agent_composer.memory.contracts import (
    _iter_memory_decl_origins,
    _rewrite_memory_decl_origins,
)


def test_decl_site_validates_and_freezes() -> None:
    site = DeclSite((), ("root",), "ask")
    assert (site.scope, site.owner, site.node) == ((), ("root",), "ask")
    assert DeclSite((), ("def", "child"), None).node is None
    with pytest.raises(FrozenInstanceError):
        site.node = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        DeclSite([], ("root",), None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeclSite((), (), None)
    with pytest.raises(ValueError):
        DeclSite((), ("root",), "")


def test_decl_site_sort_is_total_and_puts_flow_before_agent() -> None:
    sites = [
        DeclSite((), ("root",), "b"),
        DeclSite(("lib",), ("root",), None),
        DeclSite((), ("root",), None),
        DeclSite((), ("def", "child"), "a"),
        DeclSite((), ("root",), "a"),
    ]
    ordered = sorted(sites, key=DeclSite.sort_key)
    assert ordered == [
        DeclSite((), ("def", "child"), "a"),
        DeclSite((), ("root",), None),
        DeclSite((), ("root",), "a"),
        DeclSite((), ("root",), "b"),
        DeclSite(("lib",), ("root",), None),
    ]


def _decl(name: str = "notes", *, node: str | None = None, origin=None) -> MemoryDecl:
    return MemoryDecl(name, "episodic", {"recall_limit": 3}, DeclSite((), ("root",), node), origin)


def test_memory_decl_snapshots_config_and_rejects_private_keys() -> None:
    config = {"recall_limit": 3}
    decl = MemoryDecl("notes", "episodic", config, DeclSite((), ("root",), None))
    config["recall_limit"] = 9
    assert decl.config == {"recall_limit": 3}
    decl.config["recall_limit"] = 7
    assert decl.config == {"recall_limit": 3}
    assert decl.origin is None
    with pytest.raises(ValueError, match="forbidden private key"):
        MemoryDecl("notes", "episodic", {"api_key": "x"}, DeclSite((), ("root",), None))
    with pytest.raises(TypeError, match="site must be a DeclSite"):
        MemoryDecl("notes", "episodic", {}, ((), ("root",), None))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MemoryDecl("", "episodic", {}, DeclSite((), ("root",), None))
    with pytest.raises(ValueError):
        MemoryDecl("notes", "", {}, DeclSite((), ("root",), None))


def test_memory_decl_origins_traverse_and_rewrite_shared_carriers_once() -> None:
    shared = _decl(origin=OriginRef("a"))
    other = _decl("other", node="ask", origin=OriginRef("b"))
    seen: set[int] = set()
    assert list(_iter_memory_decl_origins((shared, other, shared), seen)) == [
        OriginRef("a"),
        OriginRef("b"),
    ]
    calls: list[str] = []

    def mapper(origin: OriginRef) -> OriginRef:
        calls.append(origin.key)
        return OriginRef("mapped/" + origin.key)

    _rewrite_memory_decl_origins((shared, other, shared), mapper, set())
    assert calls == ["a", "b"]
    assert shared.origin == OriginRef("mapped/a")


def test_memory_spec_requires_sorted_owners_and_derived_eligibility() -> None:
    agent = DeclSite((), ("root",), "ask")
    flow = DeclSite((), ("root",), None)
    spec = MemorySpec("chat", "session", {}, (agent,), True)
    assert spec.conversation_eligible is True
    assert MemorySpec("notes", "episodic", {}, (flow, agent), False).owners == (flow, agent)
    with pytest.raises(ValueError, match="conversation_eligible"):
        MemorySpec("chat", "session", {}, (flow,), True)
    with pytest.raises(ValueError, match="conversation_eligible"):
        MemorySpec("chat", "session", {}, (agent,), False)
    with pytest.raises(ValueError, match="sorted and unique"):
        MemorySpec("notes", "episodic", {}, (agent, flow), False)
    with pytest.raises(ValueError, match="non-empty"):
        MemorySpec("notes", "episodic", {}, (), False)
    with pytest.raises(TypeError):
        MemorySpec("notes", "episodic", {}, (agent,), 1)  # type: ignore[arg-type]


def test_memory_stack_requires_locator_and_unique_keys() -> None:
    stack = MemoryStack(("child", "ask"), ("notes", "chat"))
    assert stack.memories == ("notes", "chat")
    assert MemoryStack(("plain",), ()).memories == ()
    with pytest.raises(ValueError):
        MemoryStack((), ("notes",))
    with pytest.raises(ValueError, match="unique"):
        MemoryStack(("ask",), ("notes", "notes"))
