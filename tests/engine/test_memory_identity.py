"""Contracts for memory declaration sites, declarations, specs, and stacks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent_composer.memory import DeclSite


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
