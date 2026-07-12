"""Unit tests for the additive `NodeDescriptor` base contract (Step 1).

These cover the class-level pieces the descriptor base gains before any kind is
converted onto it: the `allowed`/`required` key sets, the `allow`/`require`
extension helpers, and the parse-template hooks (`_normalize_reader` no-op,
`_validate_fields` default asserts check). Behavioral parity of `_check_keys`/
`_read_common_fields`/`parse` with the old `helpers.py` free functions is
exercised indirectly by the existing per-kind parse tests once the kinds are
converted; here we pin the new surface directly so the additive step is green
before any kind depends on it.
"""

from dataclasses import dataclass

import pytest

from agent_composer.compose.errors import LoadError
from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.compose.reader import NodeReader


def test_allow_unions_with_parent():
    assert "kind" in NodeDescriptor.allowed
    assert NodeDescriptor.allow("x") == NodeDescriptor.allowed | {"x"}


def test_require_unions_with_parent():
    assert NodeDescriptor.required == frozenset()
    assert NodeDescriptor.require("a") == frozenset({"a"})


def test_common_keys_present():
    for k in ("kind", "node_name", "depends_on", "runs_after", "env"):
        assert k in NodeDescriptor.allowed


def test_validate_fields_default_rejects_bad_asserts():
    @dataclass(frozen=True, kw_only=True)
    class D(NodeDescriptor):
        kind = "d"

    with pytest.raises(LoadError):
        D._validate_fields("n", {"asserts": [1, 2]}, {}, None)


def test_validate_fields_default_skips_absent_asserts():
    @dataclass(frozen=True, kw_only=True)
    class D(NodeDescriptor):
        kind = "d"

    D._validate_fields("n", {}, {}, None)  # no raise


def test_normalize_reader_default_noop():
    class R:
        value = {"a": 1}

    r = R()
    NodeDescriptor._normalize_reader(r)
    assert r.value == {"a": 1}


def test_parse_template_constructs_subclass_from_spanless_reader():
    """The template `parse` reads a kind's OWN keys + the commons off a spanless
    reader and constructs the subclass with them (no ranges in spanless mode)."""

    @dataclass(frozen=True, kw_only=True)
    class D(NodeDescriptor):
        kind = "d"
        allowed = NodeDescriptor.allow("greeting")
        greeting: str = ""

    reader = NodeReader.spanless(
        {"kind": "d", "greeting": "hi", "node_name": "n1", "env": {"K": "v"}}
    )
    desc = D.parse("n1", reader)
    assert isinstance(desc, D)
    assert desc.id == "n1"
    assert desc.greeting == "hi"
    assert desc.node_name == "n1"
    assert desc.env == {"K": "v"}
    assert desc.field_ranges == {}  # spanless -> no ranges recorded


def test_parse_template_rejects_unknown_key():
    """The template `parse` runs `_check_keys`, so a key outside `allowed` raises."""

    @dataclass(frozen=True, kw_only=True)
    class D(NodeDescriptor):
        kind = "d"
        allowed = NodeDescriptor.allow("greeting")
        greeting: str = ""

    reader = NodeReader.spanless({"kind": "d", "bogus": 1})
    with pytest.raises(LoadError):
        D.parse("n1", reader)
