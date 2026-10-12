"""The two-axis `kinds` dispatch tables (`KINDS`/`BINDING_FIELDS`).

`KINDS` maps each author-written `kind:` string straight to its descriptor CLASS;
`parse`/`build` are classmethods/methods on that class, so no bundling wrapper is needed.
`start`/`end` are loader-synthesized boundaries (never author-written), so they are NOT in
`KINDS`. Adding a new author kind without a kinds/ handle fails the registration ratchet below."""

import pytest

from yian.compose.nodes import KINDS, BINDING_FIELDS
from yian.compose.nodes.agent import AgentDescriptor
from yian.compose.nodes.tool import ToolDescriptor

# The author-written surface kinds (mirrors docs/syntax.md; excludes the synthesized
# start/end boundaries). Adding a kind here without registering its handle fails below.
AUTHORED_KINDS = frozenset(
    {"agent", "code", "tool", "case", "human_input", "wait", "call", "map", "loop"}
)


def test_kinds_tables_present():
    assert isinstance(KINDS, dict)
    assert isinstance(BINDING_FIELDS, dict)


def test_kinds_maps_string_to_descriptor_class():
    # KINDS is kind-string -> descriptor class; parse/build ride on the class itself.
    assert KINDS["agent"] is AgentDescriptor
    assert KINDS["tool"] is ToolDescriptor
    for kind, cls in KINDS.items():
        assert isinstance(cls, type)
        assert cls.kind == kind


def test_retired_kindspec_is_not_public():
    with pytest.raises(ImportError):
        from yian.compose.nodes import KindSpec  # noqa: F401


def test_all_authored_kinds_registered():
    assert AUTHORED_KINDS <= set(KINDS), sorted(AUTHORED_KINDS - set(KINDS))
    assert AUTHORED_KINDS <= set(BINDING_FIELDS), sorted(AUTHORED_KINDS - set(BINDING_FIELDS))


def test_kinds_and_binding_fields_agree():
    # Every registered kind carries a binding-fields entry (possibly empty), and vice versa.
    assert set(KINDS) == set(BINDING_FIELDS)
    for kind, cls in KINDS.items():
        assert BINDING_FIELDS[kind] == cls.binding_fields
