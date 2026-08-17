"""Unit tests for converting one Compose type declaration into a `Type`."""

import pytest

from agent_composer.typesys.values import ValueKind
from agent_composer.typesys.types import read_typedefs
from agent_composer.compose import LoadError
from agent_composer.compose.helpers.types import read_type


def test_scalar():
    sh = read_type("float", {})
    assert sh.kind == ValueKind.NUMBER
    assert sh.fields is None and sh.element is None


def test_list():
    sh = read_type("list[str]", {})
    assert sh.kind == ValueKind.LIST_STRING
    assert sh.element is not None and sh.element.kind == ValueKind.STRING


def test_flat_map():
    sh = read_type({"rating": "float", "rationale": "str"}, {})
    assert sh.kind == ValueKind.OBJECT
    assert sh.fields["rating"].kind == ValueKind.NUMBER
    assert sh.fields["rationale"].kind == ValueKind.STRING
    assert sh.required == frozenset({"rating", "rationale"})


def test_nested_map():
    sh = read_type({"summary": {"count": "int", "meta": {"as_of": "date"}}}, {})
    assert sh.kind == ValueKind.OBJECT
    summary = sh.fields["summary"]
    assert summary.kind == ValueKind.OBJECT
    assert summary.fields["count"].kind == ValueKind.INTEGER
    meta = summary.fields["meta"]
    assert meta.kind == ValueKind.OBJECT
    assert meta.fields["as_of"].kind == ValueKind.DATE


def test_list_of_single_key_maps():
    listed = read_type([{"decision": "str"}, {"why": "str"}], {})
    flat = read_type({"decision": "str", "why": "str"}, {})
    assert listed == flat
    assert listed.fields.keys() == {"decision", "why"}


def test_optional_field_excluded_from_required():
    sh = read_type({"score": "float", "note": "Optional[str]"}, {})
    assert sh.fields["note"].nullable is True
    assert sh.fields["score"].nullable is False
    assert sh.required == frozenset({"score"})


def test_registry_name():
    registry = read_typedefs({"Rating": {"category": "str", "score": "float"}})
    sh = read_type("Rating", registry)
    assert sh.kind == ValueKind.OBJECT
    assert sh.fields["category"].kind == ValueKind.STRING
    assert sh.fields["score"].kind == ValueKind.NUMBER


def test_malformed_leaf_raises():
    with pytest.raises(LoadError) as exc:
        read_type("list[", {})
    assert "list[" in str(exc.value)
