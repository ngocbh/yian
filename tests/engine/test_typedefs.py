"""Unit tests for read_typedefs — the `typedefs:` registry builder."""

import pytest

from yian.typesys.values import TypeCheckError, ValueKind
from yian.typesys.types import (
    AliasDef,
    RecordDef,
    RefExpr,
    VariantDef,
    read_typedefs,
    resolve_type,
    type_for,
)


def test_record_enum_alias_build():
    reg = read_typedefs({
        "Topic": "str",
        "Amount": "float",
        "Category": "Literal[pro, con, mixed]",
        "Rating": {"category": "Category", "score": "float", "note": "Optional[str]"},
    })
    assert isinstance(reg["Topic"], AliasDef)
    assert isinstance(reg["Amount"], AliasDef)
    assert isinstance(reg["Category"], VariantDef) and reg["Category"].tags == ("pro", "con", "mixed")
    assert isinstance(reg["Rating"], RecordDef)


def test_alias_composes_alias():
    reg = read_typedefs({"Topic": "str", "Bundle": "list[Topic]"})
    assert type_for("Bundle", reg).kind == ValueKind.LIST_STRING


def test_bounded_aliases_preserve_nullable_and_nested_value_paths():
    from yian.typesys.values import build_value_as

    registry = read_typedefs({
        "Score": "float[0, 1]", "MaybeScore": "Optional[Score]",
        "Rating": {"scores": "list[MaybeScore]", "count": "int[1, 12]"},
        "Ratings": "list[Rating]",
    })
    typ = type_for("Ratings", registry)
    typ.validate_shape()
    value = [{"scores": [None, 0.5, 1], "count": 4}]
    assert build_value_as(typ, value).to_object() == value
    assert build_value_as(type_for("MaybeScore", registry), None).to_object() is None
    for value, path in (
        ([{"scores": [None, 1.1], "count": 4}], (0, "scores", 1)),
        ([{"scores": [0], "count": 0}], (0, "count")),
    ):
        with pytest.raises(TypeCheckError) as caught:
            build_value_as(typ, value)
        assert caught.value.path == path


def test_record_resolves_with_enum_and_optional():
    reg = read_typedefs({
        "Category": "Literal[pro, con]",
        "Rating": {"category": "Category", "score": "float", "note": "Optional[str]"},
    })
    sh = type_for("Rating", reg)
    assert sh.required == frozenset({"category", "score"})  # note (Optional) is not required
    assert sh.fields["category"].tags == frozenset({"pro", "con"})
    assert sh.fields["note"].nullable is True


def test_all_bare_tags_sequence_rejected_e05():
    # e05 mechanism: a tag-only union must be Literal[...], not a sequence.
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Category": ["pro", "con"]})
    assert "Literal" in str(ei.value)


def test_payload_union_sequence_rejected():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Choice": ["defer", {"approve": {"count": "int"}}]})
    assert "discriminated record" in str(ei.value) or "case" in str(ei.value)


def test_alias_cycle_rejected_eagerly():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"A": "B", "B": "A"})
    assert "cycle" in str(ei.value)


def test_recursive_record_rejected_lazily():
    # read_typedefs keeps record field types raw, so the build is fine; the cycle is
    # caught lazily at resolve via resolve_type's `_seen` guard.
    reg = read_typedefs({"R": {"x": "R"}})
    with pytest.raises(TypeCheckError):
        resolve_type(RefExpr("R"), reg)


def test_shadow_name_rejected():
    with pytest.raises(TypeCheckError):
        read_typedefs({"str": "int"})
    with pytest.raises(TypeCheckError):
        read_typedefs({"Optional": "int"})


def test_non_pascalcase_rejected():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"rating": {"x": "int"}})
    assert "PascalCase" in str(ei.value)


def test_record_field_must_be_string():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Bad": {"x": {"nested": "int"}}})  # inline nested object not allowed here
    assert "string expression" in str(ei.value)
