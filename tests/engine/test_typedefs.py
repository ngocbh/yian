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
        "Category": "Literal(pro, con, mixed)",
        "Rating": {"category": "Category", "score": "float", "note": "Optional(str)"},
    })
    assert isinstance(reg["Topic"], AliasDef)
    assert isinstance(reg["Amount"], AliasDef)
    assert isinstance(reg["Category"], VariantDef) and reg["Category"].tags == ("pro", "con", "mixed")
    assert isinstance(reg["Rating"], RecordDef)


def test_alias_composes_alias():
    reg = read_typedefs({"Topic": "str", "Bundle": "list(Topic)"})
    assert type_for("Bundle", reg).kind == ValueKind.LIST_STRING


def test_bounded_aliases_preserve_nullable_and_nested_value_paths():
    from yian.typesys.values import build_value_as

    registry = read_typedefs({
        "Score": "float(0..1)", "MaybeScore": "Optional(Score)",
        "Rating": {"scores": "list(MaybeScore)", "count": "int(1..12)"},
        "Ratings": "list(Rating)",
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
        "Category": "Literal(pro, con)",
        "Rating": {"category": "Category", "score": "float", "note": "Optional(str)"},
    })
    sh = type_for("Rating", reg)
    assert sh.required == frozenset({"category", "score"})  # note (Optional) is not required
    assert sh.fields["category"].tags == frozenset({"pro", "con"})
    assert sh.fields["note"].nullable is True


def test_all_bare_tags_sequence_rejected_e05():
    # e05 mechanism: a tag-only union must be Literal(...), not a sequence.
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Category": ["pro", "con"]})
    assert "Literal" in str(ei.value)


def test_payload_union_sequence_rejected():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Choice": ["defer", {"approve": {"count": "int"}}]})
    assert "Choice: Union(" in str(ei.value)


def test_alias_cycle_rejected_eagerly():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"A": "B", "B": "A"})
    assert "cycle" in str(ei.value)


def test_recursive_record_rejected_eagerly():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"R": {"x": "R"}})
    assert "typedef 'R' field 'x': recursive type reference: 'R'" in str(ei.value)
    assert ei.value.path == ("R", "x")


def test_unused_record_with_unknown_constructor_rejected():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Bad": {"x": "lst(int)"}})
    assert "typedef 'Bad' field 'x': unknown type constructor 'lst'" in str(ei.value)
    assert ei.value.path == ("Bad", "x")


def test_unused_record_with_unknown_name_rejected():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Order": {"pick": "Nope", "note": "str"}})
    assert "typedef 'Order' field 'pick': unknown type 'Nope'" in str(ei.value)


def test_unused_record_with_bad_bounds_rejected():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Order": {"qty": "int(5..1)"}})
    assert "typedef 'Order' field 'qty':" in str(ei.value)


def test_unused_alias_to_unknown_name_rejected():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Al": "Nope"})
    assert "typedef 'Al': unknown type 'Nope'" in str(ei.value)
    assert ei.value.path == ("Al",)


def test_mutually_recursive_records_rejected():
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"A": {"b": "B"}, "B": {"a": "A"}})
    assert "recursive type reference" in str(ei.value)


def test_error_names_the_broken_typedef_not_its_user():
    # `Wrapper` is listed first and uses `Signal`; the fault is Signal's own field.
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({
            "Wrapper": {"signal": "Signal"},
            "Signal": {"score": "float", "tags": "lst(str)"},
        })
    assert "typedef 'Signal' field 'tags'" in str(ei.value)
    assert ei.value.path == ("Signal", "tags")


def test_record_resolves_through_other_typedefs():
    reg = read_typedefs({
        "Wrapper": {"signal": "Signal", "all": "list(Signal)"},
        "Signal": {"score": "float", "tags": "list(str)"},
    })
    assert type_for("Wrapper", reg).fields["signal"].fields["tags"].kind == ValueKind.LIST_STRING


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


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"": "int"}, "typedef 'R' field '': field names must be non-empty strings"),
        ({1: "int"}, "typedef 'R' field 1: field names must be non-empty strings"),
        ({"\ud800": "int"}, "field names must not contain a Unicode surrogate"),
    ],
)
def test_record_field_name_rejected(fields, message):
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"R": fields})
    assert message in str(ei.value)


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ('Literal("", fast)', "empty Literal member"),
        ('Literal("\ud800", fast)', "contains a Unicode surrogate"),
    ],
)
def test_enum_tag_rejected(expr, message):
    with pytest.raises(TypeCheckError) as ei:
        read_typedefs({"Mode": expr})
    assert message in str(ei.value)
