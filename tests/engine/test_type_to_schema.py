"""`type_to_schema` — derive a pydantic model from a declared output `Type`."""

import pytest
from pydantic import ValidationError

from yian.nodes.agent.structured import type_to_schema
from yian.typesys.values import Type, ValueKind


def test_bare_str_returns_none():
    # scalar str stays today's text passthrough — no schema
    assert type_to_schema(Type.scalar(ValueKind.STRING)) is None


def test_variant_str_gets_scalar_schema():
    typ = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b"}))
    model = type_to_schema(typ)

    assert model is not None
    assert model.model_validate({"value": "a"}).value == "a"
    with pytest.raises(ValidationError):
        model.model_validate({"value": "not-a-tag"})


def test_scalar_int_gets_schema():
    model = type_to_schema(Type.scalar(ValueKind.INTEGER))
    assert model is not None
    inst = model.model_validate({"value": 7})  # single-field wrapper named "value"
    assert inst.value == 7


@pytest.mark.parametrize("expression,lower,upper,inside,outside", [
    ("int[1, 12]", 1, 12, 4, 13),
    ("float[-0.5, 1]", -0.5, 1, 0.4, 1.1),
    ("int[4, 4]", 4, 4, 4, 3),
])
def test_bounded_scalar_schema_exposes_and_enforces_inclusive_limits(
    expression, lower, upper, inside, outside,
):
    from yian.typesys.types import type_for

    model = type_to_schema(type_for(expression, {}))
    value_schema = model.model_json_schema()["properties"]["value"]
    assert (value_schema["minimum"], value_schema["maximum"]) == (lower, upper)
    for value in (lower, inside, upper):
        assert model.model_validate({"value": value}).value == value
    with pytest.raises(ValidationError):
        model.model_validate({"value": outside})


def test_bounded_numeric_schemas_preserve_nested_lists_records_and_nullable_fields():
    from yian.typesys.types import read_typedefs, type_for

    registry = read_typedefs({"Rating": {
        "count": "int[1, 12]", "scores": "list[list[float[0, 1]]]",
        "confidence": "Optional[float[0, 1]]",
    }})
    model = type_to_schema(type_for("list[Rating]", registry))
    schema = model.model_json_schema()
    record_name = schema["properties"]["items"]["items"]["$ref"].rsplit("/", 1)[-1]
    fields = schema["$defs"][record_name]["properties"]
    assert fields["count"]["minimum"] == 1 and fields["count"]["maximum"] == 12
    scalar = fields["scores"]["items"]["items"]
    assert scalar["minimum"] == 0 and scalar["maximum"] == 1
    numeric = next(arm for arm in fields["confidence"]["anyOf"] if arm["type"] == "number")
    assert numeric["minimum"] == 0 and numeric["maximum"] == 1
    model.model_validate({"items": [{"count": 4, "scores": [[0, 1]], "confidence": None}]})
    with pytest.raises(ValidationError) as caught:
        model.model_validate({"items": [{"count": 4, "scores": [[0, 1.1]]}]})
    assert caught.value.errors()[0]["loc"] == ("items", 0, "scores", 0, 1)


def test_nullable_bounded_scalars_keep_explicit_null_in_schema():
    from yian.typesys.types import type_for

    model = type_to_schema(type_for("list[Optional[int[1, 12]]]", {}))
    items_schema = model.model_json_schema()["properties"]["items"]["items"]
    numeric = next(arm for arm in items_schema["anyOf"] if arm["type"] == "integer")
    assert numeric["minimum"] == 1 and numeric["maximum"] == 12
    assert model.model_validate({"items": [None, 4]}).items == [None, 4]


@pytest.mark.parametrize("kind,lower,upper,outside", [
    ("int", 10**400, 10**400 + 2, 10**400 + 3),
    ("float", 2**53, 2**53 + 2, 2**53 + 4),
    ("float", -(2**53 + 2), -(2**53), -(2**53 + 4)),
])
def test_large_numeric_schema_bounds_match_resolved_type(kind, lower, upper, outside):
    from yian.typesys.types import type_for
    from yian.typesys.values import TypeCheckError, build_value_as

    typ = type_for(f"{kind}[{lower}, {upper}]", {})
    typ.validate_shape()
    model = type_to_schema(typ)
    value_schema = model.model_json_schema()["properties"]["value"]
    assert (value_schema["minimum"], value_schema["maximum"]) == (lower, upper)
    for value in (lower, upper):
        assert build_value_as(typ, value).to_object() == value
        assert model.model_validate({"value": value}).value == value
    with pytest.raises(TypeCheckError):
        build_value_as(typ, outside)
    with pytest.raises(ValidationError):
        model.model_validate({"value": outside})


def test_generic_list_schema_honors_existing_unbounded_element_refinements():
    from yian.typesys.types import type_for

    model = type_to_schema(type_for("list[list[int]]", {}))
    scalar_schema = model.model_json_schema()["properties"]["items"]["items"]["items"]
    assert scalar_schema == {"type": "integer"}
    assert model.model_validate({"items": [[1], [2]]}).items == [[1], [2]]
    with pytest.raises(ValidationError) as caught:
        model.model_validate({"items": [[1], ["not-an-int"]]})
    assert caught.value.errors()[0]["loc"] == ("items", 1, 0)

    bare = type_to_schema(Type(ValueKind.LIST_ANY))
    assert bare.model_json_schema()["properties"]["items"]["items"] == {}
    assert bare.model_validate({"items": [1, "free", {}]}).items == [1, "free", {}]


def test_record_type():
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={
            "name": Type.scalar(ValueKind.STRING),
            "score": Type.scalar(ValueKind.NUMBER),
        },
        required=frozenset({"name"}),
    )
    model = type_to_schema(typ)
    inst = model.model_validate({"name": "a", "score": 1.5})
    assert inst.name == "a" and inst.score == 1.5


def test_list_of_records():
    elem = Type(
        kind=ValueKind.OBJECT,
        fields={"x": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"x"}),
    )
    typ = Type(kind=ValueKind.LIST_OBJECT, element=elem)
    model = type_to_schema(typ)
    inst = model.model_validate({"items": [{"x": 1}, {"x": 2}]})
    assert [i.x for i in inst.items] == [1, 2]


# --------------------------------------------------------------------------- #
# Unions of records
# --------------------------------------------------------------------------- #


_TRADES = {
    "Buy": {"action": "Literal[buy]", "qty": "int"},
    "Sell": {"action": "Literal[sell]", "price": "float"},
    "Text": {"text": "str"},
    "Call": {"name": "str", "input": "object"},
}


def _union_arms(schema, arm_schema):
    """Return the member `$defs` an `anyOf` arm list refers to, in order."""
    assert "oneOf" not in arm_schema and "discriminator" not in arm_schema
    refs = [arm["$ref"] for arm in arm_schema["anyOf"] if "$ref" in arm]
    return [schema["$defs"][ref.rsplit("/", 1)[-1]] for ref in refs]


def _assert_member_defs(schema, members, size):
    assert len(members) == size
    assert all(member["additionalProperties"] is False for member in members)
    assert len({member["title"] for member in members}) == size
    assert "discriminator" not in str(schema) and "oneOf" not in str(schema)


def test_tagged_union_schema_is_any_of_closed_members_with_const_tags():
    from yian.typesys.types import read_typedefs, type_for

    union = type_for("Union[Buy, Sell]", read_typedefs(_TRADES))
    model = type_to_schema(union)
    assert model.__name__ == "ScalarWrapper"
    schema = model.model_json_schema()
    members = _union_arms(schema, schema["properties"]["value"])
    _assert_member_defs(schema, members, 2)
    assert [member["properties"]["action"]["const"] for member in members] == ["buy", "sell"]


def test_untagged_union_schema_is_any_of_closed_members():
    from yian.typesys.types import read_typedefs, type_for

    model = type_to_schema(type_for("Union[Text, Call]", read_typedefs(_TRADES)))
    schema = model.model_json_schema()
    members = _union_arms(schema, schema["properties"]["value"])
    _assert_member_defs(schema, members, 2)
    assert sorted(sorted(member["properties"]) for member in members) == [
        ["input", "name"], ["text"],
    ]


def test_record_field_of_union_type_gets_member_schemas():
    from yian.typesys.types import read_typedefs, type_for

    registry = read_typedefs({**_TRADES, "Order": {"trade": "Union[Buy, Sell]", "note": "str"}})
    model = type_to_schema(type_for("Order", registry))
    schema = model.model_json_schema()
    members = _union_arms(schema, schema["properties"]["trade"])
    _assert_member_defs(schema, members, 2)
    assert [member["properties"]["action"]["const"] for member in members] == ["buy", "sell"]


def test_list_of_union_schema_gets_member_schemas():
    from yian.typesys.types import read_typedefs, type_for

    model = type_to_schema(type_for("list[Union[Buy, Sell]]", read_typedefs(_TRADES)))
    assert model.__name__ == "ListWrapper"
    schema = model.model_json_schema()
    members = _union_arms(schema, schema["properties"]["items"]["items"])
    _assert_member_defs(schema, members, 2)


def test_union_members_reject_unknown_fields_and_unwrap_to_the_member_dict():
    from yian.nodes.agent.structured import _unwrap
    from yian.typesys.types import read_typedefs, type_for

    union = type_for("Union[Buy, Sell]", read_typedefs(_TRADES))
    model = type_to_schema(union)
    reply = model.model_validate({"value": {"action": "sell", "price": 1.5}})
    assert _unwrap(reply, union) == {"action": "sell", "price": 1.5}
    with pytest.raises(ValidationError):
        model.model_validate({"value": {"action": "buy", "qty": 2, "zzz": 1}})


def test_records_nested_in_union_members_reject_unknown_fields():
    from yian.typesys.types import read_typedefs, type_for

    registry = read_typedefs({
        "Limit": {"px": "float"},
        "Buy": {"action": "Literal[buy]", "limit": "Limit"},
        "Sell": {"action": "Literal[sell]"},
    })
    model = type_to_schema(type_for("Union[Buy, Sell]", registry))
    with pytest.raises(ValidationError):
        model.model_validate({"value": {"action": "buy", "limit": {"px": 1.0, "zzz": 1}}})
