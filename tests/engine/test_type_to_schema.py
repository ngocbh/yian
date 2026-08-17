"""`type_to_schema` — derive a pydantic model from a declared output `Type`."""

import pytest
from pydantic import ValidationError

from agent_composer.nodes.agent.structured import type_to_schema
from agent_composer.typesys.values import Type, ValueKind


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
