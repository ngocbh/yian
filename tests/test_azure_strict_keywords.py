"""Azure strict structured output without the JSON-schema keywords its strict mode rejects."""

from pydantic import BaseModel, Field

from yian.llm_clients.azure_client import _strict_schema_without
from yian.memory.reflective import _ReflectionResponse
from yian.nodes.agent.structured import type_to_schema
from yian.typesys.types import read_typedefs, type_for

REGISTRY = read_typedefs({
    "C": {"a": "int", "b": "list(str)"},
    "M": {"minimum": "int(0..3)"},
})


def _schema(text):
    return type_to_schema(type_for(text, REGISTRY))


def _value(fmt):
    return fmt["json_schema"]["schema"]["properties"]["value"]


def test_a_bounded_int_moves_its_bounds_into_the_description():
    assert _strict_schema_without(_schema("int(0..10)")) == {
        "type": "json_schema",
        "json_schema": {
            "name": "ScalarWrapper",
            "strict": True,
            "schema": {
                "properties": {"value": {
                    "title": "Value", "type": "integer",
                    "description": "{minimum: 0, maximum: 10}",
                }},
                "required": ["value"],
                "title": "ScalarWrapper",
                "type": "object",
                "additionalProperties": False,
            },
        },
    }


def test_a_bound_on_an_any_of_member_moves_on_that_member():
    assert _value(_strict_schema_without(_schema("Optional(int(0..3))"))) == {
        "anyOf": [
            {"type": "integer", "description": "{minimum: 0, maximum: 3}"},
            {"type": "null"},
        ],
        "title": "Value",
    }


def test_float_bounds_keep_their_spelling():
    value = _value(_strict_schema_without(_schema("float(0.0..1.0)")))
    assert value["description"] == "{minimum: 0.0, maximum: 1.0}"
    assert "minimum" not in value and "maximum" not in value


def test_an_open_end_gives_only_the_given_bound():
    value = _value(_strict_schema_without(_schema("int(0..)")))
    assert value["description"] == "{minimum: 0}"
    assert "minimum" not in value


class _Described(BaseModel):
    count: int = Field(ge=1, description="How many.")


def test_an_existing_description_is_kept_before_the_note():
    count = _strict_schema_without(_Described)["json_schema"]["schema"]["properties"]["count"]
    assert count["description"] == "How many.\n\n{minimum: 1}"
    assert "minimum" not in count


def test_a_field_named_like_a_keyword_is_kept():
    properties = _strict_schema_without(_schema("M"))["json_schema"]["schema"]["properties"]
    assert properties == {"minimum": {
        "title": "Minimum", "type": "integer", "description": "{minimum: 0, maximum: 3}",
    }}


def test_reflection_response_min_items_moves_into_the_description():
    schema = _strict_schema_without(_ReflectionResponse)["json_schema"]["schema"]
    actions = schema["properties"]["actions"]
    assert "minItems" not in actions
    assert actions["description"].endswith("{minItems: 1}")


def test_closed_unbounded_schemas_need_no_change():
    assert _strict_schema_without(_schema("C")) is None
    assert _strict_schema_without(_schema("int")) is None
