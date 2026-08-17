"""_answer_schema derives a lightweight, lossless answer schema from an output Type."""

from agent_composer.compose.nodes.human_input import _answer_schema
from agent_composer.compose.helpers.types import read_type
from agent_composer.typesys.types import read_typedefs

_REG = read_typedefs({})


def _schema(type_expr, typedefs=None):
    reg = read_typedefs(typedefs) if typedefs else _REG
    return _answer_schema(read_type(type_expr, reg))


def test_none_type_is_empty():
    assert _answer_schema(None) == []


def test_scalar():
    assert _schema("str") == [{"name": "answer", "type": "str", "required": True}]


def test_nullable_scalar_is_optional():
    assert _schema("Optional[str]") == [{"name": "answer", "type": "str", "required": False}]


def test_literal_enum_carries_values():
    assert _schema("Literal[approve, reject]") == [
        {"name": "answer", "type": "str", "required": True, "enum": ["approve", "reject"]}
    ]


def test_list_of_enum_keeps_the_element_enum():
    # The regression the review flagged: a list output dropped its element enum.
    schema = _schema("list[Literal[approve, reject]]")
    assert schema[0]["name"] == "answer"
    assert schema[0]["enum"] == ["approve", "reject"]


def test_record_expands_per_field_with_required():
    schema = _schema({"rating": "float", "note": "Optional[str]"})
    by_name = {e["name"]: e for e in schema}
    assert by_name["rating"] == {"name": "rating", "type": "float", "required": True}
    assert by_name["note"] == {"name": "note", "type": "str", "required": False}


def test_nested_record_recurses():
    schema = _schema({"inner": {"x": "int"}})
    inner = next(e for e in schema if e["name"] == "inner")
    assert inner["fields"] == [{"name": "x", "type": "int", "required": True}]
