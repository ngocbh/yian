"""Strict type inference for parameters of generated inline composite flows."""

from __future__ import annotations

import pytest

from agent_composer.compose.build import inline_inputs
from agent_composer.compose.build.inline_inputs import infer_inline_binding_type
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange
from agent_composer.typesys.values import Type, ValueKind


def _infer(value, *, label: str = "each.input.value") -> Type:
    return infer_inline_binding_type(value, binding_label=label)


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (None, ValueKind.NONE),
        (False, ValueKind.BOOLEAN),
        (7, ValueKind.INTEGER),
        (2.5, ValueKind.NUMBER),
        ("plain text", ValueKind.STRING),
        ("$$ remains a top-level string", ValueKind.STRING),
    ],
)
def test_literal_scalar_types(value, kind):
    assert _infer(value) == Type.scalar(kind)


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("${null}", ValueKind.NONE),
        ("${none}", ValueKind.NONE),
        ("${true}", ValueKind.BOOLEAN),
        ("${false}", ValueKind.BOOLEAN),
        ("${42}", ValueKind.INTEGER),
        ("${2.5}", ValueKind.NUMBER),
        ('${"literal"}', ValueKind.STRING),
        ("${'literal'}", ValueKind.STRING),
    ],
)
def test_lone_scalar_expression_token_types(source, kind):
    assert _infer(source) == Type.scalar(kind)


@pytest.mark.parametrize(
    "source",
    [
        "prefix ${producer.output}",
        "${producer.output} suffix",
        "${left.output}${right.output}",
        "prefix ${left.output} middle ${right.output} suffix",
    ],
)
def test_embedded_or_multiple_template_spans_are_strings(source):
    assert _infer(source) == Type.scalar(ValueKind.STRING)


@pytest.mark.parametrize(
    "source",
    [
        "${producer.output}",
        "${[1, 2]}",
        "${1 + 2}",
        "${1 == 2}",
        "${true and false}",
        "${not true}",
        '${upper("x")}',
        "${left | right}",
        "${left :- 1}",
        '${left :? "required"}',
        "${-1}",
        "${${producer.output}}",
    ],
)
def test_lone_non_scalar_expression_is_rejected(source):
    with pytest.raises(LoadError) as caught:
        _infer(source, label="call.input.payload")

    message = str(caught.value)
    assert "call.input.payload" in message
    assert "explicitly typed named `defs:` body" in message


def test_mapping_literal_is_a_sorted_closed_record():
    inferred = _infer({"z": True, "a": 1, "nested": {"text": "literal"}})

    assert inferred.kind == ValueKind.OBJECT
    assert list(inferred.fields or {}) == ["a", "nested", "z"]
    assert inferred.required == frozenset({"a", "nested", "z"})
    assert inferred.fields == {
        "a": Type.scalar(ValueKind.INTEGER),
        "nested": Type(
            kind=ValueKind.OBJECT,
            fields={"text": Type.scalar(ValueKind.STRING)},
            required=frozenset({"text"}),
        ),
        "z": Type.scalar(ValueKind.BOOLEAN),
    }


def test_empty_mapping_literal_is_an_exact_empty_record():
    assert _infer({}) == Type(
        kind=ValueKind.OBJECT,
        fields={},
        required=frozenset(),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            ["a", "b"],
            Type(
                kind=ValueKind.LIST_STRING,
                element=Type.scalar(ValueKind.STRING),
            ),
        ),
        (
            [False, True],
            Type(
                kind=ValueKind.LIST_BOOLEAN,
                element=Type.scalar(ValueKind.BOOLEAN),
            ),
        ),
        (
            [1, 2],
            Type(
                kind=ValueKind.LIST_INTEGER,
                element=Type.scalar(ValueKind.INTEGER),
            ),
        ),
        (
            [1.0, 2.0],
            Type(
                kind=ValueKind.LIST_NUMBER,
                element=Type.scalar(ValueKind.NUMBER),
            ),
        ),
        (
            [1, 2.5],
            Type(
                kind=ValueKind.LIST_NUMBER,
                element=Type.scalar(ValueKind.NUMBER),
            ),
        ),
        (
            [None, None],
            Type(
                kind=ValueKind.LIST_ANY,
                element=Type.scalar(ValueKind.NONE),
            ),
        ),
        (
            [[1], [2]],
            Type(
                kind=ValueKind.LIST_ANY,
                element=Type(
                    kind=ValueKind.LIST_INTEGER,
                    element=Type.scalar(ValueKind.INTEGER),
                ),
            ),
        ),
        (
            [{"x": 1}, {"x": 2}],
            Type(
                kind=ValueKind.LIST_OBJECT,
                element=Type(
                    kind=ValueKind.OBJECT,
                    fields={"x": Type.scalar(ValueKind.INTEGER)},
                    required=frozenset({"x"}),
                ),
            ),
        ),
    ],
)
def test_nonempty_literal_lists_retain_their_structural_element(value, expected):
    assert _infer(value) == expected


def test_equivalent_record_elements_ignore_authored_key_order():
    inferred = _infer([{"a": 1, "b": "x"}, {"b": "y", "a": 2}])
    assert inferred.kind == ValueKind.LIST_OBJECT
    assert list(inferred.element.fields or {}) == ["a", "b"]


def test_nested_container_strings_never_scan_templates(monkeypatch):
    def fail_if_scanned(_source: str):
        pytest.fail("nested literal strings must not be scanned as templates")

    monkeypatch.setattr(inline_inputs, "scan_template", fail_if_scanned)
    inferred = _infer(
        {
            "deep": {"value": "${input.x}"},
            "ref": "${producer.output}",
            "items": ["${item.text}", "$$"],
        }
    )

    string_type = Type.scalar(ValueKind.STRING)
    assert inferred.fields["deep"].fields["value"] == string_type
    assert inferred.fields["ref"] == string_type
    assert inferred.fields["items"] == Type(
        kind=ValueKind.LIST_STRING,
        element=string_type,
    )


@pytest.mark.parametrize(
    "value",
    [
        [],
        [1, "one"],
        [None, 1],
        [True, 1],
        [[1], [1.0]],
        [{"x": 1}, {"y": 1}],
        [{"x": 1}, {"x": 1.0}],
        (1, 2),
        object(),
    ],
)
def test_unproven_or_unsupported_literal_is_rejected(value):
    with pytest.raises(LoadError, match="explicitly typed named `defs:` body"):
        _infer(value)


@pytest.mark.parametrize("key", [1, None])
def test_non_string_mapping_key_is_rejected(key):
    with pytest.raises(LoadError) as caught:
        _infer({key: "value"}, label="loop.input.record")

    assert "string keys" in str(caught.value)
    assert "loop.input.record" in str(caught.value)


@pytest.mark.parametrize("container_kind", ["mapping", "list"])
def test_active_literal_cycle_is_rejected(container_kind):
    if container_kind == "mapping":
        value = {}
        value["self"] = value
    else:
        value = []
        value.append(value)

    with pytest.raises(LoadError, match="cyclic literal container"):
        _infer(value)


def test_repeated_noncyclic_alias_is_allowed():
    shared = {"value": 3}
    inferred = _infer({"left": shared, "right": shared})
    assert inferred.fields["left"] == inferred.fields["right"]


def test_inference_error_preserves_binding_range_line_and_hint():
    source_range = SourceRange(17, 9, 17, 11)

    with pytest.raises(LoadError) as caught:
        infer_inline_binding_type(
            [],
            binding_label="each.input.items",
            binding_range=source_range,
        )

    error = caught.value
    assert error.range == source_range
    assert error.line == 17
    assert "each.input.items" in str(error)
    assert "explicitly typed named `defs:` body" in str(error)


def test_template_scan_error_preserves_binding_range_and_line():
    source_range = SourceRange(23, 7, 23, 21)

    with pytest.raises(LoadError) as caught:
        infer_inline_binding_type(
            "${unterminated",
            binding_label="call.input.query",
            binding_range=source_range,
        )

    error = caught.value
    assert error.range == source_range
    assert error.line == 23
    assert "call.input.query" in str(error)
    assert "unbalanced" in str(error)
    assert "explicitly typed named `defs:` body" in str(error)
