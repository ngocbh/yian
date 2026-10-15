"""Strict type inference for parameters of generated inline composite flows."""

from __future__ import annotations

import pickle

import pytest

from yian.compose.build.inline_inputs import (
    canonicalize_inline_type,
    infer_inline_binding_type,
    require_proven_type,
)
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.diagnostics import SourceRange
from yian.nodes.contracts import PartialRecordContract, ValueContract, ValueTemplate
from yian.typesys import canonical_type_label
from yian.typesys.values import Type, ValueKind


def _diagnostics() -> LoadDiagnostics:
    return LoadDiagnostics(label="flow.yaml", text=(" " * 100 + "\n") * 100)


def _only_frame(error: LoadError):
    assert len(error.diagnostic.frames) == 1
    return error.diagnostic.frames[0]


def _unused_producer_type(
    _node_id: str,
    _binding_label: str,
    _binding_range: SourceRange | None,
) -> Type:
    """Fail if inference unexpectedly asks for a producer type."""
    pytest.fail("literal/template inference must not ask for a producer type")


def _unused_item_type(
    _binding_label: str,
    _binding_range: SourceRange | None,
) -> Type:
    """Fail if inference unexpectedly asks for a MAP item type."""
    pytest.fail("literal/template inference must not ask for a MAP item type")


def _infer(
    value,
    *,
    label: str = "each.input.value",
    parent_input_types: dict[str, Type] | None = None,
) -> Type:
    return infer_inline_binding_type(
        value,
        parent_input_types=parent_input_types or {},
        producer_type=_unused_producer_type,
        item_type=_unused_item_type,
        binding_label=label,
    )


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
    ],
)
def test_lone_non_scalar_expression_is_rejected(source):
    with pytest.raises(LoadError) as caught:
        _infer(source, label="call.input.payload")

    message = str(caught.value)
    assert "call.input.payload" in message
    assert (
        "declare its type with a tag (`payload: !<type> ...`) or use an "
        "explicitly typed named `defs:` body"
    ) in message


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


def test_nested_template_strings_scan_as_templates():
    inferred = _infer(
        ValueTemplate(
            {
                "deep": {"value": "${input.x}"},
                "ref": "${input.x}",
                "items": ["n=${input.x}", "$$"],
            }
        ),
        parent_input_types={"x": Type.scalar(ValueKind.INTEGER)},
    )

    int_type = Type.scalar(ValueKind.INTEGER)
    string_type = Type.scalar(ValueKind.STRING)
    assert inferred.fields["deep"].fields["value"] == int_type
    assert inferred.fields["ref"] == int_type
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


def test_direct_parent_input_reference_uses_declared_type_without_callbacks():
    declared = Type(
        kind=ValueKind.OBJECT,
        fields={"value": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"value"}),
    )

    inferred = _infer(
        "${input.record.value}",
        parent_input_types={"record": declared},
    )

    assert inferred == Type.scalar(ValueKind.INTEGER)


def test_whole_unrefined_object_reference_is_allowed_but_copied():
    declared = Type.scalar(ValueKind.OBJECT)

    inferred = _infer(
        "${input.payload}",
        parent_input_types={"payload": declared},
    )

    assert inferred == declared
    assert inferred is not declared


def test_direct_producer_reference_passes_request_context_and_returns_a_copy():
    source_range = SourceRange(31, 8, 31, 32)
    output_type = Type(
        kind=ValueKind.OBJECT,
        fields={"result": Type.scalar(ValueKind.STRING)},
        required=frozenset({"result"}),
    )
    calls: list[tuple[str, str, SourceRange | None]] = []

    def producer_type(
        node_id: str,
        binding_label: str,
        binding_range: SourceRange | None,
    ) -> Type:
        calls.append((node_id, binding_label, binding_range))
        return output_type

    inferred = infer_inline_binding_type(
        "${prepare.output}",
        parent_input_types={},
        producer_type=producer_type,
        binding_label="call.input.payload",
        binding_range=source_range,
    )

    assert calls == [("prepare", "call.input.payload", source_range)]
    assert inferred == output_type
    assert inferred is not output_type


def test_direct_item_reference_passes_request_context_and_walks_fields():
    source_range = SourceRange(42, 6, 42, 18)
    item = Type(
        kind=ValueKind.OBJECT,
        fields={"name": Type.scalar(ValueKind.STRING)},
        required=frozenset({"name"}),
    )
    calls: list[tuple[str, SourceRange | None]] = []

    def item_type(
        binding_label: str,
        binding_range: SourceRange | None,
    ) -> Type:
        calls.append((binding_label, binding_range))
        return item

    inferred = infer_inline_binding_type(
        "${item.name}",
        parent_input_types={},
        producer_type=_unused_producer_type,
        item_type=item_type,
        binding_label="each.input.name",
        binding_range=source_range,
    )

    assert calls == [("each.input.name", source_range)]
    assert inferred == Type.scalar(ValueKind.STRING)


def test_whole_item_reference_returns_a_fresh_nullable_type_without_mutation():
    source_range = SourceRange(47, 6, 47, 13)
    item = Type(
        kind=ValueKind.OBJECT,
        fields={"name": Type.scalar(ValueKind.STRING)},
        required=frozenset({"name"}),
        nullable=True,
    )
    before = pickle.dumps(item)
    calls: list[tuple[str, SourceRange | None]] = []

    def item_type(
        binding_label: str,
        binding_range: SourceRange | None,
    ) -> Type:
        calls.append((binding_label, binding_range))
        return item

    inferred = infer_inline_binding_type(
        "${item}",
        parent_input_types={},
        producer_type=_unused_producer_type,
        item_type=item_type,
        binding_label="each.input.row",
        binding_range=source_range,
    )

    assert calls == [("each.input.row", source_range)]
    assert inferred == item
    assert inferred is not item
    assert inferred.nullable is True
    assert pickle.loads(before) == item


def test_computed_item_expression_rejects_without_invoking_item_provider():
    calls = 0

    def item_type(
        _binding_label: str,
        _binding_range: SourceRange | None,
    ) -> Type:
        nonlocal calls
        calls += 1
        return Type.scalar(ValueKind.INTEGER)

    with pytest.raises(LoadError, match="explicitly typed named `defs:` body"):
        infer_inline_binding_type(
            "${item.value + 1}",
            parent_input_types={},
            producer_type=_unused_producer_type,
            item_type=item_type,
            binding_label="each.input.computed",
        )

    assert calls == 0


@pytest.mark.parametrize(
    "value",
    [
        3,
        "prefix ${item.name}",
        "${input.record.name}",
        {"nested": "${item.name}"},
        ["${item.name}"],
    ],
)
def test_item_provider_is_demand_driven_only_for_a_direct_item_reference(value):
    calls = 0

    def item_type(
        _binding_label: str,
        _binding_range: SourceRange | None,
    ) -> Type:
        nonlocal calls
        calls += 1
        return Type.scalar(ValueKind.STRING)

    infer_inline_binding_type(
        value,
        parent_input_types={
            "record": Type(
                kind=ValueKind.OBJECT,
                fields={"name": Type.scalar(ValueKind.STRING)},
                required=frozenset({"name"}),
            )
        },
        producer_type=_unused_producer_type,
        item_type=item_type,
        binding_label="each.input.value",
    )

    assert calls == 0


@pytest.mark.parametrize(
    ("root_nullable", "middle_nullable", "value_nullable", "expected_nullable"),
    [
        (False, False, False, False),
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
    ],
)
def test_direct_record_walk_accumulates_every_nullable_step_without_mutation(
    root_nullable,
    middle_nullable,
    value_nullable,
    expected_nullable,
):
    value_type = Type.scalar(ValueKind.INTEGER)
    if value_nullable:
        value_type = Type(kind=ValueKind.INTEGER, nullable=True)
    middle_type = Type(
        kind=ValueKind.OBJECT,
        fields={"value": value_type},
        required=frozenset() if value_nullable else frozenset({"value"}),
        nullable=middle_nullable,
    )
    root = Type(
        kind=ValueKind.OBJECT,
        fields={"middle": middle_type},
        required=frozenset({"middle"}),
        nullable=root_nullable,
    )
    before = pickle.dumps(root)

    inferred = _infer(
        "${input.record.middle.value}",
        parent_input_types={"record": root},
    )

    assert inferred.kind == ValueKind.INTEGER
    assert inferred.nullable is expected_nullable
    assert inferred is not value_type
    assert pickle.loads(before) == root


def test_nullable_producer_root_propagates_to_a_selected_field():
    output_type = Type(
        kind=ValueKind.OBJECT,
        fields={"value": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"value"}),
        nullable=True,
    )

    inferred = infer_inline_binding_type(
        "${source.output.value}",
        parent_input_types={},
        producer_type=lambda _node, _label, _range: output_type,
        binding_label="call.input.value",
    )

    assert inferred == Type(kind=ValueKind.INTEGER, nullable=True)


def test_known_partial_producer_field_is_available_to_inline_inference():
    """A partial sibling does not hide an independently proven selected field."""
    partial = PartialRecordContract(
        fields=(
            ("value", ValueContract(Type.scalar(ValueKind.INTEGER))),
            ("note", None),
        )
    )

    inferred = infer_inline_binding_type(
        "${source.output.value}",
        parent_input_types={},
        producer_type=lambda _node, _label, _range: partial,
        binding_label="call.input.value",
    )

    assert inferred == Type.scalar(ValueKind.INTEGER)


def test_inline_inference_rejects_a_field_on_a_scalar_producer():
    with pytest.raises(LoadError, match="reads field 'x' on type int, which has no fields"):
        infer_inline_binding_type(
            "${source.output.x}",
            parent_input_types={},
            producer_type=lambda _node, _label, _range: ValueContract(
                Type.scalar(ValueKind.INTEGER)
            ),
            binding_label="call.input.value",
        )


@pytest.mark.parametrize("reference", ["${source.output}", "${source.output.note}"])
def test_opaque_partial_producer_reads_remain_unavailable_to_inline_inference(
    reference: str,
):
    """Whole and opaque-field reads cannot invent an exact generated input type."""
    partial = PartialRecordContract(
        fields=(
            ("value", ValueContract(Type.scalar(ValueKind.INTEGER))),
            ("note", None),
        )
    )

    with pytest.raises(LoadError, match="no statically proven exact type"):
        infer_inline_binding_type(
            reference,
            parent_input_types={},
            producer_type=lambda _node, _label, _range: partial,
            binding_label="call.input.value",
        )


def test_nullable_item_root_propagates_to_a_selected_field():
    item = Type(
        kind=ValueKind.OBJECT,
        fields={"value": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"value"}),
        nullable=True,
    )

    inferred = infer_inline_binding_type(
        "${item.value}",
        parent_input_types={},
        producer_type=_unused_producer_type,
        item_type=lambda _label, _range: item,
        binding_label="each.input.value",
    )

    assert inferred == Type(kind=ValueKind.INTEGER, nullable=True)


def test_zero_step_direct_walk_always_returns_a_fresh_type():
    source_type = Type(kind=ValueKind.INTEGER, nullable=True)

    inferred = _infer(
        "${input.value}",
        parent_input_types={"value": source_type},
    )

    assert inferred == source_type
    assert inferred is not source_type


@pytest.mark.parametrize(
    "source",
    [
        "${input}",
        "${input.missing}",
        "${input.opaque.value}",
        "${input.record.missing}",
        "${item.value}",
        "${system.now}",
        "${producer}",
        "${producer.value}",
        "${producer()}",
        "${producer.output + 1}",
        '${upper("x")}',
    ],
)
def test_unproven_direct_reference_is_a_located_named_def_error(source):
    source_range = SourceRange(51, 4, 51, 28)
    diagnostics = _diagnostics()
    inputs = {
        "opaque": Type.scalar(ValueKind.OBJECT),
        "record": Type(
            kind=ValueKind.OBJECT,
            fields={"known": Type.scalar(ValueKind.STRING)},
            required=frozenset({"known"}),
        ),
    }

    with pytest.raises(LoadError) as caught:
        infer_inline_binding_type(
            source,
            parent_input_types=inputs,
            producer_type=_unused_producer_type,
            binding_label="call.input.payload",
            binding_range=source_range,
            diagnostics=diagnostics,
        )

    error = caught.value
    assert _only_frame(error).range == source_range
    assert "call.input.payload" in str(error)
    assert (
        "declare its type with a tag (`payload: !<type> ...`) or use an "
        "explicitly typed named `defs:` body"
    ) in str(error)


@pytest.mark.parametrize("provider", ["producer", "item"])
def test_provider_load_error_propagates_unchanged(provider):
    source_range = SourceRange(64, 3, 64, 22)
    diagnostics = _diagnostics()
    expected = diagnostics.error("provider could not prove a type", range=source_range)
    calls = []

    def producer_type(node_id, binding_label, binding_range):
        calls.append((node_id, binding_label, binding_range))
        raise expected

    def item_type(binding_label, binding_range):
        calls.append((binding_label, binding_range))
        raise expected

    source = "${worker.output}" if provider == "producer" else "${item}"
    with pytest.raises(LoadError) as caught:
        infer_inline_binding_type(
            source,
            parent_input_types={},
            producer_type=producer_type,
            item_type=item_type,
            binding_label="each.input.payload",
            binding_range=source_range,
            diagnostics=diagnostics,
        )

    assert caught.value is expected
    if provider == "producer":
        assert calls == [("worker", "each.input.payload", source_range)]
    else:
        assert calls == [("each.input.payload", source_range)]


def test_inference_error_preserves_binding_range_line_and_hint():
    source_range = SourceRange(17, 9, 17, 11)

    with pytest.raises(LoadError) as caught:
        infer_inline_binding_type(
            [],
            parent_input_types={},
            producer_type=_unused_producer_type,
            binding_label="each.input.items",
            binding_range=source_range,
            diagnostics=_diagnostics(),
        )

    error = caught.value
    assert _only_frame(error).range == source_range
    assert "each.input.items" in str(error)
    assert (
        "declare its type with a tag (`items: !<type> ...`) or use an "
        "explicitly typed named `defs:` body"
    ) in str(error)


def test_template_scan_error_preserves_binding_range_and_line():
    source_range = SourceRange(23, 7, 23, 21)

    with pytest.raises(LoadError) as caught:
        infer_inline_binding_type(
            "${unterminated",
            parent_input_types={},
            producer_type=_unused_producer_type,
            binding_label="call.input.query",
            binding_range=source_range,
            diagnostics=_diagnostics(),
        )

    error = caught.value
    assert _only_frame(error).range == source_range
    assert "call.input.query" in str(error)
    assert "unbalanced" in str(error)
    assert (
        "declare its type with a tag (`query: !<type> ...`) or use an "
        "explicitly typed named `defs:` body"
    ) in str(error)


_LIST_VALUE_KINDS = tuple(kind for kind in ValueKind if kind.is_list())


def _list_label(kind: ValueKind) -> str:
    """The label of a list kind with no element type, e.g. `list(str)`."""
    element = kind.element_kind
    return f"list({'Optional(Any)' if element is None else element.value})"


@pytest.mark.parametrize("kind", _LIST_VALUE_KINDS)
def test_every_list_kind_requires_a_recursively_proven_element(kind):
    element = Type.scalar(ValueKind.STRING)
    proven = Type(kind=kind, element=element)

    assert require_proven_type(proven, binding_label="call.input.items") is proven

    with pytest.raises(LoadError) as caught:
        require_proven_type(Type(kind=kind), binding_label="call.input.items")

    assert _list_label(kind) in str(caught.value)
    assert "no proven element type" in str(caught.value)


def test_list_proof_recurses_into_a_present_malformed_element():
    malformed_element = Type(
        kind=ValueKind.OBJECT,
        fields={"optional_but_nonnullable": Type.scalar(ValueKind.STRING)},
        required=frozenset(),
    )
    typ = Type(kind=ValueKind.LIST_ANY, element=malformed_element)

    with pytest.raises(LoadError, match="optional_but_nonnullable"):
        require_proven_type(typ, binding_label="call.input.items")


@pytest.mark.parametrize(
    ("typ", "label"),
    [
        (
            Type(kind=ValueKind.LIST_ANY, element=Type.scalar(ValueKind.NONE)),
            "list(None)",
        ),
        (
            Type(
                kind=ValueKind.LIST_ANY,
                element=Type(
                    kind=ValueKind.LIST_ANY,
                    element=Type.scalar(ValueKind.DATE),
                ),
            ),
            "list(list(date))",
        ),
        (
            Type(kind=ValueKind.LIST_ANY, element=Type.scalar(ValueKind.DATETIME)),
            "list(datetime)",
        ),
        (
            Type(kind=ValueKind.LIST_ANY, element=Type.scalar(ValueKind.FILE)),
            "list(file)",
        ),
    ],
)
def test_list_any_retains_proven_special_and_nested_elements(typ, label):
    assert require_proven_type(typ, binding_label="call.input.items") is typ
    assert canonical_type_label(typ) == label


def test_unrefined_object_is_proven_whole_but_not_dottable():
    opaque = Type.scalar(ValueKind.OBJECT)

    assert require_proven_type(opaque, binding_label="call.input.payload") is opaque
    assert (
        _infer(
            "${input.payload}",
            parent_input_types={"payload": opaque},
        )
        == opaque
    )

    with pytest.raises(LoadError, match="without proven record fields"):
        _infer(
            "${input.payload.value}",
            parent_input_types={"payload": opaque},
        )


def test_record_rejects_required_names_absent_from_fields():
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={"present": Type.scalar(ValueKind.STRING)},
        required=frozenset({"present", "absent"}),
    )

    with pytest.raises(LoadError) as caught:
        require_proven_type(typ, binding_label="call.input.record")

    assert "requires unknown fields ['absent']" in str(caught.value)


def test_record_rejects_any_nonrequired_nonnullable_field():
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={
            "required": Type.scalar(ValueKind.INTEGER),
            "ordinary_string": Type.scalar(ValueKind.STRING),
        },
        required=frozenset({"required"}),
    )

    with pytest.raises(LoadError) as caught:
        require_proven_type(typ, binding_label="loop.input.record")

    assert "ordinary_string" in str(caught.value)
    assert "neither required nor nullable" in str(caught.value)


def test_record_accepts_required_nullable_optional_nullable_and_required_none():
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={
            "required_nullable": Type(kind=ValueKind.STRING, nullable=True),
            "optional_nullable": Type(kind=ValueKind.INTEGER, nullable=True),
            "required_none": Type.scalar(ValueKind.NONE),
        },
        required=frozenset({"required_nullable", "required_none"}),
    )

    assert require_proven_type(typ, binding_label="call.input.record") is typ


def test_recursive_proof_rejects_a_malformed_nested_record():
    malformed_child = Type(
        kind=ValueKind.OBJECT,
        fields={"unproven": Type.scalar(ValueKind.STRING)},
        required=frozenset(),
    )
    root = Type(
        kind=ValueKind.OBJECT,
        fields={"child": malformed_child},
        required=frozenset({"child"}),
    )

    with pytest.raises(LoadError, match="unproven"):
        require_proven_type(root, binding_label="call.input.record")


def test_direct_reference_proves_the_entire_root_before_selecting_a_field():
    root = Type(
        kind=ValueKind.OBJECT,
        fields={
            "selected": Type.scalar(ValueKind.INTEGER),
            "malformed_sibling": Type(kind=ValueKind.LIST_ANY),
        },
        required=frozenset({"selected", "malformed_sibling"}),
    )

    with pytest.raises(LoadError) as caught:
        _infer(
            "${input.record.selected}",
            parent_input_types={"record": root},
            label="call.input.value",
        )

    assert "list(Optional(Any))" in str(caught.value)
    assert "no proven element type" in str(caught.value)


def test_structural_proof_error_preserves_source_range_line_and_hint():
    source_range = SourceRange(73, 5, 73, 27)

    with pytest.raises(LoadError) as caught:
        require_proven_type(
            Type(kind=ValueKind.LIST_OBJECT),
            binding_label="each.input.rows",
            binding_range=source_range,
            diagnostics=_diagnostics(),
        )

    error = caught.value
    assert _only_frame(error).range == source_range
    assert "each.input.rows" in str(error)
    assert (
        "declare its type with a tag (`rows: !<type> ...`) or use an "
        "explicitly typed named `defs:` body"
    ) in str(error)


def test_canonicalization_deep_copies_sorts_and_preserves_required_membership():
    required_nullable = Type(kind=ValueKind.STRING, nullable=True)
    variant = Type(
        kind=ValueKind.STRING,
        tags=frozenset({"é", "z", 'a,"b'}),
    )
    listed_variant = Type(kind=ValueKind.LIST_STRING, element=variant)
    original = Type(
        kind=ValueKind.OBJECT,
        fields={
            "é": Type.scalar(ValueKind.BOOLEAN),
            "optional": Type(kind=ValueKind.INTEGER, nullable=True),
            "required_nullable": required_nullable,
            "a": listed_variant,
        },
        required=frozenset({"é", "required_nullable", "a"}),
    )
    before = pickle.dumps(original)

    canonical = canonicalize_inline_type(original)

    assert canonical == original
    assert canonical is not original
    assert list(canonical.fields or {}) == ["a", "optional", "required_nullable", "é"]
    assert canonical.required == frozenset({"a", "required_nullable", "é"})
    assert canonical.fields["required_nullable"].nullable is True
    assert canonical.fields["a"] is not listed_variant
    assert canonical.fields["a"].element is not variant
    assert canonical.fields["a"].element.tags == variant.tags
    assert pickle.loads(before) == original
    assert list(original.fields or {}) == ["é", "optional", "required_nullable", "a"]


def test_canonicalization_normalizes_refined_record_required_none_to_empty():
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={"optional": Type(kind=ValueKind.STRING, nullable=True)},
        required=None,
    )

    canonical = canonicalize_inline_type(typ)

    assert canonical.required == frozenset()
    assert typ.required is None


def test_inline_type_copy_preserves_bounded_nested_validation():
    from yian.typesys.types import type_for
    from yian.typesys.values import TypeCheckError, build_value_as

    declared = Type(ValueKind.OBJECT, fields={
        "scores": type_for("list(Optional(float(-0.5..1)))", {}),
        "count": type_for("int(1..12)", {}),
    }, required=frozenset({"scores", "count"}))
    inferred = _infer("${input.payload}", parent_input_types={"payload": declared})
    copied = canonicalize_inline_type(inferred)
    assert copied == declared and copied is not declared
    assert copied.fields["scores"].element is not declared.fields["scores"].element
    assert build_value_as(copied, {"scores": [None, 1], "count": 4}).to_object() == {
        "scores": [None, 1], "count": 4,
    }
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(copied, {"scores": [1.1], "count": 4})
    assert caught.value.path == ("scores", 0)


@pytest.mark.parametrize(
    "kind",
    [
        ValueKind.NONE,
        ValueKind.STRING,
        ValueKind.INTEGER,
        ValueKind.NUMBER,
        ValueKind.DATE,
        ValueKind.DATETIME,
        ValueKind.BOOLEAN,
        ValueKind.OBJECT,
        ValueKind.FILE,
    ],
)
def test_canonical_scalar_labels_use_author_vocabulary(kind):
    assert canonical_type_label(Type.scalar(kind)) == kind.value


@pytest.mark.parametrize("kind", _LIST_VALUE_KINDS)
def test_every_list_kind_label_uses_its_structural_element(kind):
    typ = Type(kind=kind, element=Type(kind=ValueKind.INTEGER, nullable=True))

    assert canonical_type_label(typ) == "list(Optional(int))"


@pytest.mark.parametrize("kind", _LIST_VALUE_KINDS)
def test_canonical_flat_list_label_uses_its_storage_kind(kind):
    assert canonical_type_label(Type(kind=kind)) == _list_label(kind)


def test_canonical_variant_label_sorts_and_json_quotes_adversarial_tags():
    typ = Type(
        kind=ValueKind.STRING,
        tags=frozenset({"é", "z", 'a,"b'}),
    )

    assert canonical_type_label(typ) == 'Literal("a,\\"b", z, é)'


def test_canonical_record_label_sorts_and_json_quotes_adversarial_keys():
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={
            "é": Type.scalar(ValueKind.BOOLEAN),
            'quote"key': Type.scalar(ValueKind.STRING),
            "a: int, b": Type.scalar(ValueKind.INTEGER),
        },
        required=frozenset({"é", 'quote"key', "a: int, b"}),
    )

    assert canonical_type_label(typ) == ('{"a: int, b":int,"quote\\"key":str,"é":bool}')


def test_nullable_wrapper_is_outermost_for_a_structural_record_label():
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={"value": Type.scalar(ValueKind.STRING)},
        required=frozenset({"value"}),
        nullable=True,
    )

    assert canonical_type_label(typ) == 'Optional({"value":str})'
