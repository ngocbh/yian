import datetime
import re
from typing import Any

import pytest
from yian.compile.validation import type_relation, types_compatible
from yian.compose import load_flow
from yian.compose.build.inline_inputs import require_proven_type
from yian.compose.helpers.errors import LoadError
from yian.nodes.contracts import ValueContract, output_contract_field_error
from yian.typesys.types import ScalarExpr, parse_type, read_typedefs, type_for
from yian.typesys.values import (
    ANY_VALUE_ADAPTER,
    BooleanValue,
    FileRef,
    IntegerValue,
    ListAnyValue,
    NoneValue,
    NumberValue,
    ObjectValue,
    StringValue,
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
    canonical_type_label,
    canonical_union_order,
    has_file,
    requires_file,
    types_overlap,
)

ANY = Type.scalar(ValueKind.ANY)


def _nest(n: int, leaf: Any = 1) -> Any:
    """Return `leaf` wrapped in `n` lists."""
    value = leaf
    for _ in range(n):
        value = [value]
    return value


def _types(value: Any) -> Any:
    """Return `value` with each scalar replaced by its type, keeping lists and dicts."""
    if isinstance(value, dict):
        return {key: _types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_types(item) for item in value]
    return type(value)


@pytest.mark.parametrize(
    ("value", "carrier", "stored"),
    [
        ("hello", StringValue, "hello"),
        (3, IntegerValue, 3),
        (2**80, IntegerValue, 2**80),
        (2.5, NumberValue, 2.5),
        (True, BooleanValue, True),
        ({"a": [1, None]}, ObjectValue, {"a": [1, None]}),
        ([1, "two", None], ListAnyValue, [1, "two", None]),
        ([1, 2.5], ListAnyValue, [1, 2.5]),
        ((1, 2), ListAnyValue, [1, 2]),
        ([True, 1], ListAnyValue, [True, 1]),
    ],
)
def test_any_stores_a_json_value_in_its_natural_carrier(value, carrier, stored):
    typed = build_value_as(ANY, value)

    assert type(typed) is carrier
    assert typed.value == stored


def test_any_keeps_list_items_as_given():
    ints = build_value_as(ANY, [1, 2.5]).value
    bools = build_value_as(ANY, [True, 1]).value

    assert type(ints[0]) is int
    assert type(bools[0]) is bool
    assert type(bools[1]) is int


def test_any_accepts_a_bare_value_kind():
    assert build_value_as(ValueKind.ANY, "hello") == StringValue(value="hello")


@pytest.mark.parametrize(
    ("value", "message", "path"),
    [
        (None, "None does not match declared type Any", ()),
        ({1, 2}, "{1, 2} is not a JSON value", ()),
        (b"x", "b'x' is not a JSON value", ()),
        (datetime.date(2026, 1, 1), "datetime.date(2026, 1, 1) is not a JSON value", ()),
        (FileRef(uri="f"), "is not a JSON value", ()),
        (object(), "is not a JSON value", ()),
        ({1: "a"}, "object keys must be text, got 1", ()),
        ({"a": {1, 2}}, "{1, 2} is not a JSON value", ("a",)),
        ({"a": [1, {2}]}, "{2} is not a JSON value", ("a", 1)),
    ],
)
def test_any_rejects_a_non_json_value_at_its_path(value, message, path):
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(ANY, value)

    assert message in str(caught.value)
    assert caught.value.path == path


def test_any_leaves_the_nesting_limit_to_the_boundaries():
    assert build_value_as(ANY, _nest(150)).value == _nest(150)


def test_any_ends_a_cyclic_value():
    cyclic: dict = {}
    cyclic["self"] = {"items": [1, cyclic]}

    with pytest.raises(TypeCheckError) as caught:
        build_value_as(ANY, cyclic)

    assert str(caught.value) == "value contains itself"
    assert caught.value.path == ("self", "items", 1)


def test_any_accepts_a_shared_value_that_is_not_a_cycle():
    shared = [1, 2]

    assert build_value_as(ANY, {"a": shared, "b": shared}).value == {"a": [1, 2], "b": [1, 2]}


def test_any_path_unwraps_a_str_subclass_key():
    class Key(str):
        pass

    with pytest.raises(TypeCheckError, match="is not a JSON value") as caught:
        build_value_as(ANY, {Key("k"): {1}})

    assert caught.value.path == ("k",)
    assert type(caught.value.path[0]) is str


def test_any_reports_the_first_bad_part_in_order():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(ANY, {"a": [1, {2}], 1: "b"})

    assert str(caught.value) == "{2} is not a JSON value"
    assert caught.value.path == ("a", 1)


def test_optional_any_accepts_none():
    assert build_value_as(Type(kind=ValueKind.ANY, nullable=True), None) == NoneValue()


def test_object_rejects_a_non_text_key():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(ValueKind.OBJECT, {1: "a"})

    assert str(caught.value) == "object keys must be text, got 1"
    assert caught.value.path == ()


@pytest.mark.parametrize(
    ("value", "path"),
    [
        ({"a": {1, 2}}, ("a",)),
        ({"a": object()}, ("a",)),
        ({"d": datetime.date(2026, 1, 1)}, ("d",)),
    ],
)
def test_object_rejects_a_non_json_member(value, path):
    with pytest.raises(TypeCheckError, match="is not a JSON value") as caught:
        build_value_as(ValueKind.OBJECT, value)

    assert caught.value.path == path


def test_object_accepts_json_members():
    value = {"a": None, "b": [1, {"c": "d"}], "t": (1, 2)}

    assert build_value_as(ValueKind.OBJECT, value) == ObjectValue(value=value)


def test_flat_list_of_objects_rejects_a_non_text_key_at_the_item():
    with pytest.raises(TypeCheckError, match="object keys must be text, got 1") as caught:
        build_value_as(ValueKind.LIST_OBJECT, [{"a": 1}, {1: "b"}])

    assert caught.value.path == (1,)


def test_elementless_list_any_checks_items_are_json_values():
    assert build_value_as(ValueKind.LIST_ANY, [1, None, "x"]).value == [1, None, "x"]

    with pytest.raises(TypeCheckError, match="is not a JSON value") as caught:
        build_value_as(ValueKind.LIST_ANY, [{1, 2}])

    assert caught.value.path == (0,)


def test_record_with_required_any_field_rejects_null():
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"v": ANY},
        required=frozenset({"v"}),
    )

    with pytest.raises(TypeCheckError) as caught:
        build_value_as(record, {"v": None})

    assert caught.value.path == ("v",)


def test_record_with_optional_any_field_accepts_absent_and_null():
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"v": Type(kind=ValueKind.ANY, nullable=True)},
        required=frozenset(),
    )

    assert build_value_as(record, {}).value == {}
    assert build_value_as(record, {"v": None}).value == {"v": None}


@pytest.mark.parametrize(
    "value",
    [
        "hello",
        3,
        2**80,
        2.5,
        True,
        {"a": [1, None, 2.5, True]},
        [1, "two", None],
        [1, 2.5],
        (1, 2),
        [True, {"b": [1, 2.0]}],
    ],
)
def test_any_value_round_trips_through_the_wire(value):
    typed = build_value_as(ANY, value)

    decoded = ANY_VALUE_ADAPTER.validate_python(
        ANY_VALUE_ADAPTER.dump_python(typed, mode="json")
    )

    assert type(decoded) is type(typed)
    assert decoded.value == typed.value
    assert _types(decoded.value) == _types(typed.value)


def test_any_type_shape_is_valid():
    Type(kind=ValueKind.ANY).validate_shape()


@pytest.mark.parametrize(
    "refinement",
    [
        {"fields": {}},
        {"tags": frozenset({"a"})},
        {"bounds": (0, 1)},
        {"element": Type.scalar(ValueKind.STRING)},
    ],
)
def test_any_type_rejects_refinements(refinement):
    with pytest.raises(ValueError):
        Type(kind=ValueKind.ANY, **refinement).validate_shape()


# --------------------------------------------------------------------------- #
# Authored `Any`: parsing, resolution, labels
# --------------------------------------------------------------------------- #

OPTIONAL_ANY = Type(kind=ValueKind.ANY, nullable=True)
LIST_ANY = Type(kind=ValueKind.LIST_ANY, element=ANY)
LIST_OPTIONAL_ANY = Type(kind=ValueKind.LIST_ANY, element=OPTIONAL_ANY)
UNPROVEN_LIST = Type(kind=ValueKind.LIST_ANY)


def test_any_parses_as_its_own_scalar():
    assert parse_type("Any") == ScalarExpr("Any")


@pytest.mark.parametrize(
    ("text", "typ", "label"),
    [
        ("Any", ANY, "Any"),
        ("Optional(Any)", OPTIONAL_ANY, "Optional(Any)"),
        ("list(Any)", LIST_ANY, "list(Any)"),
        ("list(Optional(Any))", LIST_OPTIONAL_ANY, "list(Optional(Any))"),
    ],
)
def test_authored_any_types_resolve_and_label_back(text, typ, label):
    assert type_for(text, {}) == typ
    assert canonical_type_label(typ) == label
    assert type_for(canonical_type_label(typ), {}) == typ


def test_a_list_with_unproven_items_is_labelled_list_of_optional_any():
    assert canonical_type_label(UNPROVEN_LIST) == "list(Optional(Any))"


def test_list_of_any_rejects_a_null_item_and_list_of_optional_any_accepts_it():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(type_for("list(Any)", {}), [1, None])
    assert caught.value.path == (1,)

    typed = build_value_as(type_for("list(Optional(Any))", {}), [1, None])
    assert typed == ListAnyValue(value=[1, None])


def test_issue_examples_pass():
    assert build_value_as(type_for("Any", {}), "hello") == StringValue(value="hello")
    assert build_value_as(type_for("list(Any)", {}), [1, "two"]) == ListAnyValue(
        value=[1, "two"]
    )


def test_records_that_differ_only_in_list_item_nullability_sort_deterministically():
    reg = read_typedefs({"A": {"xs": "list(Any)"}, "B": {"xs": "list(Optional(Any))"}})
    a, b = type_for("A", reg), type_for("B", reg)

    assert canonical_union_order((a, b)) == canonical_union_order((b, a)) == (a, b)


def test_lowercase_any_is_not_a_type():
    with pytest.raises(TypeCheckError, match="unknown type 'any'; write Any"):
        parse_type("any")


def test_a_typedef_cannot_be_named_any():
    with pytest.raises(TypeCheckError, match="shadows"):
        read_typedefs({"Any": "str"})


def test_any_resolves_through_an_alias_and_a_record_field():
    reg = read_typedefs({"Payload": "Any", "Rec": {"data": "Any"}})

    assert type_for("Payload", reg) == ANY
    assert type_for("Rec", reg).fields["data"] == ANY


# --------------------------------------------------------------------------- #
# Lists with unproven items
# --------------------------------------------------------------------------- #


def test_unproven_list_items_compare_as_optional_any():
    list_str = Type(kind=ValueKind.LIST_STRING, element=Type.scalar(ValueKind.STRING))
    list_file = Type(kind=ValueKind.LIST_ANY, element=Type.scalar(ValueKind.FILE))

    assert types_compatible(UNPROVEN_LIST, type_for("list(Optional(Any))", {}))
    assert types_compatible(list_str, UNPROVEN_LIST)
    assert not types_compatible(list_file, UNPROVEN_LIST)
    assert type_relation(UNPROVEN_LIST, list_str) == "narrower"


def test_proof_rejects_an_unproven_list_and_accepts_list_of_optional_any():
    with pytest.raises(LoadError) as caught:
        require_proven_type(UNPROVEN_LIST, binding_label="call.input.xs")
    assert "list(Optional(Any))" in str(caught.value)
    assert "no proven element type" in str(caught.value)

    proven = type_for("list(Optional(Any))", {})
    assert require_proven_type(proven, binding_label="call.input.xs") is proven


@pytest.mark.parametrize(
    ("text", "label"),
    [("list(list(int))", "list(list(int))"), ("list(Any)", "list(Any)")],
)
def test_a_refined_list_names_its_declared_type(text, label):
    with pytest.raises(TypeCheckError, match=re.escape(f"3 is not a list for declared {label}")):
        build_value_as(type_for(text, {}), 3)


# --------------------------------------------------------------------------- #
# Field reads on `Any`
# --------------------------------------------------------------------------- #


def test_a_field_read_on_any_is_not_checked():
    contract = ValueContract(ANY)

    assert output_contract_field_error(contract, ["anything"], "xs.anything") is None


def test_an_assert_reading_a_field_of_an_any_output_loads():
    load_flow(
        "id: p\nname: p\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        "  step:\n"
        "    kind: code\n"
        "    input:\n"
        "      x: ${input.s}\n"
        "    output: Any\n"
        "    code: return None\n"
        "    asserts:\n"
        "      - output.a.b == 1\n"
        "output: ${step.output}\n"
    )


# --------------------------------------------------------------------------- #
# Compatibility with `Any`
# --------------------------------------------------------------------------- #

_REGISTRY = read_typedefs({
    "Buy": {"action": "Literal(buy)", "qty": "int"},
    "Sell": {"action": "Literal(sell)", "qty": "int"},
    "Rec": {"a": "int"},
})
FILE = Type.scalar(ValueKind.FILE)
LIST_FILE = Type(kind=ValueKind.LIST_ANY, element=FILE)
FILE_RECORD = Type(kind=ValueKind.OBJECT, fields={"f": FILE}, required=frozenset({"f"}))


def _t(text: str) -> Type:
    """Resolve a type expression against the compatibility typedefs."""
    return type_for(text, _REGISTRY)


_JSON_TYPES = [
    "str", "int", "float", "bool", "date", "datetime", "object", "Rec", "list(int)",
    "list(Any)", "list(Optional(Any))", "Literal(a)", "int(1..3)", "int(0..)",
    "Union(Buy, Sell)", "Any",
]


@pytest.mark.parametrize("text", _JSON_TYPES)
def test_a_non_null_json_type_fits_any(text):
    assert types_compatible(_t(text), ANY)
    assert type_relation(_t(text), ANY) == "fits"


def test_an_unproven_list_fits_any():
    assert types_compatible(UNPROVEN_LIST, ANY)
    assert type_relation(UNPROVEN_LIST, ANY) == "fits"


@pytest.mark.parametrize("source", [_t("Optional(int)"), _t("None"), OPTIONAL_ANY])
def test_a_nullable_source_does_not_fit_any(source):
    assert not types_compatible(source, ANY)


@pytest.mark.parametrize("source", [_t("Optional(int)"), _t("None"), OPTIONAL_ANY, ANY])
def test_a_nullable_source_fits_optional_any(source):
    assert types_compatible(source, OPTIONAL_ANY)


@pytest.mark.parametrize(
    "sink",
    ["str", "int", "float", "object", "Rec", "Union(Buy, Sell)", "list(Any)", "list(str)",
     "Literal(a)", "int(1..3)", "Optional(str)", "None"],
)
def test_any_fits_nothing_but_any(sink):
    assert not types_compatible(ANY, _t(sink))


@pytest.mark.parametrize("source", [FILE, LIST_FILE, FILE_RECORD])
def test_a_type_holding_a_file_does_not_fit_any(source):
    assert not types_compatible(source, ANY)
    assert type_relation(source, ANY) == "disjoint"


def test_a_list_fits_list_of_any_when_its_items_fit_any():
    assert types_compatible(_t("list(int)"), LIST_ANY)
    assert types_compatible(_t("list(Rec)"), LIST_ANY)
    assert not types_compatible(_t("list(Optional(int))"), LIST_ANY)
    assert not types_compatible(LIST_FILE, LIST_ANY)


@pytest.mark.parametrize(
    "source",
    [_t("list(int)"), _t("list(Optional(str))"), LIST_ANY, LIST_OPTIONAL_ANY, UNPROVEN_LIST],
)
def test_a_list_of_json_items_fits_list_of_optional_any(source):
    assert types_compatible(source, LIST_OPTIONAL_ANY)


@pytest.mark.parametrize("source", [LIST_OPTIONAL_ANY, UNPROVEN_LIST])
@pytest.mark.parametrize("sink", [LIST_ANY, _t("list(str)")])
def test_a_list_of_optional_any_does_not_fit_a_list_of_non_null_items(source, sink):
    assert not types_compatible(source, sink)


@pytest.mark.parametrize("inferred", [_t("Optional(int)"), OPTIONAL_ANY, _t("Optional(Rec)")])
def test_a_nullable_json_type_is_narrower_than_any(inferred):
    assert type_relation(inferred, ANY) == "narrower"


@pytest.mark.parametrize(
    "declared",
    ["str", "int", "float(..2.5)", "int(1..3)", "int(0..)", "object", "Rec", "Buy",
     "Union(Buy, Sell)", "list(str)", "list(Any)", "date", "Literal(a)", "Optional(str)"],
)
def test_any_is_narrower_than_any_other_type(declared):
    assert type_relation(ANY, _t(declared)) == "narrower"


@pytest.mark.parametrize("declared", [ANY, OPTIONAL_ANY])
def test_any_fits_any_and_optional_any(declared):
    assert types_compatible(ANY, declared)
    assert type_relation(ANY, declared) == "fits"


@pytest.mark.parametrize("declared", [FILE, _t("None")])
def test_any_is_disjoint_from_file_and_none(declared):
    assert type_relation(ANY, declared) == "disjoint"


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        (_t("str"), True),
        (_t("Rec"), True),
        (_t("list(int)"), True),
        (ANY, True),
        (_t("None"), False),
        (FILE, True),  # a file slot reads a file's JSON form, which is an `Any` value
        (FILE_RECORD, True),
    ],
)
def test_any_overlaps_every_type_but_none(other, expected):
    assert types_overlap(ANY, other) is expected
    assert types_overlap(other, ANY) is expected


def test_a_record_with_an_any_field_overlaps_one_with_an_int_field():
    reg = read_typedefs({"A": {"v": "Any"}, "B": {"v": "int"}})

    assert types_overlap(type_for("A", reg), type_for("B", reg))


def test_any_overlaps_a_list_of_files_through_the_empty_list():
    list_of_files = Type(kind=ValueKind.LIST_ANY, element=Type.scalar(ValueKind.FILE))

    assert types_overlap(ANY, list_of_files)
    with pytest.raises(TypeCheckError, match="can match the same value"):
        read_typedefs({"A": {"v": "Any"}, "B": {"v": "list(file)"}, "W": "Union(A, B)"})


def test_an_untagged_union_of_any_and_int_fields_does_not_resolve():
    with pytest.raises(TypeCheckError, match="can match the same value"):
        read_typedefs({"A": {"v": "Any"}, "B": {"v": "int"}, "U": "Union(A, B)"})


_FILE_TYPEDEFS = {
    "F": {"f": "file"},
    "N": {"n": "int"},
    "U": "Union(F, N)",
    "OF": {"f": "Optional(file)", "n": "int"},
}


def test_any_is_narrower_than_a_union_with_one_file_member_and_back():
    reg = read_typedefs(_FILE_TYPEDEFS)
    union = type_for("U", reg)

    assert type_relation(ANY, union) == "narrower"
    assert type_relation(union, ANY) == "narrower"
    assert types_overlap(ANY, union)
    assert types_overlap(union, ANY)


def test_any_is_narrower_than_a_record_whose_file_field_may_be_null():
    reg = read_typedefs(_FILE_TYPEDEFS)

    assert type_relation(ANY, type_for("OF", reg)) == "narrower"
    assert type_relation(type_for("OF", reg), ANY) == "narrower"


@pytest.mark.parametrize(
    "typ",
    [
        FILE,
        Type(kind=ValueKind.FILE, nullable=True),
        LIST_FILE,
        FILE_RECORD,
        Type(kind=ValueKind.OBJECT, variants=(FILE_RECORD, FILE_RECORD)),
    ],
)
def test_requires_file_when_every_non_null_value_holds_one(typ):
    assert requires_file(typ)
    assert has_file(typ)


@pytest.mark.parametrize("text", ["U", "OF"])
def test_a_type_that_may_hold_a_file_does_not_require_one(text):
    typ = type_for(text, read_typedefs(_FILE_TYPEDEFS))

    assert has_file(typ)
    assert not requires_file(typ)


def test_requires_file_needs_a_non_null_list_element_and_a_required_field():
    optional_file = Type(kind=ValueKind.FILE, nullable=True)
    optional_field = Type(kind=ValueKind.OBJECT, fields={"f": FILE}, required=frozenset())

    assert not requires_file(Type(kind=ValueKind.LIST_ANY, element=optional_file))
    assert not requires_file(optional_field)
    assert not requires_file(ANY)


def test_an_untagged_union_of_any_and_a_union_with_a_file_member_does_not_resolve():
    typedefs = {**_FILE_TYPEDEFS, "A": {"v": "Any"}, "B": {"v": "U"}, "W": "Union(A, B)"}

    with pytest.raises(TypeCheckError, match="can match the same value"):
        read_typedefs(typedefs)


def test_tagging_an_any_output_as_a_union_with_a_file_member_loads():
    load_flow(
        "id: p\nname: p\n"
        "typedefs:\n  F:\n    f: file\n  N:\n    n: int\n  U: Union(F, N)\n"
        "nodes:\n"
        "  step:\n    kind: code\n    output: Any\n    code: return dict(n=1)\n"
        "  use:\n    kind: code\n    input:\n      x: !U ${step.output}\n"
        "    output: str\n    code: return 'ok'\n"
        "output: ${use.output}\n"
    )


def _call_flow(
    child_type: str,
    binding: str,
    *,
    step_output: str | None = "Any",
    over: bool = False,
) -> str:
    """Return a flow whose `c` node calls `child`, taking `t: <child_type>`, with `binding`.

    Args:
        child_type (`str`): The child's input tag text.
        binding (`str`): The parent's binding for `t`.
        step_output (`str | None`, *optional*, defaults to `"Any"`): The producer's
            `output:` type, or `None` for an untyped producer.
        over (`bool`, *optional*, defaults to `False`): Whether `c` maps over
            `${step.output}` instead of calling once.
    """
    output = "" if step_output is None else f"    output: {step_output}\n"
    node = "    kind: map\n    over: ${step.output}\n" if over else "    kind: call\n"
    return (
        "id: p\nname: p\n"
        "typedefs:\n  Rec:\n    a: int\n  F:\n    f: file\n"
        "defs:\n"
        "  child:\n"
        f"    input:\n      t: !{child_type}\n"
        "    nodes:\n"
        "      n:\n        kind: code\n        input:\n          t: ${input.t}\n"
        "        output: str\n        code: return repr(inputs['t'])\n"
        "    output: ${n.output}\n"
        "nodes:\n"
        f"  step:\n    kind: code\n{output}    code: return 'hello'\n"
        f"  c:\n{node}    call: child\n    input:\n      t: {binding}\n"
        "output: ${c.output}\n"
    )


def _load_error(text: str) -> str:
    """Return the message of the load error `text` raises."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return str(caught.value)


def test_any_into_a_child_str_input_suggests_a_tag_or_isinstance():
    message = _load_error(_call_flow("str", "${step.output}"))

    assert "child expects 'str', source is 'Any'" in message
    assert "`t: !str ${...}`" in message
    assert "isinstance" in message


def test_the_any_hint_names_the_typedef_of_a_child_record_input():
    message = _load_error(_call_flow("Rec", "${step.output}"))

    assert "`t: !Rec ${...}`" in message


def test_optional_any_into_a_child_any_input_is_a_plain_nullability_error():
    message = _load_error(_call_flow("Any", "${step.output}", step_output="Optional(Any)"))

    assert "child expects 'Any', source is 'Optional(Any)'" in message
    assert "isinstance" not in message


def test_any_into_a_child_input_that_requires_a_file_gets_no_hint():
    message = _load_error(_call_flow("F", "${step.output}"))

    assert "source is 'Any'" in message
    assert "isinstance" not in message


def test_a_tagged_any_binding_into_a_child_str_input_loads():
    load_flow(_call_flow("str", "!str ${step.output}"))


def test_an_untyped_source_into_a_child_any_input_is_left_to_runtime():
    load_flow(_call_flow("Any", "${step.output}", step_output=None))


def test_a_map_over_a_list_of_any_into_a_child_str_input_suggests_a_tag():
    message = _load_error(_call_flow("str", "${item}", step_output="list(Any)", over=True))

    assert "child expects 'str', source is 'Any'" in message
    assert "`t: !str ${...}`" in message
    assert "isinstance" in message


def test_a_map_over_a_list_of_any_into_a_child_any_input_loads():
    load_flow(_call_flow("Any", "${item}", step_output="list(Any)", over=True))


_ANY_INTO_LOOP = """\
id: p
name: p
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: return inputs["n"] + 1
    output:
      n: ${step.output}
nodes:
  seed:
    kind: code
    output: Any
    code: return 1
  loop:
    kind: loop
    times: 2
    input:
      n: SEED
    call: bump
output: ${loop.output}
"""


def test_any_into_a_loop_carried_int_suggests_a_tag_or_isinstance():
    with pytest.raises(LoadError) as caught:
        load_flow(_ANY_INTO_LOOP.replace("SEED", "${seed.output}"))

    message = str(caught.value)
    assert "source is 'Any'" in message
    assert "`n: !int ${...}`" in message
    assert "isinstance" in message


def test_a_tagged_loop_seed_over_an_any_output_loads():
    load_flow(_ANY_INTO_LOOP.replace("SEED", "!int ${seed.output}"))
