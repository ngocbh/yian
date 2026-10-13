"""Unit tests for the authoring Type grammar + registry resolver."""

import pytest

from yian.typesys.types import ListExpr, RefExpr, ScalarExpr, parse_type


def test_parse_scalars():
    assert parse_type("str") == ScalarExpr("str")
    assert parse_type("int") == ScalarExpr("int")
    assert parse_type("float") == ScalarExpr("float")
    assert parse_type("date") == ScalarExpr("date")
    assert parse_type("datetime") == ScalarExpr("datetime")
    assert parse_type("bool") == ScalarExpr("bool")
    assert parse_type("object") == ScalarExpr("object")


def test_parse_lists_and_topics():
    assert parse_type("List[float]") == ListExpr(ScalarExpr("float"))
    assert parse_type("list[str]") == ListExpr(ScalarExpr("str"))
    assert parse_type("topics") == ListExpr(ScalarExpr("str"))
    assert parse_type("List[Rating]") == ListExpr(RefExpr("Rating"))


def test_parse_ref():
    assert parse_type("Rating") == RefExpr("Rating")
    assert parse_type("Action") == RefExpr("Action")


def test_parse_optional():
    from yian.typesys.types import OptionalExpr

    assert parse_type("Optional[str]") == OptionalExpr(ScalarExpr("str"))
    assert parse_type("Optional[Rating]") == OptionalExpr(RefExpr("Rating"))


def test_resolve_optional_is_nullable():
    from yian.typesys.types import OptionalExpr, resolve_type

    sh = resolve_type(OptionalExpr(ScalarExpr("str")), {})
    assert sh.kind == ValueKind.STRING and sh.nullable is True


def test_record_optional_field_excluded_from_required():
    reg = {"Sig": RecordDef(fields={"score": "float", "note": "Optional[str]"})}
    sh = type_for("Sig", reg)
    assert sh.required == frozenset({"score"})  # note is Optional -> not required
    assert sh.fields["note"].nullable is True
    assert sh.fields["score"].nullable is False


# --- ast type-expr parser (Python + engine names) ----------------- #


def test_parse_python_scalar_names():
    assert parse_type("str") == ScalarExpr("str")
    assert parse_type("int") == ScalarExpr("int")
    assert parse_type("float") == ScalarExpr("float")
    assert parse_type("bool") == ScalarExpr("bool")
    assert parse_type("Any") == ScalarExpr("object")
    assert parse_type("date") == ScalarExpr("date")
    assert parse_type("datetime") == ScalarExpr("datetime")


def test_parse_python_generics():
    from yian.typesys.types import OptionalExpr

    assert parse_type("list[str]") == ListExpr(ScalarExpr("str"))
    assert parse_type("List[int]") == ListExpr(ScalarExpr("int"))
    assert parse_type("Optional[date]") == OptionalExpr(ScalarExpr("date"))
    assert parse_type("list[Rating]") == ListExpr(RefExpr("Rating"))


def test_parse_literal_quoted_and_unquoted():
    from yian.typesys.types import EnumExpr

    assert parse_type("Literal[pro, con, mixed]") == EnumExpr(("pro", "con", "mixed"))
    assert parse_type('Literal["pro", "con"]') == EnumExpr(("pro", "con"))
    assert parse_type("Literal[defer]") == EnumExpr(("defer",))  # single member


def test_legacy_engine_names_no_longer_resolve():
    # type unification: the OLD engine vocabulary is gone. A bare `string`/`integer`/
    # `number`/`boolean` is no longer a scalar — it parses as an unknown registry RefExpr
    # and RAISES TypeCheckError on resolution.
    from yian.typesys.values import TypeCheckError as _SE

    for legacy in ("string", "integer", "number", "boolean"):
        assert parse_type(legacy) == RefExpr(legacy)
        with pytest.raises(_SE):
            type_for(legacy, {})
    with pytest.raises(_SE):
        type_for("list[string]", {})
    # topics stays a domain alias -> list[str]
    assert parse_type("topics") == ListExpr(ScalarExpr("str"))


def test_parse_union_rejected():
    from yian.typesys.values import TypeCheckError as _SE

    with pytest.raises(_SE) as ei:
        parse_type("Union[int, str]")
    assert "discriminated record" in str(ei.value) or "case" in str(ei.value)


def test_parse_malformed_raises():
    from yian.typesys.values import TypeCheckError as _SE

    with pytest.raises(_SE):
        parse_type("list[")
    with pytest.raises(_SE):
        parse_type("123")  # a number literal is not a type


def test_is_shadow_guard():
    from yian.typesys.types import _is_shadow

    assert _is_shadow("str") and _is_shadow("int") and _is_shadow("Optional")
    assert _is_shadow("Literal") and _is_shadow("Any") and _is_shadow("list")
    assert not _is_shadow("Rating") and not _is_shadow("Topic")
    assert not _is_shadow("string")  # legacy engine name is no longer a scalar keyword


# --- registry + resolve_type ----------------------------------------------- #

from yian.typesys.values import (  # noqa: E402
    ListObjectValue,
    ObjectValue,
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
)
from yian.typesys.types import (  # noqa: E402
    RecordDef,
    VariantDef,
    resolve_type,  # noqa: F401
    type_for,
)

REG = {
    "Rating": RecordDef(fields={"value": "float", "confidence": "float"}),
    "Action": VariantDef(tags=("Approve", "Reject", "Defer")),
    "Prices": RecordDef(fields={"closes": "List[float]", "last": "float"}),
}


def test_resolve_scalar_and_list():
    assert type_for("str", REG).kind == ValueKind.STRING
    assert type_for("date", REG).kind == ValueKind.DATE
    assert type_for("datetime", REG).kind == ValueKind.DATETIME
    assert type_for("List[float]", REG).kind == ValueKind.LIST_NUMBER


@pytest.mark.parametrize("expression,bounds,accepted,rejected", [
    ("int[1, 12]", (1, 12), (1, 4, 12), (0, 13, True, 1.5, "4")),
    ("float[-0.5, 1.0]", (-0.5, 1.0), (-0.5, 0, 1.0), (-0.6, 1.1, True, "0")),
])
def test_bounded_numeric_types_validate_inclusive_values(expression, bounds, accepted, rejected):
    typ = type_for(expression, {})
    assert typ.bounds == bounds
    typ.validate_shape()
    for value in accepted:
        assert build_value_as(typ, value).to_object() == value
    for value in rejected:
        with pytest.raises(TypeCheckError):
            build_value_as(typ, value)


@pytest.mark.parametrize("expression,accepted,rejected", [
    ("int[-12, -1]", (-12, -4, -1), (-13, 0)),
    ("int[+4, 4]", (4,), (3, 5)),
    ("float[-0.25, +0.25]", (-0.25, 0, 0.25), (-0.251, 0.251)),
    ("float[2, 2]", (2, 2.0), (1.9, 2.1)),
])
def test_bounded_numeric_signed_and_equal_endpoints(expression, accepted, rejected):
    typ = type_for(expression, {})
    for value in accepted:
        assert build_value_as(typ, value).to_object() == value
    for value in (*rejected, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(TypeCheckError):
            build_value_as(typ, value)


def test_large_integer_bounds_remain_exact():
    large = 10 ** 400
    typ = type_for(f"int[{-large}, {large}]", {})
    typ.validate_shape()
    assert typ.bounds == (-large, large)
    for value in (-large, 0, large):
        assert build_value_as(typ, value).to_object() == value
    with pytest.raises(TypeCheckError):
        build_value_as(typ, large + 1)


@pytest.mark.parametrize("expression", [
    "int[]", "int[1]", "int[1, 2, 3]", "float[0]", "float[0, 1, 2]",
    "int[2, 1]", "float[2.0, 1.0]", "int[True, 2]", "int[0, False]",
    "float[False, 1]", "float[0, True]", "int[1.0, 2]", "int[1, 2.0]",
    "float[0, 1e999]", "float[-1e999, 0]", "float[nan, 1]", "float[0, inf]",
    "int['1', 2]", "float[0, None]", "int[1 + 1, 3]", "int[-(1 + 1), 3]",
    "float[0, float('inf')]", "int[lo, hi]", "int[1:2]", "int[[1, 2]]",
    "int[--1, 2]", "float[0j, 1]", "bool[0, 1]", "str[1, 2]",
])
def test_numeric_bounds_reject_nonliteral_or_malformed_endpoints(expression):
    with pytest.raises(TypeCheckError):
        parse_type(expression)


@pytest.mark.parametrize("kind,bounds", [
    (ValueKind.INTEGER, [1, 2]), (ValueKind.INTEGER, (1,)),
    (ValueKind.INTEGER, (1, 2, 3)), (ValueKind.INTEGER, (True, 2)),
    (ValueKind.INTEGER, (1, 2.0)), (ValueKind.INTEGER, (2, 1)),
    (ValueKind.NUMBER, (0, False)), (ValueKind.NUMBER, (None, 1)),
    (ValueKind.NUMBER, (float("nan"), 1)), (ValueKind.NUMBER, (0, float("inf"))),
    (ValueKind.STRING, (1, 2)), (ValueKind.LIST_INTEGER, (1, 2)),
    (ValueKind.OBJECT, (1, 2)),
])
def test_numeric_bounds_shape_rejects_forged_metadata(kind, bounds):
    typ = Type(kind=kind)
    object.__setattr__(typ, "bounds", bounds)
    with pytest.raises((TypeError, ValueError)) as caught:
        typ.validate_shape()
    assert type(caught.value) is not TypeCheckError


def test_numeric_bounds_do_not_combine_with_incompatible_refinements():
    for refinement in ({"fields": {}}, {"tags": frozenset({"a"})},
                       {"element": Type.scalar(ValueKind.INTEGER)}):
        with pytest.raises(ValueError):
            Type(ValueKind.INTEGER, bounds=(1, 2), **refinement).validate_shape()


def test_bounded_type_defaults_preserve_compatible_integer_payloads():
    from yian.compose.helpers.errors import LoadError
    from yian.compose.helpers.types import read_flow_inputs
    from yian.nodes.binding import apply_defaults

    params = read_flow_inputs({"score": "float[0, 1] = 1"}, {})
    resolved = apply_defaults(params, {})
    assert resolved == {"score": 1} and type(resolved["score"]) is int
    for declaration in ("int[1, 12] = 13", "float[0, 1] = 1.01", "int[0, 1] = true"):
        with pytest.raises(LoadError):
            read_flow_inputs({"score": declaration}, {})


def test_bounded_numeric_labels_include_nested_domains():
    from yian.typesys.values import canonical_type_label

    assert canonical_type_label(type_for("int[ -2, +4 ]", {})) == "int[-2,4]"
    assert canonical_type_label(type_for("list[Optional[float[0.0, 1.5]]]", {})) == (
        "list[Optional[float[0.0,1.5]]]"
    )


@pytest.mark.parametrize("endpoint", [10**400, -(10**400), 2**53 + 1, -(2**53 + 1)])
def test_float_domain_rejects_endpoints_without_lossless_finite_representation(endpoint):
    with pytest.raises(TypeCheckError, match="finite|lossless"):
        type_for(f"float[{endpoint}, {endpoint}]", {})
    with pytest.raises(ValueError, match="finite|lossless") as caught:
        Type(ValueKind.NUMBER, bounds=(endpoint, endpoint)).validate_shape()
    assert type(caught.value) is not TypeCheckError


def test_bounded_typed_write_preserves_the_previous_value_on_nested_rejection():
    from yian.typesys.pool import VariablePool

    typ = type_for("list[Score]", {
        "Score": RecordDef(fields={"value": "float[0, 1]"}),
    })
    pool = VariablePool()
    pool.set("scores", [{"value": 1}], typ)
    with pytest.raises(TypeCheckError) as caught:
        pool.set("scores", [{"value": 0.5}, {"value": 1.1}], typ)
    assert caught.value.path == (1, "value")
    assert pool.get("scores") == [{"value": 1}]
    assert type(pool.get("scores")[0]["value"]) is int


def test_type_for_preserves_its_public_keyword_name():
    assert type_for(type_str="str", registry=REG).kind == ValueKind.STRING


def test_resolve_inline_literal_is_enum():
    from yian.typesys.types import EnumExpr, resolve_type

    sh = resolve_type(EnumExpr(("pro", "con")), {})
    assert sh.kind == ValueKind.STRING and sh.tags == frozenset({"pro", "con"})


def test_resolve_dict_is_lenient_object_placeholder():
    # dict[K,V] parses and resolves to a lenient object placeholder.
    sh = type_for("dict[str, int]", REG)
    assert sh.kind == ValueKind.OBJECT


def test_resolve_record_and_variant():
    s = type_for("Rating", REG)
    assert s.kind == ValueKind.OBJECT and set(s.required) == {"value", "confidence"}
    a = type_for("Action", REG)
    assert a.kind == ValueKind.STRING and a.tags == frozenset({"Approve", "Reject", "Defer"})


def test_resolve_list_of_record():
    s = type_for("List[Rating]", REG)
    assert s.kind == ValueKind.LIST_OBJECT and s.element.fields is not None


def test_unknown_type_raises():
    with pytest.raises(TypeCheckError):
        type_for("Nope", REG)


def test_recursive_record_rejected():
    bad = {"Node": RecordDef(fields={"child": "Node"})}
    with pytest.raises(TypeCheckError):
        type_for("Node", bad)


def test_runtime_type_shape_accepts_resolved_and_shared_graphs():
    shared = type_for("Rating", REG)
    Type(
        kind=ValueKind.OBJECT,
        fields={"first": shared, "second": shared},
        required=frozenset({"first", "second"}),
    ).validate_shape()
    Type(
        kind=ValueKind.LIST_ANY,
        element=Type.scalar(ValueKind.STRING),
    ).validate_shape()
    for expression in (
        "str",
        "Optional[str]",
        "Action",
        "Rating",
        "List[Action]",
        "List[Rating]",
        "List[date]",
        "List[List[int]]",
    ):
        type_for(expression, REG).validate_shape()


@pytest.mark.parametrize(
    ("typ", "error_type", "match"),
    [
        (Type(kind="str"), TypeError, "kind"),
        (Type(kind=ValueKind.STRING, nullable=1), TypeError, "nullable"),
        (Type(kind=ValueKind.OBJECT, fields=[]), TypeError, "fields"),
        (
            Type(kind=ValueKind.OBJECT, fields={}, required=set()),
            TypeError,
            "required fields",
        ),
        (Type(kind=ValueKind.STRING, tags={"yes"}), TypeError, "tags"),
        (Type(kind=ValueKind.LIST_STRING, element="str"), TypeError, "nested type"),
        (
            Type(kind=ValueKind.STRING, fields={}),
            ValueError,
            "only object types",
        ),
        (
            Type(kind=ValueKind.OBJECT, required=frozenset()),
            ValueError,
            "need a record field mapping",
        ),
        (
            Type(
                kind=ValueKind.OBJECT,
                fields={},
                required=frozenset({"missing"}),
            ),
            ValueError,
            "unknown record fields",
        ),
        (
            Type(kind=ValueKind.OBJECT, tags=frozenset({"yes"})),
            ValueError,
            "only string types",
        ),
        (
            Type(kind=ValueKind.STRING, element=Type.scalar(ValueKind.STRING)),
            ValueError,
            "only list types",
        ),
        (
            Type(
                kind=ValueKind.OBJECT,
                fields={},
                tags=frozenset({"yes"}),
            ),
            ValueError,
            "structural refinement|only string types",
        ),
    ],
)
def test_runtime_type_shape_rejects_malformed_carriers(
    typ: Type,
    error_type: type[Exception],
    match: str,
):
    with pytest.raises(error_type, match=match):
        typ.validate_shape()


@pytest.mark.parametrize(
    "typ",
    [
        Type(
            kind=ValueKind.LIST_INTEGER,
            element=Type.scalar(ValueKind.STRING),
        ),
        Type(
            kind=ValueKind.LIST_STRING,
            element=Type.scalar(ValueKind.DATE),
        ),
        Type(
            kind=ValueKind.LIST_OBJECT,
            element=Type(kind=ValueKind.STRING, tags=frozenset({"yes"})),
        ),
    ],
)
def test_runtime_type_shape_rejects_mismatched_list_storage(typ: Type):
    with pytest.raises(ValueError, match="list storage kind"):
        typ.validate_shape()


@pytest.mark.parametrize("metadata", ["bad\ud800"])
def test_runtime_type_shape_rejects_noncanonical_metadata(metadata: str):
    for typ in (
        Type(
            kind=ValueKind.OBJECT,
            fields={metadata: Type.scalar(ValueKind.STRING)},
            required=frozenset({metadata}),
        ),
        Type(kind=ValueKind.STRING, tags=frozenset({metadata})),
    ):
        with pytest.raises(ValueError, match="Unicode surrogate"):
            typ.validate_shape()


def test_runtime_type_shape_rejects_cycles():
    cyclic = Type(kind=ValueKind.OBJECT, fields={})
    object.__setattr__(cyclic, "fields", {"self": cyclic})

    with pytest.raises(ValueError, match="acyclic"):
        cyclic.validate_shape()


def test_end_to_end_write_boundary():
    assert build_value_as(type_for("Action", REG), "Approve").value == "Approve"
    with pytest.raises(TypeCheckError):
        build_value_as(type_for("Action", REG), "approve")
    assert isinstance(
        build_value_as(type_for("Rating", REG), {"value": 0.8, "confidence": 0.9}),
        ObjectValue,
    )
    assert isinstance(
        build_value_as(type_for("List[Rating]", REG), [{"value": 0.1, "confidence": 0.2}]),
        ListObjectValue,
    )
    assert isinstance(
        build_value_as(type_for("Prices", REG), {"closes": [1.0, 2.0], "last": 2.0}),
        ObjectValue,
    )


def test_public_exports():
    import yian.typesys as state

    for name in (
        "Type",
        "DateValue",
        "parse_type",
        "resolve_type",
        "type_for",
        "RecordDef",
        "VariantDef",
    ):
        assert hasattr(state, name), f"missing export: {name}"
