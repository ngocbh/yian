"""Unit tests for the authoring Type grammar + registry resolver."""

import pytest

from agent_composer.typesys.types import ListExpr, RefExpr, ScalarExpr, parse_type


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
    from agent_composer.typesys.types import OptionalExpr

    assert parse_type("Optional[str]") == OptionalExpr(ScalarExpr("str"))
    assert parse_type("Optional[Rating]") == OptionalExpr(RefExpr("Rating"))


def test_resolve_optional_is_nullable():
    from agent_composer.typesys.types import OptionalExpr, resolve_type

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
    from agent_composer.typesys.types import OptionalExpr

    assert parse_type("list[str]") == ListExpr(ScalarExpr("str"))
    assert parse_type("List[int]") == ListExpr(ScalarExpr("int"))
    assert parse_type("Optional[date]") == OptionalExpr(ScalarExpr("date"))
    assert parse_type("list[Rating]") == ListExpr(RefExpr("Rating"))


def test_parse_literal_quoted_and_unquoted():
    from agent_composer.typesys.types import EnumExpr

    assert parse_type("Literal[pro, con, mixed]") == EnumExpr(("pro", "con", "mixed"))
    assert parse_type('Literal["pro", "con"]') == EnumExpr(("pro", "con"))
    assert parse_type("Literal[defer]") == EnumExpr(("defer",))  # single member


def test_legacy_engine_names_no_longer_resolve():
    # type unification: the OLD engine vocabulary is gone. A bare `string`/`integer`/
    # `number`/`boolean` is no longer a scalar — it parses as an unknown registry RefExpr
    # and RAISES TypeCheckError on resolution.
    from agent_composer.typesys.values import TypeCheckError as _SE

    for legacy in ("string", "integer", "number", "boolean"):
        assert parse_type(legacy) == RefExpr(legacy)
        with pytest.raises(_SE):
            type_for(legacy, {})
    with pytest.raises(_SE):
        type_for("list[string]", {})
    # topics stays a domain alias -> list[str]
    assert parse_type("topics") == ListExpr(ScalarExpr("str"))


def test_parse_union_rejected():
    from agent_composer.typesys.values import TypeCheckError as _SE

    with pytest.raises(_SE) as ei:
        parse_type("Union[int, str]")
    assert "discriminated record" in str(ei.value) or "case" in str(ei.value)


def test_parse_malformed_raises():
    from agent_composer.typesys.values import TypeCheckError as _SE

    with pytest.raises(_SE):
        parse_type("list[")
    with pytest.raises(_SE):
        parse_type("123")  # a number literal is not a type


def test_is_shadow_guard():
    from agent_composer.typesys.types import _is_shadow

    assert _is_shadow("str") and _is_shadow("int") and _is_shadow("Optional")
    assert _is_shadow("Literal") and _is_shadow("Any") and _is_shadow("list")
    assert not _is_shadow("Rating") and not _is_shadow("Topic")
    assert not _is_shadow("string")  # legacy engine name is no longer a scalar keyword


# --- registry + resolve_type ----------------------------------------------- #

from agent_composer.typesys.values import (  # noqa: E402
    ListObjectValue,
    ObjectValue,
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
)
from agent_composer.typesys.types import (  # noqa: E402
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


def test_type_for_preserves_its_public_keyword_name():
    assert type_for(type_str="str", registry=REG).kind == ValueKind.STRING


def test_resolve_inline_literal_is_enum():
    from agent_composer.typesys.types import EnumExpr, resolve_type

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
    import agent_composer.typesys as state

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
