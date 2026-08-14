"""Unit tests for the typed segment value system.

Feature -> contract:
- build_value(value)            : raw Python value -> natural TypedValue
- build_value_as(t, v)   : value -> TypedValue of declared type, or raise
- AnyValue round-trip           : JSON dumps/loads preserves the exact type
"""

import pytest

from agent_composer.typesys.values import (
    ANY_VALUE_ADAPTER,
    BooleanValue,
    FileRef,
    FileValue,
    IntegerValue,
    ListAnyValue,
    ListNumberValue,
    ListStringValue,
    NumberValue,
    ObjectValue,
    TypeCheckError,
    ValueKind,
    StringValue,
    build_value,
    build_value_as,
)


# --- inference -------------------------------------------------------------- #


def test_build_segment_infers_scalars():
    assert isinstance(build_value("hi"), StringValue)
    assert isinstance(build_value(3), IntegerValue)
    assert isinstance(build_value(3.5), NumberValue)
    assert isinstance(build_value({"a": 1}), ObjectValue)
    assert build_value(None).kind == ValueKind.NONE


def test_bool_is_not_int():
    # bool subclasses int in Python; the value system must keep them distinct.
    assert isinstance(build_value(True), BooleanValue)
    assert build_value(True).kind == ValueKind.BOOLEAN
    assert build_value(1).kind == ValueKind.INTEGER


def test_list_inference():
    assert isinstance(build_value(["a", "b"]), ListStringValue)
    assert isinstance(build_value([1, 2.0]), ListNumberValue)  # mixed int/float -> number
    assert isinstance(build_value([]), ListAnyValue)
    assert isinstance(build_value([1, "a"]), ListAnyValue)  # heterogeneous -> any


def test_file_is_never_auto_inferred():
    # A dict that looks file-ish stays an object; only an explicit FileRef is a file.
    assert isinstance(build_value({"uri": "s3://x"}), ObjectValue)
    assert isinstance(build_value(FileRef(uri="s3://x")), FileValue)


def test_build_segment_idempotent():
    seg = build_value(5)
    assert build_value(seg) is seg


def test_unwrappable_value_raises():
    with pytest.raises(TypeCheckError):
        build_value(object())


# --- declared-type write boundary ------------------------------------------ #


def test_with_type_widens_int_to_number():
    seg = build_value_as(ValueKind.NUMBER, 7)
    assert isinstance(seg, NumberValue)
    assert seg.value == 7.0 and isinstance(seg.value, float)


def test_with_type_rejects_mismatch():
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.INTEGER, "not-an-int")
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.INTEGER, True)  # bool is not int here


def test_with_type_typed_list_validates_elements():
    seg = build_value_as(ValueKind.LIST_STRING, ["ACME", "BETA"])
    assert isinstance(seg, ListStringValue)
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.LIST_STRING, ["ACME", 3])


# --- lossless serialization (the run-record primitive) ---------------------- #


@pytest.mark.parametrize(
    "value, expected_type, expected_py",
    [
        (3, ValueKind.INTEGER, int),
        (3.0, ValueKind.NUMBER, float),
        (True, ValueKind.BOOLEAN, bool),
        ("x", ValueKind.STRING, str),
        ({"k": [1, 2]}, ValueKind.OBJECT, dict),
        (["a", "b"], ValueKind.LIST_STRING, list),
    ],
)
def test_json_round_trip_preserves_type(value, expected_type, expected_py):
    seg = build_value(value)
    blob = ANY_VALUE_ADAPTER.dump_json(seg)
    back = ANY_VALUE_ADAPTER.validate_json(blob)
    assert back.kind == expected_type
    # int-vs-float-vs-bool survive the JSON number ambiguity via the type tag.
    assert type(back.value) is expected_py
    assert back.value == value


def test_file_segment_round_trip():
    seg = build_value(FileRef(uri="s3://b/k", mime="text/csv", name="d.csv"))
    back = ANY_VALUE_ADAPTER.validate_json(ANY_VALUE_ADAPTER.dump_json(seg))
    assert isinstance(back, FileValue)
    assert back.value.uri == "s3://b/k" and back.value.name == "d.csv"


# --- date scalar ------------------------------------------------------------ #


def test_date_segment_build_and_roundtrip():
    from agent_composer.typesys.values import DateValue

    seg = build_value_as(ValueKind.DATE, "2026-06-08")
    assert isinstance(seg, DateValue)
    assert seg.value == "2026-06-08"
    # lossless JSON round-trip via the discriminated union
    back = ANY_VALUE_ADAPTER.validate_python(seg.model_dump())
    assert isinstance(back, DateValue) and back.value == "2026-06-08"


def test_date_segment_rejects_nondate():
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.DATE, "not-a-date")
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.DATE, 20260608)


def test_plain_string_not_inferred_as_date():
    assert build_value("2026-06-08").kind == ValueKind.STRING


# --- datetime scalar -------------------------------------------------------- #


def test_datetime_segment_build_and_roundtrip():
    from agent_composer.typesys.values import DateTimeValue

    seg = build_value_as(ValueKind.DATETIME, "2026-06-12T14:30:00+00:00")
    assert isinstance(seg, DateTimeValue)
    assert seg.value == "2026-06-12T14:30:00+00:00"
    # lossless JSON round-trip via the discriminated union
    back = ANY_VALUE_ADAPTER.validate_python(seg.model_dump())
    assert isinstance(back, DateTimeValue) and back.value == "2026-06-12T14:30:00+00:00"


def test_datetime_segment_rejects_nondatetime():
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.DATETIME, "not-a-datetime")
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.DATETIME, 20260612)


def test_datetime_distinct_from_date():
    # a bare DATE string must NOT type-check as a datetime (date and datetime are distinct
    # scalars; datetime.fromisoformat would otherwise accept a bare date as midnight).
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.DATETIME, "2026-06-12")
    # and a datetime string is not a date
    with pytest.raises(TypeCheckError):
        build_value_as(ValueKind.DATE, "2026-06-12T14:30:00+00:00")


def test_plain_string_not_inferred_as_datetime():
    assert build_value("2026-06-12T14:30:00+00:00").kind == ValueKind.STRING


# --- structural Type (records / variants / typed lists) -------------------- #


def test_type_back_compat_kind_accepted():
    from agent_composer.typesys.values import Type

    assert isinstance(build_value_as(Type.scalar(ValueKind.STRING), "x"), StringValue)
    # passing a bare ValueKind still works (back-compat)
    assert build_value_as(ValueKind.NUMBER, 3).value == 3.0


def test_type_variant_membership():
    from agent_composer.typesys.values import Type

    action = Type(kind=ValueKind.STRING, tags=frozenset({"Approve", "Reject", "Defer"}))
    assert build_value_as(action, "Approve").value == "Approve"
    with pytest.raises(TypeCheckError):
        build_value_as(action, "approve")


def test_type_record_fields():
    from agent_composer.typesys.values import Type

    rating = Type(
        kind=ValueKind.OBJECT,
        fields={
            "value": Type.scalar(ValueKind.NUMBER),
            "confidence": Type.scalar(ValueKind.NUMBER),
        },
        required=frozenset({"value", "confidence"}),
    )
    assert isinstance(build_value_as(rating, {"value": 0.8, "confidence": 0.9}), ObjectValue)
    with pytest.raises(TypeCheckError):  # missing required field
        build_value_as(rating, {"value": 0.8})
    with pytest.raises(TypeCheckError):  # unknown field
        build_value_as(rating, {"value": 0.8, "confidence": 0.9, "x": 1})
    with pytest.raises(TypeCheckError):  # wrong field type
        build_value_as(rating, {"value": "hi", "confidence": 0.9})


def test_type_nullable_field_accepts_none_and_absent():
    from agent_composer.typesys.values import Type

    sig = Type(
        kind=ValueKind.OBJECT,
        fields={
            "score": Type.scalar(ValueKind.NUMBER),
            "note": Type(kind=ValueKind.STRING, nullable=True),
        },
        required=frozenset({"score"}),  # note is Optional -> not required
    )
    # present-None on the nullable field -> ok
    assert isinstance(build_value_as(sig, {"score": 0.5, "note": None}), ObjectValue)
    # absent nullable field -> ok
    assert isinstance(build_value_as(sig, {"score": 0.5}), ObjectValue)
    # a present non-null value on the nullable field still type-checks
    assert isinstance(build_value_as(sig, {"score": 0.5, "note": "hi"}), ObjectValue)
    # a non-nullable field still rejects None
    with pytest.raises(TypeCheckError):
        build_value_as(sig, {"score": None, "note": "x"})


def test_type_list_of_record():
    from agent_composer.typesys.values import ListObjectValue, Type

    rating = Type(
        kind=ValueKind.OBJECT,
        fields={"value": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"value"}),
    )
    lst = Type(kind=ValueKind.LIST_OBJECT, element=rating)
    seg = build_value_as(lst, [{"value": 1.0}, {"value": 2.0}])
    assert isinstance(seg, ListObjectValue) and len(seg.value) == 2
    with pytest.raises(TypeCheckError):
        build_value_as(lst, [{"value": "bad"}])
