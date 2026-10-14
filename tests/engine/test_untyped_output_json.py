"""Untyped values that cross a JSON boundary are checked as JSON values."""

from datetime import date
import re
from typing import Any

import pytest
from yian.runtime.log import canonical_typed
from yian.typesys.values import (
    FileRef,
    TypeCheckError,
    ValueKind,
    build_value,
    reject_non_json,
)


def _cyclic() -> list[Any]:
    """Return a list that contains itself."""
    value: list[Any] = []
    value.append(value)
    return value


@pytest.mark.parametrize(
    ("value", "message", "path"),
    [
        (
            {"d": {"x": [1, date(2026, 1, 1)]}},
            "output at d.x[1]: datetime.date(2026, 1, 1) is not a JSON value",
            ("d", "x", 1),
        ),
        ({1: "a"}, "output: object keys must be text, got 1", ()),
        ({"a": {1: "a"}}, "output at a: object keys must be text, got 1", ("a",)),
        ({"s": {1, 2}}, "output at s: {1, 2} is not a JSON value", ("s",)),
        ({1, 2}, "output: {1, 2} is not a JSON value", ()),
        (_cyclic(), "output at [0]: value contains itself", (0,)),
    ],
)
def test_reject_non_json_locates_the_bad_part(value, message, path):
    with pytest.raises(TypeCheckError) as caught:
        reject_non_json(value, "output")
    assert str(caught.value) == message
    assert caught.value.path == path


def test_reject_non_json_rejects_a_top_level_object():
    with pytest.raises(TypeCheckError) as caught:
        reject_non_json(object(), "output")
    assert re.fullmatch(r"output: <object object at .*> is not a JSON value", str(caught.value))
    assert caught.value.path == ()


@pytest.mark.parametrize(
    "value",
    [FileRef(uri="f"), [FileRef(uri="f")], {"f": FileRef(uri="f")}],
    ids=["top", "item", "field"],
)
def test_reject_non_json_accepts_files_only_when_asked(value):
    with pytest.raises(TypeCheckError, match=re.escape("is not a JSON value")):
        reject_non_json(value, "output")
    assert reject_non_json(value, "output", files=True) is None


def test_reject_non_json_with_files_still_rejects_other_values():
    with pytest.raises(TypeCheckError) as caught:
        reject_non_json({"f": FileRef(uri="f"), "d": date(2026, 1, 1)}, "output", files=True)
    assert str(caught.value) == "output at d: datetime.date(2026, 1, 1) is not a JSON value"
    assert caught.value.path == ("d",)


@pytest.mark.parametrize(
    "value",
    [None, True, 2**80, 1.5, "x", {"t": (1, 2)}, [None, {"a": []}]],
)
def test_reject_non_json_accepts_json_values(value):
    assert reject_non_json(value, "output") is None


@pytest.mark.parametrize(
    ("value", "message", "path"),
    [
        (
            {"d": date(2026, 1, 1)},
            "output at d: datetime.date(2026, 1, 1) is not a JSON value",
            ("d",),
        ),
        ({"s": {1, 2}}, "output at s: {1, 2} is not a JSON value", ("s",)),
        ({1: "a"}, "output: object keys must be text, got 1", ()),
        ({"a": {1: "a"}}, "output at a: object keys must be text, got 1", ("a",)),
    ],
)
def test_canonical_typed_rejects_untyped_non_json_values(value, message, path):
    with pytest.raises(TypeCheckError) as caught:
        canonical_typed(value, None)
    assert str(caught.value) == message
    assert caught.value.path == path


@pytest.mark.parametrize(
    "value",
    [{"f": FileRef(uri="f")}, [FileRef(uri="f")]],
    ids=["field", "item"],
)
def test_canonical_typed_accepts_untyped_files(value):
    wire, _ = canonical_typed(value, None)
    # A nested file is recorded as a plain object.
    assert {"uri": "f", "mime": None, "name": None} in (
        list(wire["value"].values()) if isinstance(wire["value"], dict) else wire["value"]
    )


def test_canonical_typed_does_not_walk_a_built_value(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a built value was walked")

    monkeypatch.setattr("yian.runtime.log.reject_non_json", fail)
    wire, _ = canonical_typed(build_value({"a": 1}), None)
    assert wire["value"] == {"a": 1}


@pytest.mark.parametrize(
    ("value", "declared", "message"),
    [
        ({1: "a"}, ValueKind.OBJECT, "object keys must be text, got 1"),
        ({"s": {1, 2}}, ValueKind.OBJECT, "{1, 2} is not a JSON value"),
        ({1: "a"}, ValueKind.STRING, "does not match declared type str"),
    ],
)
def test_canonical_typed_reports_the_type_check_for_typed_values(value, declared, message):
    with pytest.raises(TypeCheckError, match=re.escape(message)):
        canonical_typed(value, declared)


@pytest.mark.parametrize(
    ("value", "declared"),
    [("\ud800", ValueKind.STRING), ({"k": "\ud800"}, ValueKind.OBJECT), ("\ud800", None)],
    ids=["str", "object", "untyped"],
)
def test_canonical_typed_still_rejects_lone_surrogates(value, declared):
    with pytest.raises(ValueError, match="^Unicode surrogate is not valid canonical JSON$"):
        canonical_typed(value, declared)
