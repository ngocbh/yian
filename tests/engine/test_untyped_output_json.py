"""Untyped values that cross a JSON boundary are checked as JSON values."""

from datetime import date
import re
from typing import Any

import pytest
from yian.typesys.values import FileRef, TypeCheckError, reject_non_json


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
