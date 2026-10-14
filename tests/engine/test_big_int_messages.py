"""Error messages that show a value still format when it holds an int too long to display."""

import math
import sys

import pytest

from yian import load_flow, run_flow
from yian.nodes.contracts import PartialRecordContract, validate_output_contract
from yian.typesys.values import (
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
    reject_nonfinite,
    value_repr,
)

BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit
PLACEHOLDER = "<int too large to display>"

STR = Type.scalar(ValueKind.STRING)
INT = Type.scalar(ValueKind.INTEGER)


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None:
        yield  # this Python has no limit
        return
    before = get()
    sys.set_int_max_str_digits(4300)
    try:
        yield
        after = get()
    finally:
        sys.set_int_max_str_digits(before)
    assert after == 4300, "yian code changed the global digit limit"


def lit(*tags: str) -> Type:
    return Type(kind=ValueKind.STRING, tags=frozenset(tags))


def rec(**fields: Type) -> Type:
    """A closed record whose fields are all required."""
    return Type(kind=ValueKind.OBJECT, fields=dict(fields), required=frozenset(fields))


def union(*members: Type) -> Type:
    return Type(kind=ValueKind.OBJECT, variants=tuple(members))


def test_value_repr_names_the_type_of_a_value_too_long_to_display():
    assert value_repr(BIG) == PLACEHOLDER
    assert value_repr([1, BIG]) == "<list too large to display>"
    assert value_repr({"a": BIG}) == "<dict too large to display>"
    assert value_repr("x") == "'x'"
    assert value_repr(12) == "12"


@pytest.mark.parametrize("typ, value, path, message", [
    (STR, BIG, (), f"{PLACEHOLDER} does not match declared type str"),
    (STR, [BIG], (), "<list too large to display> does not match declared type str"),
    (
        Type.scalar(ValueKind.LIST_STRING),
        [BIG],
        (0,),
        f"list element {PLACEHOLDER} is not str (declared list[str])",
    ),
    (
        Type.scalar(ValueKind.LIST_STRING),
        BIG,
        (),
        f"{PLACEHOLDER} is not a list for declared list[str]",
    ),
    (lit("a", "b"), BIG, (), f"{PLACEHOLDER} is not a member of variant ['a', 'b']"),
    (rec(a=INT), BIG, (), f"{PLACEHOLDER} is not an object for a record type"),
    (
        Type(kind=ValueKind.LIST_OBJECT, element=rec(a=INT)),
        BIG,
        (),
        f"{PLACEHOLDER} is not a list for declared list[object]",
    ),
    (
        union(rec(a=INT), rec(b=INT)),
        BIG,
        (),
        f"{PLACEHOLDER} is not an object for a union type",
    ),
    (
        union(rec(action=lit("buy")), rec(action=lit("sell"))),
        {"action": BIG},
        ("action",),
        f"{PLACEHOLDER} is not a tag of the union (expected one of ['buy', 'sell'])",
    ),
    (
        Type(kind=ValueKind.INTEGER, bounds=(0, 10)),
        BIG,
        (),
        f"{PLACEHOLDER} is outside inclusive int bounds (0, 10)",
    ),
], ids=[
    "scalar", "list-in-scalar", "list-element", "not-a-list", "variant", "record",
    "record-list", "untagged-union", "union-tag", "bounds",
])
def test_a_type_mismatch_holding_a_huge_int_raises_a_located_type_error(
    typ, value, path, message
):
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(typ, value)
    assert str(caught.value) == message
    assert caught.value.path == path


def test_a_huge_int_key_becomes_a_placeholder_path_segment():
    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite({BIG: math.nan}, "output")
    assert caught.value.path == (PLACEHOLDER,)
    assert str(caught.value) == (
        f"output at [{PLACEHOLDER!r}]: non-finite number nan is not a yian value"
    )


@pytest.mark.parametrize("value, message", [
    ({"a": 1, BIG: 1}, f"record has unknown fields: [{PLACEHOLDER}]"),
    ({"a": 1, BIG: 1, "z": 2}, f"record has unknown fields: [{PLACEHOLDER}, 'z']"),
], ids=["huge-key", "huge-and-str-keys"])
def test_a_record_names_an_unknown_huge_int_key(value, message):
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(rec(a=INT), value)
    assert str(caught.value) == message
    assert caught.value.path == ()


def test_a_partial_record_contract_names_an_unknown_huge_int_key():
    contract = PartialRecordContract(fields=(("a", None),))
    with pytest.raises(TypeCheckError) as caught:
        validate_output_contract(contract, {"a": 1, BIG: 1, "z": 2})
    assert str(caught.value) == f"record has unknown fields: [{PLACEHOLDER}, 'z']"
    assert caught.value.path == ()


def test_a_partial_record_contract_names_a_huge_int_it_rejects():
    contract = PartialRecordContract(fields=(("a", None),))
    with pytest.raises(TypeCheckError) as caught:
        validate_output_contract(contract, BIG)
    assert str(caught.value) == f"{PLACEHOLDER} is not an object for a partial record contract"


def test_a_code_node_returning_a_huge_int_for_a_str_output_fails_its_contract():
    flow = load_flow("""
id: big
name: big
nodes:
  n:
    kind: code
    output: str
    code: |
      return 10**4999
output: ${n.output}
""")

    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "n"
    assert result.failure.message == f"{PLACEHOLDER} does not match declared type str"


def test_a_wait_until_a_huge_int_fails_its_input_contract():
    flow = load_flow("""
id: big
name: big
nodes:
  w:
    kind: wait
    until: ${10 ** 4999}
output: ${w.output}
""")

    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.node_id == "w"
    assert result.failure.message == (
        "wait `until` ('${10 ** 4999}') did not resolve to a date/datetime: "
        f"{PLACEHOLDER}"
    )


def test_a_huge_int_input_name_is_reported_as_unknown():
    flow = load_flow("""
id: big
name: big
input:
  x: int
nodes:
  n:
    kind: code
    code: |
      return 1
output: ${n.output}
""")

    with pytest.raises(ValueError) as caught:
        run_flow(flow, {BIG: 1})

    assert str(caught.value) == f"unknown input {PLACEHOLDER}; declared inputs: x"
