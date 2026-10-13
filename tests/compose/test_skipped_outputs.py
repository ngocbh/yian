"""Output contracts for bindings that default a maybe-skipped reference to null."""

from __future__ import annotations

from dataclasses import replace

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.output_contracts import resolve_output_binding_contract
from yian.diagnostics import OriginRef
from yian.nodes.contracts import (
    ContractOrigins,
    PartialRecordContract,
    ValueContract,
)
from yian.typesys import Type, ValueKind

STR = Type.scalar(ValueKind.STRING)
INT = Type.scalar(ValueKind.INTEGER)
OPTIONAL_STR = replace(STR, nullable=True)


def _record(**fields: Type) -> Type:
    """Return a closed record type whose fields are all required."""
    return Type(kind=ValueKind.OBJECT, fields=fields, required=frozenset(fields))


def test_null_default_over_a_reference_is_optional_and_keeps_origins() -> None:
    """`${p.output:-null}` types as Optional of the producer's type, same origins."""
    origins = ContractOrigins(default=OriginRef("producer-output"))
    producers = {"p": ValueContract(STR, origins)}

    resolved = resolve_output_binding_contract("${p.output:-null}", producers, {})

    assert resolved == ValueContract(OPTIONAL_STR, origins)


def test_null_default_over_a_dotted_field_is_optional() -> None:
    """A dotted field read with a null default types as Optional of the field."""
    producers = {"p": ValueContract(_record(f=INT))}

    resolved = resolve_output_binding_contract("${p.output.f :- null}", producers, {})

    assert type(resolved) is ValueContract
    assert resolved.type == replace(INT, nullable=True)


def test_null_default_over_a_coalesce_of_equal_types_is_optional() -> None:
    """`(a | b) :- null` over equal operand types is Optional of that type."""
    producers = {"a": ValueContract(STR), "b": ValueContract(STR)}

    resolved = resolve_output_binding_contract(
        "${(a.output | b.output) :- null}", producers, {}
    )

    assert resolved == ValueContract(OPTIONAL_STR)


def test_null_default_over_a_coalesce_of_different_types_is_untyped() -> None:
    """`(a | b) :- null` over different operand types stays unproven."""
    producers = {"a": ValueContract(STR), "b": ValueContract(INT)}

    assert (
        resolve_output_binding_contract(
            "${(a.output | b.output) :- null}", producers, {}
        )
        is None
    )


def test_null_default_over_a_nullable_head_stays_optional() -> None:
    """An already-nullable head keeps its Optional type."""
    producers = {"p": ValueContract(OPTIONAL_STR)}

    resolved = resolve_output_binding_contract("${p.output :- null}", producers, {})

    assert type(resolved) is ValueContract
    assert resolved.type == OPTIONAL_STR


def test_null_default_over_a_partial_or_unknown_producer_is_untyped() -> None:
    """A partial-record head or an unknown producer stays unproven."""
    partial = PartialRecordContract(fields=(("n", ValueContract(INT)), ("m", None)))

    assert (
        resolve_output_binding_contract("${p.output :- null}", {"p": partial}, {})
        is None
    )
    assert resolve_output_binding_contract("${q.output :- null}", {}, {}) is None


def test_other_defaults_stay_untyped() -> None:
    """A non-null default, or a nested default under a null default, stays unproven."""
    producers = {"p": ValueContract(STR), "a": ValueContract(STR), "b": ValueContract(STR)}

    assert (
        resolve_output_binding_contract('${p.output :- "x"}', producers, {}) is None
    )
    assert (
        resolve_output_binding_contract(
            "${(a.output :- ${b.output}) :- null}", producers, {}
        )
        is None
    )


def test_null_default_over_a_flow_input_is_optional() -> None:
    """`${input.x :- null}` over a str flow input types as Optional[str]."""
    resolved = resolve_output_binding_contract(
        "${input.x :- null}", {}, {"x": ValueContract(STR)}
    )

    assert type(resolved) is ValueContract
    assert resolved.type == OPTIONAL_STR


_NULL_DEFAULT_ON = """
id: f
name: f
nodes:
  cls:
    kind: code
    output: Literal[a, b, c]
    code: |
      return "a"
  route:
    kind: case
    on: ${cls.output :- null}
    cases:
      - when: a
        then: na
      - when: b
        then: nb
  na:
    kind: code
    output: str
    code: |
      return "A"
  nb:
    kind: code
    output: str
    code: |
      return "B"
output: ${na.output | nb.output}
"""


def test_on_over_a_null_default_is_checked_for_exhaustiveness() -> None:
    """`on: ${cls.output :- null}` now has the enum's type, so a missing tag is a load error."""
    with pytest.raises(LoadError, match="non-exhaustive"):
        load_flow(_NULL_DEFAULT_ON)
