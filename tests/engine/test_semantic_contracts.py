"""Unit contracts for the provenance-bearing semantic IR vocabulary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest

import yian.nodes as nodes_api
from yian.diagnostics import OriginRef
from yian.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    OutputContract,
    PartialRecordContract,
    Parameter,
    ValueContract,
    ValuePath,
    as_binding,
    binding_source,
    output_contract_field_error,
    output_contract_origin,
    validate_output_contract,
)
from yian.typesys import Type, ValueKind


def _origin(key: str) -> OriginRef:
    """Build a short valid origin for carrier tests."""
    return OriginRef(key)


def test_carriers_have_exact_final_fields() -> None:
    """The shared vocabulary must not grow parallel semantic authorities."""
    for carrier in (
        Binding,
        Assertion,
        ContractOrigins,
        ValueContract,
        PartialRecordContract,
        Parameter,
    ):
        assert carrier.__dataclass_params__.frozen is True

    assert [field.name for field in fields(Binding)] == ["source", "origin"]
    assert [field.name for field in fields(Assertion)] == [
        "expression",
        "display",
        "origin",
    ]
    assert [field.name for field in fields(ContractOrigins)] == ["default", "entries"]
    assert [field.name for field in fields(ValueContract)] == ["type", "origins"]
    assert [field.name for field in fields(PartialRecordContract)] == ["fields"]
    assert [field.name for field in fields(Parameter)] == [
        "name",
        "contract",
        "required",
        "has_default",
        "default",
        "coerce_strings",
        "origin",
        "default_origin",
    ]


def test_carriers_are_field_frozen_but_preserve_arbitrary_payload_identity() -> None:
    """Freezing carrier fields must not copy or constrain user semantic values."""
    source: dict[str, list[int]] = {"values": [1]}
    default: list[str] = ["fallback"]
    binding = Binding(source)
    parameter = Parameter("x", has_default=True, default=default)

    assert binding.source is source
    assert parameter.default is default
    with pytest.raises(FrozenInstanceError):
        binding.origin = _origin("changed")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        parameter.required = True  # type: ignore[misc]


def test_binding_helpers_normalize_once_and_preserve_existing_identity() -> None:
    """Construction wraps raw values once; consumption unwraps one layer only."""
    raw = {"x": 1}
    wrapped = as_binding(raw)
    inner = Binding("value", _origin("inner"))
    outer = Binding(inner, _origin("outer"))

    assert wrapped == Binding(raw, None)
    assert wrapped.source is raw
    assert as_binding(inner) is inner
    assert binding_source(raw) is raw
    assert binding_source(inner) == "value"
    assert binding_source(outer) is inner


def test_contract_origins_use_a_python_310_safe_canonical_order() -> None:
    """Mixed field/index paths have one total, prefix-first order."""
    paths: tuple[ValuePath, ...] = (
        ("a",),
        ("a", "field"),
        ("a", 0),
        ("b",),
        (0,),
    )
    origins = ContractOrigins(
        entries=tuple((path, _origin(str(index))) for index, path in enumerate(paths))
    )

    assert tuple(path for path, _ in origins.entries) == paths


@pytest.mark.parametrize(
    ("entries", "error", "message"),
    [
        ((((), _origin("root")),), ValueError, "empty"),
        (
            ((("same",), _origin("one")), (("same",), _origin("two"))),
            ValueError,
            "duplicate",
        ),
        (
            ((("z",), _origin("z")), (("a",), _origin("a"))),
            ValueError,
            "sorted",
        ),
        ((((True,), _origin("bool")),), TypeError, "segment"),
    ],
)
def test_contract_origins_reject_ambiguous_entries(
    entries: Any,
    error: type[Exception],
    message: str,
) -> None:
    """Root duplication, duplicate paths, bad order, and bool indices fail loud."""
    with pytest.raises(error, match=message):
        ContractOrigins(entries=entries)


def test_contract_origins_require_immutable_well_typed_structure() -> None:
    """Mutable containers and non-origin values cannot enter the frozen index."""
    with pytest.raises(TypeError, match="tuple"):
        ContractOrigins(entries=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="path"):
        ContractOrigins(
            entries=((["field"], _origin("x")),)  # type: ignore[list-item]
        )
    with pytest.raises(TypeError, match="OriginRef"):
        ContractOrigins(default="root")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OriginRef"):
        ContractOrigins(
            entries=((("field",), "not-an-origin"),)  # type: ignore[arg-type]
        )


def test_contract_origins_select_exact_then_ancestor_then_default() -> None:
    """Nested validation paths inherit the nearest authored contract site."""
    fallback = _origin("fallback")
    payload = _origin("payload")
    items = _origin("items")
    first = _origin("first")
    origins = ContractOrigins(
        default=fallback,
        entries=(
            (("payload",), payload),
            (("payload", "items"), items),
            (("payload", "items", 0), first),
        ),
    )

    assert origins.at(("payload", "items", 0)) is first
    assert origins.at(("payload", "items", 0, "name")) is first
    assert origins.at(("payload", "items", 3)) is items
    assert origins.at(("payload", "other")) is payload
    assert origins.at(("unknown",)) is fallback
    assert origins.at(()) is fallback
    with pytest.raises(TypeError, match="segment"):
        origins.at((True,))


def test_parameter_distinguishes_explicit_null_from_no_default() -> None:
    """Default presence is explicit and has no second textual type authority."""
    value_type = Type(kind=ValueKind.INTEGER, nullable=True)
    contract = ValueContract(value_type)
    absent = Parameter(
        "value",
        contract=contract,
        coerce_strings=True,
    )
    explicit_null = Parameter(
        "value",
        contract=contract,
        has_default=True,
        default=None,
        coerce_strings=True,
    )

    assert absent.has_default is False
    assert explicit_null.has_default is True
    assert absent.contract is contract
    assert explicit_null.contract is contract
    assert explicit_null.contract.type is value_type
    assert absent.required is explicit_null.required is False
    assert absent.coerce_strings is explicit_null.coerce_strings is True
    assert absent != explicit_null
    assert not hasattr(absent, "type_str")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"required": True, "has_default": True},
        {"default": 0},
        {"default_origin": _origin("default")},
    ],
)
def test_parameter_rejects_conflicting_default_states(kwargs: dict[str, Any]) -> None:
    """One flag owns default presence; unused values cannot hide beside it."""
    with pytest.raises(ValueError, match="default"):
        Parameter("value", **kwargs)


def test_value_contract_uses_a_fresh_empty_origin_index() -> None:
    """The semantic Type stays unchanged and provenance defaults separately."""
    value_type = Type.scalar(ValueKind.STRING)
    first = ValueContract(value_type)
    second = ValueContract(value_type)

    assert first.type is value_type
    assert first.origins == ContractOrigins()
    assert first.origins is not second.origins


def test_partial_record_contract_enforces_closed_shape_and_known_fields() -> None:
    """Opaque fields stay present while proven siblings retain exact contracts."""
    origin = _origin("known")
    contract = PartialRecordContract(
        fields=(
            (
                "count",
                ValueContract(
                    Type.scalar(ValueKind.INTEGER),
                    ContractOrigins(default=origin),
                ),
            ),
            ("note", None),
        )
    )

    validate_output_contract(contract, {"count": 1, "note": object()})
    assert contract.names == ("count", "note")
    assert output_contract_field_error(contract, ["count"], "p.output.count") is None
    assert output_contract_field_error(contract, ["note", "anything"], "p") is None
    assert output_contract_field_error(contract, ["absent"], "p.output.absent") == (
        "reference ${p.output.absent} reads unknown field 'absent' on record"
    )
    assert output_contract_field_error(contract, ["count", "x"], "p.output.count.x") == (
        "reference ${p.output.count.x} reads field 'x' on type int, which has no fields"
    )
    assert output_contract_origin(contract, ("count",)) is origin
    assert output_contract_origin(contract, ("note",)) is None

    with pytest.raises(ValueError, match="does not match declared type int"):
        validate_output_contract(contract, {"count": "one", "note": "x"})
    with pytest.raises(ValueError, match="missing required"):
        validate_output_contract(contract, {"count": 1})
    with pytest.raises(ValueError, match="unknown fields"):
        validate_output_contract(
            contract,
            {"count": 1, "note": "x", "extra": True},
        )


def test_output_contract_field_error_rejects_fields_on_fieldless_types() -> None:
    """Scalars and lists have no fields; opaque `object` and `file` stay unchecked."""

    def err(typ: Type, *fields: str) -> str | None:
        return output_contract_field_error(ValueContract(typ), list(fields), "x")

    assert err(Type.scalar(ValueKind.STRING), "a") == (
        "reference ${x} reads field 'a' on type str, which has no fields"
    )
    assert "on type list[str]" in err(Type.scalar(ValueKind.LIST_STRING), "a")
    assert "on type Optional[int]" in err(Type(kind=ValueKind.INTEGER, nullable=True), "a")
    assert "on type None" in err(Type.scalar(ValueKind.NONE), "a")
    assert err(Type.scalar(ValueKind.OBJECT), "a", "b") is None
    assert err(Type.scalar(ValueKind.FILE), "uri") is None
    assert err(Type.scalar(ValueKind.STRING)) is None  # no field read at all


def test_partial_record_contract_rejects_ambiguous_representations() -> None:
    """Duplicate, mutable, and fully typed field sets have one canonical rejection."""
    integer = ValueContract(Type.scalar(ValueKind.INTEGER))
    with pytest.raises(TypeError, match="tuple"):
        PartialRecordContract(fields=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        PartialRecordContract(fields=(("x", integer), ("x", None)))
    with pytest.raises(ValueError, match="fully typed"):
        PartialRecordContract(fields=(("x", integer),))


def test_parameter_maps_every_nested_origin_without_copying_semantics() -> None:
    """Origin rebasing is recursive while semantic payload objects stay identical."""
    value_type = Type.scalar(ValueKind.OBJECT)
    default_value = {"name": "Ada"}
    parameter = Parameter(
        name="profile",
        contract=ValueContract(
            value_type,
            ContractOrigins(
                default=_origin("contract"),
                entries=((("name",), _origin("field")),),
            ),
        ),
        has_default=True,
        default=default_value,
        coerce_strings=True,
        origin=_origin("parameter"),
        default_origin=_origin("default"),
    )

    mapped = parameter.map_origins(lambda origin: _origin(f"mapped:{origin.key}"))

    assert mapped.name == parameter.name
    assert mapped.default is default_value
    assert mapped.contract is not None
    assert mapped.contract.type is value_type
    assert mapped.contract.origins.default == _origin("mapped:contract")
    assert mapped.contract.origins.entries == (
        (("name",), _origin("mapped:field")),
    )
    assert mapped.origin == _origin("mapped:parameter")
    assert mapped.default_origin == _origin("mapped:default")


def test_each_carrier_maps_present_origins_and_skips_none() -> None:
    """A mapper is never called for absent provenance."""
    calls: list[OriginRef] = []

    def mapper(origin: OriginRef) -> OriginRef:
        calls.append(origin)
        return _origin(f"new:{origin.key}")

    present = _origin("present")
    assert Binding("x", present).map_origins(mapper).origin == _origin("new:present")
    assert Assertion("x", "x", present).map_origins(mapper).origin == _origin(
        "new:present"
    )
    assert ContractOrigins(default=present).map_origins(mapper).default == _origin(
        "new:present"
    )
    assert ValueContract(
        Type.scalar(ValueKind.STRING), ContractOrigins(default=present)
    ).map_origins(mapper).origins.default == _origin("new:present")
    partial = PartialRecordContract(
        fields=(
            (
                "known",
                ValueContract(
                    Type.scalar(ValueKind.STRING),
                    ContractOrigins(default=present),
                ),
            ),
            ("opaque", None),
        )
    )
    assert partial.map_origins(mapper).fields[0][1].origins.default == _origin(
        "new:present"
    )

    call_count = len(calls)
    Binding("x").map_origins(mapper)
    Assertion("x", "x").map_origins(mapper)
    ContractOrigins().map_origins(mapper)
    ValueContract(Type.scalar(ValueKind.STRING)).map_origins(mapper)
    Parameter("x").map_origins(mapper)
    assert len(calls) == call_count


def test_origin_mapper_exceptions_propagate() -> None:
    """Rebasing failures must fail loud rather than retain a partial carrier."""

    class MappingError(RuntimeError):
        """Sentinel raised by the test mapper."""

    def fail(_origin_ref: OriginRef) -> OriginRef:
        raise MappingError("cannot rebase")

    with pytest.raises(MappingError, match="cannot rebase"):
        Binding("x", _origin("x")).map_origins(fail)


@pytest.mark.parametrize(
    "carrier",
    [
        Binding("x", _origin("binding")),
        Assertion("x", "x", _origin("assertion")),
        ContractOrigins(default=_origin("contract-default")),
        ContractOrigins(entries=((("field",), _origin("contract-entry")),)),
        Parameter("x", origin=_origin("parameter")),
        Parameter(
            "x",
            has_default=True,
            default_origin=_origin("parameter-default"),
        ),
    ],
)
@pytest.mark.parametrize("invalid", [None, "not-an-origin"])
def test_origin_mapper_rejects_invalid_results_for_present_origins(
    carrier: Any,
    invalid: Any,
) -> None:
    """Every origin slot must reject erasure and non-origin replacement values."""
    with pytest.raises(TypeError, match="mapper must return an OriginRef"):
        carrier.map_origins(lambda _origin_ref: invalid)


def test_nodes_package_exports_the_canonical_contract_vocabulary() -> None:
    """The nodes package is the public owner of the shared semantic contracts."""
    expected = {
        "Assertion": Assertion,
        "Binding": Binding,
        "ContractOrigins": ContractOrigins,
        "OutputContract": OutputContract,
        "PartialRecordContract": PartialRecordContract,
        "Parameter": Parameter,
        "ValueContract": ValueContract,
        "ValuePath": ValuePath,
        "as_binding": as_binding,
        "binding_source": binding_source,
    }

    assert expected.keys() <= set(nodes_api.__all__)
    for name, value in expected.items():
        assert getattr(nodes_api, name) is value
