"""A binding's declared contract: carrier shape, origins, identity, and expansion."""

from __future__ import annotations

import copy
from dataclasses import fields, replace

import pytest

from yian.compile.expand import _rens_internal, build_grow
from yian.compile.manifest import (
    expansion_digest,
    flow_fingerprint,
    flow_manifest,
)
from yian.compile.model import CompiledFlow, FlowOutput
from yian.compile.origins import iter_flow_origins, rewrite_flow_origins
from yian.compose import load_flow
from yian.diagnostics import OriginRef
from yian.nodes.contracts import Binding, ContractOrigins, ValueContract
from yian.run import run_flow
from yian.typesys import type_for
from yian.typesys.types import read_typedefs


_TWO_STEP = """
id: two-step
name: two_step
input:
  x: int
nodes:
  first:
    kind: code
    input:
      x: ${input.x}
    output: int
    code: |
      return inputs["x"] + 1
  second:
    kind: code
    input:
      y: ${first.output}
    output: int
    code: |
      return inputs["y"] * 2
output: ${second.output}
"""

_RECORD_OUTPUT = """
id: record-out
name: record_out
input:
  x: int
nodes:
  first:
    kind: code
    input:
      x: ${input.x}
    output: int
    code: |
      return inputs["x"] + 1
output:
  a: ${first.output}
  b: ${input.x}
"""

_CHILD = """
id: child
name: child
input:
  x: int
nodes:
  inner:
    kind: code
    input:
      x: ${input.x}
    output: int
    code: |
      return inputs["x"] + 1
output: ${inner.output}
"""

_PARENT = """
id: parent
name: parent
input:
  x: int
uses:
  child: child
nodes:
  invoke:
    kind: call
    call: child
    input:
      x: ${input.x}
output: ${invoke.output}
"""


def _int_contract(origin: OriginRef | None = None) -> ValueContract:
    """Return an exact int contract with an optional root origin."""
    return ValueContract(type_for("int", {}), ContractOrigins(default=origin))


def _record_contract() -> ValueContract:
    """Return a record contract, whose Type is unhashable because fields is a dict."""
    registry = read_typedefs({"Payload": {"value": "int"}})
    return ValueContract(type_for("Payload", registry))


def _typed(flow: CompiledFlow, node_id: str, param: str, contract: ValueContract) -> CompiledFlow:
    """Return a deep copy of ``flow`` whose one wiring binding carries ``contract``."""
    typed = copy.deepcopy(flow)
    binding = typed.wiring[node_id][param]
    typed.wiring[node_id][param] = replace(binding, contract=contract)
    return typed


def _typed_output(flow: CompiledFlow, index: int, contract: ValueContract) -> CompiledFlow:
    """Return a deep copy of ``flow`` whose one declared output binding carries ``contract``."""
    typed = copy.deepcopy(flow)
    output = typed.outputs[index]
    typed.outputs[index] = FlowOutput(
        name=output.name,
        binding=replace(output.binding, contract=contract),
    )
    return typed


def _resolver(**children):
    """Return a child resolver serving each keyword's flow text loaded by its id."""
    loaded = {flow_id: load_flow(text) for flow_id, text in children.items()}
    return lambda flow_id, version=None: loaded[flow_id]


def test_binding_fields_end_with_an_optional_contract() -> None:
    """The contract is the third, optional binding field."""
    assert [field.name for field in fields(Binding)] == ["source", "origin", "contract"]
    assert Binding("${x}").contract is None
    contract = _int_contract()
    assert Binding("${x}", contract=contract).contract is contract


def test_binding_rejects_a_non_value_contract() -> None:
    """Only an exact ValueContract may be a binding's declared contract."""
    with pytest.raises(TypeError, match="ValueContract"):
        Binding("${x}", contract=type_for("int", {}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ValueContract"):
        Binding("${x}", contract="int")  # type: ignore[arg-type]


def test_map_origins_keeps_and_rebases_the_contract() -> None:
    """Rebasing maps the binding origin and every contract origin, keeping the type."""
    contract = _int_contract(OriginRef("type-site"))
    binding = Binding("${x}", OriginRef("binding-site"), contract)

    mapped = binding.map_origins(lambda origin: OriginRef(f"mapped:{origin.key}"))

    assert mapped.source == "${x}"
    assert mapped.origin == OriginRef("mapped:binding-site")
    assert mapped.contract is not None
    assert mapped.contract.type is contract.type
    assert mapped.contract.origins.default == OriginRef("mapped:type-site")
    assert Binding("${x}").map_origins(lambda origin: origin).contract is None


def test_iter_origins_yields_binding_then_contract_origins() -> None:
    """A binding's own origin precedes its contract's origins."""
    contract = ValueContract(
        type_for("int", {}),
        ContractOrigins(default=OriginRef("type-site")),
    )
    binding = Binding("${x}", OriginRef("binding-site"), contract)

    assert list(binding.iter_origins()) == [
        OriginRef("binding-site"),
        OriginRef("type-site"),
    ]
    assert list(Binding("${x}", contract=contract).iter_origins()) == [
        OriginRef("type-site"),
    ]


def test_typed_wiring_binding_changes_flow_identity_only_when_typed() -> None:
    """A wiring contract is canonical, and an untyped flow keeps its identity."""
    flow = load_flow(_TWO_STEP).compiled
    original_fingerprint = flow_fingerprint(flow)
    original_topology = expansion_digest(flow)
    untyped_manifest = flow_manifest(flow)
    assert "binding_types" not in untyped_manifest

    typed = _typed(flow, "second", "y", _int_contract(OriginRef("type-site")))

    assert flow_fingerprint(typed) != original_fingerprint
    assert expansion_digest(typed) != original_topology
    binding_types = flow_manifest(typed)["binding_types"]
    assert list(binding_types) == ["second"]
    assert list(binding_types["second"]) == ["y"]
    assert binding_types["second"]["y"]["kind"] == "int"
    assert flow_fingerprint(flow) == original_fingerprint
    assert flow_fingerprint(copy.deepcopy(flow)) == original_fingerprint


def test_binding_contract_origins_stay_outside_identity() -> None:
    """Two typed bindings that differ only in contract origins share identity."""
    flow = load_flow(_TWO_STEP).compiled
    left = _typed(flow, "second", "y", _int_contract(OriginRef("left")))
    right = _typed(flow, "second", "y", _int_contract(OriginRef("right")))

    assert flow_fingerprint(left) == flow_fingerprint(right)
    assert expansion_digest(left) == expansion_digest(right)


def test_typed_value_output_changes_fingerprint() -> None:
    """A value flow's one declared output records its contract type."""
    flow = load_flow(_TWO_STEP).compiled
    assert [output.name for output in flow.outputs] == ["result"]
    assert "type" not in flow_manifest(flow)["outputs"][0]

    typed = _typed_output(flow, 0, _int_contract())

    assert flow_fingerprint(typed) != flow_fingerprint(flow)
    assert flow_manifest(typed)["outputs"][0]["type"]["kind"] == "int"


def test_typed_record_output_field_changes_fingerprint() -> None:
    """A record flow's typed output field records its contract type."""
    flow = load_flow(_RECORD_OUTPUT).compiled
    assert [output.name for output in flow.outputs] == ["a", "b"]

    typed = _typed_output(flow, 1, _int_contract())

    assert flow_fingerprint(typed) != flow_fingerprint(flow)
    outputs = flow_manifest(typed)["outputs"]
    assert "type" not in outputs[0]
    assert outputs[1]["type"]["kind"] == "int"


def test_renamespacing_keeps_the_binding_contract() -> None:
    """Re-namespacing a source or a literal keeps origin and contract."""
    contract = _int_contract(OriginRef("type-site"))
    expression = Binding("${inner.output}", OriginRef("binding-site"), contract)
    literal = Binding(3, OriginRef("binding-site"), contract)

    renamed = _rens_internal(expression, "invoke")
    assert renamed == Binding("${invoke/inner.output}", OriginRef("binding-site"), contract)
    assert renamed.contract is contract
    assert _rens_internal(literal, "invoke").contract is contract


def test_flow_origin_traversal_reaches_the_binding_contract() -> None:
    """Flow-wide origin iteration and rewriting cover a typed binding's contract."""
    flow = load_flow(_TWO_STEP).compiled
    typed = _typed(flow, "second", "y", _int_contract(OriginRef("type-site")))

    assert OriginRef("type-site") in list(iter_flow_origins(typed))

    rewritten = rewrite_flow_origins(typed, lambda origin: OriginRef(f"new:{origin.key}"))

    contract = rewritten.wiring["second"]["y"].contract
    assert contract is not None
    assert contract.origins.default == OriginRef("new:type-site")
    assert typed.wiring["second"]["y"].contract.origins.default == OriginRef("type-site")


def test_call_expansion_keeps_a_typed_child_binding() -> None:
    """A CALL child's typed wiring binding keeps its contract in the grown fragment."""
    parent = load_flow(_PARENT, child_resolver=_resolver(child=_CHILD)).compiled
    call = parent.nodes["invoke"]
    contract = _int_contract()
    binding = call.child.wiring["inner"]["x"]
    call.child.wiring["inner"]["x"] = replace(binding, contract=contract)

    fragment = build_grow(call, {"x": 2})

    assert fragment.wiring["invoke/inner"]["x"].contract is contract


def test_record_typed_binding_is_never_hashed() -> None:
    """An unhashable record contract survives copy, manifest, expansion, and a run."""
    contract = _record_contract()
    with pytest.raises(TypeError):
        hash(contract.type)

    loaded = load_flow(_PARENT, child_resolver=_resolver(child=_CHILD))
    call = loaded.compiled.nodes["invoke"]
    binding = call.child.wiring["inner"]["x"]
    call.child.wiring["inner"]["x"] = replace(binding, contract=contract)
    top = loaded.compiled.wiring["invoke"]["x"]
    loaded.compiled.wiring["invoke"]["x"] = replace(top, contract=contract)

    copied = copy.deepcopy(loaded.compiled)
    assert copied.wiring["invoke"]["x"].contract.type.fields.keys() == {"value"}
    manifest = flow_manifest(loaded.compiled)
    assert manifest["binding_types"]["invoke"]["x"]["kind"] == "object"
    assert flow_fingerprint(copied) == flow_fingerprint(loaded.compiled)

    fragment = build_grow(call, {"x": 2})
    assert fragment.wiring["invoke/inner"]["x"].contract is contract
    expansion_digest(fragment)

    # The binder checks the record contract without hashing it and rejects the int.
    result = run_flow(loaded, {"x": 2})
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "binding.type"
