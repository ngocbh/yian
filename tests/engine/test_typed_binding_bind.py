"""The binder validates a typed binding's value against its declared type.

A typed binding checks the resolved value, converting only its file and object
representation, and fails the consuming node with ``binding.type``; the receiving
parameter's own contract still runs afterwards.
"""

import dataclasses

import pytest

from yian.compose import load_flow
from yian.diagnostics import OriginRef
from yian.nodes.base import Node
from yian.nodes.binding import BindingError, bind_params
from yian.nodes.contracts import (
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
    ValueTemplate,
)
from yian.run import run_flow
from yian.runtime.eval_node import EvaluationFailure, eval_node
from yian.typesys import Type, ValueKind
from yian.typesys.pool import VariablePool


INTEGER = Type.scalar(ValueKind.INTEGER)
NUMBER = Type.scalar(ValueKind.NUMBER)
OPTIONAL_INTEGER = Type(kind=ValueKind.INTEGER, nullable=True)


def _pool(**outputs):
    """Return a pool holding each keyword as that node's output."""
    pool = VariablePool()
    for node_id, value in outputs.items():
        pool.set(node_id, value)
    return pool


def _bind(source, contract, pool, *, param=None, item=None):
    """Bind one parameter (``qty`` by default) to a typed binding of ``source``."""
    parameter = param if param is not None else Parameter("qty")
    return bind_params(
        [parameter],
        {parameter.name: Binding(source, contract=contract)},
        pool,
        item=item,
    )


def test_matching_value_is_delivered_unchanged():
    record = _bind("${a.output}", ValueContract(NUMBER), _pool(a=30))

    assert record == {"qty": 30}
    assert type(record["qty"]) is int


def test_mismatch_fails_with_binding_type_at_the_contract_origin():
    contract_origin = OriginRef("contract:qty")
    binding_origin = OriginRef("binding:qty")
    contract = ValueContract(INTEGER, ContractOrigins(default=contract_origin))

    with pytest.raises(BindingError) as caught:
        bind_params(
            [Parameter("qty")],
            {"qty": Binding("${a.output}", binding_origin, contract)},
            _pool(a="many"),
        )

    assert caught.value.failure_code == "binding.type"
    assert str(caught.value).startswith("binding 'qty':")
    assert caught.value.origin == contract_origin


def test_nested_mismatch_uses_the_nearest_contract_origin():
    root_origin = OriginRef("contract:root")
    field_origin = OriginRef("contract:n")
    record_type = Type(
        kind=ValueKind.OBJECT,
        fields={"n": INTEGER},
        required=frozenset({"n"}),
    )
    contract = ValueContract(
        record_type,
        ContractOrigins(default=root_origin, entries=((("n",), field_origin),)),
    )

    with pytest.raises(BindingError) as caught:
        _bind("${a.output}", contract, _pool(a={"n": "x"}))

    assert caught.value.failure_code == "binding.type"
    assert caught.value.origin == field_origin


def test_mismatch_falls_back_to_the_binding_origin():
    binding_origin = OriginRef("binding:qty")

    with pytest.raises(BindingError) as caught:
        bind_params(
            [Parameter("qty")],
            {"qty": Binding("${a.output}", binding_origin, ValueContract(INTEGER))},
            _pool(a="many"),
        )

    assert caught.value.failure_code == "binding.type"
    assert caught.value.origin == binding_origin


def test_none_is_rejected_for_int_and_accepted_for_optional_int():
    with pytest.raises(BindingError) as caught:
        _bind("${a.output}", ValueContract(INTEGER), _pool(a=None))
    assert caught.value.failure_code == "binding.type"

    assert _bind("${a.output}", ValueContract(OPTIONAL_INTEGER), _pool(a=None)) == {
        "qty": None
    }


def test_string_is_rejected_before_parameter_coercion():
    parameter = Parameter("qty", contract=ValueContract(INTEGER), coerce_strings=True)

    with pytest.raises(BindingError) as caught:
        _bind("${a.output}", ValueContract(INTEGER), _pool(a="30"), param=parameter)

    assert caught.value.failure_code == "binding.type"


def test_parameter_contract_still_runs_after_a_passing_binding_check():
    parameter = Parameter("qty", contract=ValueContract(INTEGER))

    with pytest.raises(BindingError) as caught:
        _bind(
            "${a.output}", ValueContract(OPTIONAL_INTEGER), _pool(a=None),
            param=parameter,
        )

    assert caught.value.failure_code == "contract.input"


def test_map_item_scope_is_checked_per_item():
    contract = ValueContract(INTEGER)

    assert _bind("${item}", contract, _pool(), item=3) == {"qty": 3}
    with pytest.raises(BindingError) as caught:
        _bind("${item}", contract, _pool(), item="x")
    assert caught.value.failure_code == "binding.type"


def test_typed_binding_checks_its_evaluated_value_template():
    record_type = Type(
        kind=ValueKind.OBJECT,
        fields={"n": INTEGER, "label": Type.scalar(ValueKind.STRING)},
        required=frozenset({"n", "label"}),
    )
    template = ValueTemplate({"n": "${a.output}", "label": "n=${a.output}"})

    assert _bind(template, ValueContract(record_type), _pool(a=2)) == {
        "qty": {"n": 2, "label": "n=2"}
    }
    with pytest.raises(BindingError) as caught:
        _bind(template, ValueContract(record_type), _pool(a="two"))
    assert caught.value.failure_code == "binding.type"


def test_typed_binding_checks_a_non_string_literal():
    with pytest.raises(BindingError) as caught:
        _bind(30, ValueContract(Type.scalar(ValueKind.STRING)), _pool())
    assert caught.value.failure_code == "binding.type"

    assert _bind(30, ValueContract(INTEGER), _pool()) == {"qty": 30}


def test_binding_type_failure_keeps_the_route_without_usable_inputs():
    origin = OriginRef("contract:bad")
    node = Node("node")
    node.params = [Parameter("ok"), Parameter("bad")]
    wiring = {
        "ok": {"safe": [1]},
        "bad": Binding("wrong", contract=ValueContract(INTEGER, ContractOrigins(default=origin))),
    }

    evaluated = eval_node(
        node, None, _pool(), wiring_override=wiring, capture_bindings=True,
    )

    assert isinstance(evaluated.outcome, EvaluationFailure)
    assert evaluated.outcome.failure.code == "binding.type"
    assert evaluated.outcome.failure.trace == (origin,)
    assert evaluated.bindings.bound_inputs is None
    assert evaluated.bindings.route.names == ("ok", "bad")


def test_untyped_binding_and_raw_sources_are_unchanged():
    pool = _pool(a="many")

    assert bind_params(
        [Parameter("qty")], {"qty": Binding("${a.output}")}, pool
    ) == {"qty": "many"}
    assert bind_params([Parameter("qty")], {"qty": "${a.output}"}, pool) == {
        "qty": "many"
    }


_FLOW = """\
id: typed
name: typed
nodes:
  produce:
    kind: code
    code: |
      return "thirty"
    output: str
  consume:
    kind: code
    code: |
      return inputs["qty"]
    input:
      qty: ${produce.output}
    output: str
output: ${consume.output}
"""


def _type_wiring(compiled, node_id, name, value_type):
    binding = compiled.wiring[node_id][name]
    compiled.wiring[node_id][name] = dataclasses.replace(
        binding, contract=ValueContract(value_type)
    )


def test_run_fails_the_consumer_with_binding_type():
    loaded = load_flow(_FLOW)
    _type_wiring(loaded.compiled, "consume", "qty", INTEGER)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "consume"


_MAP_FLOW = """\
id: typed-map
name: typed_map
nodes:
  outer:
    kind: map
    over: [item]
    input:
      value: ${item}
    nodes:
      work:
        kind: code
        code: |
          return inputs["value"]
        input:
          value: ${input.value}
        output: str
    output: ${work.output}
output: ${outer.output}
"""


def test_map_run_fails_an_item_with_binding_type():
    loaded = load_flow(_MAP_FLOW)
    _type_wiring(loaded.compiled, "outer", "value", INTEGER)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "outer"


def test_run_fails_a_typed_flow_output_with_binding_type():
    loaded = load_flow(_FLOW)
    compiled = loaded.compiled
    _type_wiring(compiled, compiled.end_id, "result", INTEGER)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == compiled.end_id
