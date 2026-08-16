"""The raw ``eval_node`` read/execution boundary.

The evaluator binds from flow-owned wiring, runs the node, and returns exactly one
``Output | Route | Grow | EvaluationFailure``. Durable starts and terminal observers
belong to ``FlowEngine`` rather than this pure evaluation seam.
"""

import pytest

from agent_composer.compile.model import START_ID
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Grow, Node, NodeKind, Output, Route
from agent_composer.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
)
from agent_composer.nodes.map import MapNode
from agent_composer.nodes.wait import WaitNode
from agent_composer.runtime.eval_node import EvaluationFailure
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import Type, ValueKind
from tests.engine._fakes import BranchNode, FailNode, FuncNode, drive, stamp_reads
from tests.engine.test_golden_baseline import MAP_OVER_NOT_LIST_FMT


def test_success_returns_output_without_observer_events():
    result = drive(FuncNode("n", lambda _inputs: {"output": "ok"}))

    assert result == Output({"output": "ok"})


class _CommitAsNode(Node):
    """Return one configured Output to isolate commit-target ownership."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, output: Output) -> None:
        super().__init__(node_id)
        self._output = output

    def _run_impl(self, inputs, **caps):
        return self._output


def test_baked_commit_target_remains_engine_owned():
    node = _CommitAsNode("n", Output("v"))
    node.commit_as = "spawner"

    result = drive(node)

    assert result == Output("v")
    assert result.commit_as is None


def test_node_chosen_commit_target_remains_on_output():
    node = _CommitAsNode("n", Output("v", commit_as="chosen"))
    node.commit_as = "spawner"

    result = drive(node)

    assert result == Output("v", commit_as="chosen")


def test_binding_failure_returns_diagnostic_evaluation_failure():
    node = FuncNode("n", lambda inputs: inputs)
    node.params = [Parameter(name="x", required=True)]
    node._wiring_src = {}

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.error_type == "BindingError"


def test_binding_carrier_resolves_at_the_node_read_boundary():
    """Ordinary parameter binding evaluates the carrier source without mutating it."""
    pool = VariablePool()
    pool.set(START_ID, {"value": "resolved"})
    origin = OriginRef("binding")
    wrapped = Binding("${input.value}", origin)
    node = FuncNode("n", lambda inputs: inputs["x"], reads={"x": wrapped})

    result = drive(node, pool)

    assert result == Output("resolved")
    assert node._wiring_src["x"] is wrapped
    assert wrapped.origin is origin


def test_binding_carrier_unwraps_exactly_one_layer():
    """A nested carrier remains an opaque user payload after the outer IR layer."""
    pool = VariablePool()
    pool.set(START_ID, {"value": "must-not-resolve"})
    inner = Binding("${input.value}", OriginRef("inner"))
    outer = Binding(inner, OriginRef("outer"))
    node = FuncNode("n", lambda inputs: inputs["x"], reads={"x": outer})

    result = drive(node, pool)

    assert isinstance(result, Output)
    assert result.value == inner


def test_map_over_accepts_a_binding_carrier():
    """MAP's reserved iteration source uses the ordinary binding read seam."""
    node = MapNode("m", flow_id="child", child=object())
    node._wiring_src = {
        "over": Binding(["first", "second"], OriginRef("map-over"))
    }

    result = drive(node)

    assert result == Grow(seed=[{}, {}])


def test_wait_until_accepts_a_binding_carrier():
    """WAIT's dedicated resolver accepts the carrier shape it bypasses elsewhere."""
    node = WaitNode("wait")
    node._wiring_src = {
        "until": Binding("2000-01-01T00:00:00+00:00", OriginRef("wait-until"))
    }

    result = drive(node)

    assert result == Output(None)


def test_node_raise_returns_evaluation_failure():
    result = drive(FailNode("n", "boom"))

    assert isinstance(result, EvaluationFailure)
    assert (result.error, result.error_type) == ("boom", "RuntimeError")


def test_routing_returns_route_without_output_value():
    result = drive(BranchNode("n", "case_a"))

    assert result == Route("case_a")
    assert not hasattr(result, "output")


class _NonSpawnerGrowNode(Node):
    """Return Grow without the required spawner trait."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        return Grow(seed={})


class _BadReturnNode(Node):
    """Return a value outside the closed node-result sum."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        return {"not": "a NodeResult"}


class _MutatingNode(Node):
    """Mutate its input record to test the wrapper's post-assert snapshot."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        stamp_reads(self, {"x": "${input.x}"})

    def _run_impl(self, inputs):
        inputs["x"] = 999
        return Output("done")


def test_non_spawner_grow_returns_clear_evaluation_failure():
    result = drive(_NonSpawnerGrowNode("n"))

    assert isinstance(result, EvaluationFailure)
    assert result.error_type == "RuntimeError"
    assert "Grow" in result.error and "spawner" in result.error


def test_bad_return_type_returns_clear_evaluation_failure():
    result = drive(_BadReturnNode("n"))

    assert isinstance(result, EvaluationFailure)
    assert result.error_type == "RuntimeError"
    assert "not a NodeResult" in result.error


def test_post_assert_isolated_from_node_input_mutation():
    pool = VariablePool()
    pool.set(START_ID, {"x": 1})
    node = _MutatingNode("n")
    node.post_asserts = [Assertion("${x} == 1", "${x} == 1")]

    result = drive(node, pool)

    assert result == Output("done")


def test_map_over_not_a_list_message_is_byte_identical():
    """Raw and carrier-wrapped sources produce the same origin-neutral error text."""
    pool = VariablePool()
    pool.set(START_ID, {"bad": "notalist"})
    raw_source = "${input.bad}"
    for source in (raw_source, Binding(raw_source, OriginRef("map-over"))):
        node = MapNode("m", flow_id="f")
        node._wiring_src = {"over": source}

        result = drive(node, pool)

        assert isinstance(result, EvaluationFailure)
        assert result.error == MAP_OVER_NOT_LIST_FMT.format(
            id="m",
            src=raw_source,
        )


def test_map_deferred_item_binding_keeps_input_contract_stage() -> None:
    """MAP's binder capability does not become a node-body failure."""
    contract_origin = OriginRef("map:x:contract")
    node = MapNode("m", flow_id="child", child=object())
    node.params = [
        Parameter(
            "x",
            contract=ValueContract(
                Type.scalar(ValueKind.INTEGER),
                ContractOrigins(default=contract_origin),
            ),
        )
    ]
    node._wiring_src = {
        "over": ["bad"],
        "x": Binding("${item}", OriginRef("map:x:binding")),
    }

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "contract.input"
    assert result.failure.trace == (contract_origin,)


@pytest.mark.parametrize(
    ("node", "field"),
    [
        (MapNode("map", flow_id="child", child=object()), "over"),
        (WaitNode("wait"), "until"),
    ],
)
def test_missing_reserved_wiring_is_an_ir_fault(node: Node, field: str) -> None:
    """A validated compiled graph must contain every trait-declared reserved key."""
    node._wiring_src = {}

    with pytest.raises(KeyError, match=field):
        drive(node)


def test_malformed_parameter_carrier_is_an_ir_fault() -> None:
    """Malformed compiled setup escapes instead of becoming a flow failure."""
    node = FuncNode("n", lambda inputs: inputs)
    node.params = (Parameter("x"),)  # type: ignore[assignment]

    with pytest.raises(TypeError, match="exact list"):
        drive(node)


@pytest.mark.parametrize(
    ("node", "field", "value"),
    [
        (MapNode("map", flow_id="child", child=object()), "over", "not-a-list"),
        (WaitNode("wait"), "until", "not-an-iso-date"),
    ],
)
def test_reserved_input_rejection_is_an_input_contract_failure(
    node: Node,
    field: str,
    value: str,
) -> None:
    """Expected reserved-value rejection keeps its authored binding origin."""
    origin = OriginRef(f"{node.id}:{field}")
    node._wiring_src = {field: Binding(value, origin)}

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "contract.input"
    assert result.failure.trace == (origin,)
