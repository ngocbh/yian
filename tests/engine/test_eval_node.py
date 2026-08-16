"""The raw ``eval_node`` read/execution boundary.

The evaluator binds from flow-owned wiring, runs the node, and returns exactly one
``Output | Route | Grow | EvaluationFailure``. Durable starts and terminal observers
belong to ``FlowEngine`` rather than this pure evaluation seam.
"""

import pytest

from agent_composer.compile.model import START_ID
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import (
    Grow,
    Node,
    NodeAssertFailed,
    NodeKind,
    Output,
    Route,
    _AssertionStageError,
    _BindingStageError,
)
from agent_composer.nodes.binding import BindingError
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.case import Case, CaseNode
from agent_composer.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
)
from agent_composer.nodes.loop import LoopNode
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
    node = FuncNode("n", lambda inputs: inputs, origin=OriginRef("node:n"))
    node.params = [Parameter(name="x", required=True)]
    node._wiring_src = {}

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "binding.evaluate"
    assert result.failure.trace == (node.origin,)
    assert result.failure.cause_type == "BindingError"


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
    node = FailNode("n", "boom", origin=OriginRef("node:n"))

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "node.execute"
    assert result.failure.trace == (node.origin,)
    assert (result.failure.message, result.failure.cause_type) == (
        "boom",
        "RuntimeError",
    )


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
    node = _NonSpawnerGrowNode("n", origin=OriginRef("node:n"))

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "contract.node_result"
    assert result.failure.trace == (node.origin,)
    assert result.failure.cause_type == "RuntimeError"
    assert "Grow" in result.failure.message and "spawner" in result.failure.message


def test_malformed_result_context_is_an_ir_fault() -> None:
    """Formatting an expected rejection must not absorb malformed node metadata."""
    node = _NonSpawnerGrowNode("n")
    node.kind = None  # type: ignore[assignment]

    with pytest.raises(AttributeError):
        drive(node)


def test_bad_return_type_returns_clear_evaluation_failure():
    result = drive(_BadReturnNode("n"))

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "contract.node_result"
    assert result.failure.trace == ()
    assert result.failure.cause_type == "RuntimeError"
    assert "not a NodeResult" in result.failure.message


def test_post_assert_isolated_from_node_input_mutation():
    pool = VariablePool()
    pool.set(START_ID, {"x": 1})
    node = _MutatingNode("n")
    node.post_asserts = [Assertion("${x} == 1", "${x} == 1")]

    result = drive(node, pool)

    assert result == Output("done")


def test_wrapper_assertion_failure_uses_assertion_stage_and_origin() -> None:
    """Only the wrapper-owned authored assertion gets `assertion.failed`."""
    assertion_origin = OriginRef("assert:n:0")
    node = FuncNode("n", lambda _inputs: "never", origin=OriginRef("node:n"))
    node.pre_asserts = [
        Assertion("1 == 2", "must hold", origin=assertion_origin)
    ]

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "assertion.failed"
    assert result.failure.trace == (assertion_origin,)
    assert result.failure.cause_type == "NodeAssertFailed"
    assert "Traceback (most recent call last)" in (result.failure.traceback or "")
    assert "_AssertionStageError" not in (result.failure.traceback or "")


def test_wrapper_post_assertion_uses_assertion_stage_and_origin() -> None:
    """Post-asserts retain the same wrapper-owned classification as pre-asserts."""
    assertion_origin = OriginRef("assert:n:post:0")
    node = FuncNode("n", lambda _inputs: "value", origin=OriginRef("node:n"))
    node.post_asserts = [
        Assertion("${output} == 'other'", "output must match", origin=assertion_origin)
    ]

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "assertion.failed"
    assert result.failure.trace == (assertion_origin,)
    assert result.failure.cause_type == "NodeAssertFailed"


class _BodyRaisesAssertion(Node):
    """Raise the wrapper's exception type from a node body."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        raise NodeAssertFailed("body chose this exception type")


class _BodyRaisesBinding(Node):
    """Raise the binder's exception type from ordinary node code."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        raise BindingError(
            "body chose this exception type",
            failure_code="contract.input",
            origin=OriginRef("spoofed-binding-origin"),
        )


class _GeneratorRaisesBinding(Node):
    """Raise the binder's exception type while advancing a node generator."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        if False:
            yield None
        raise BindingError(
            "generator chose this exception type",
            failure_code="binding.evaluate",
            origin=OriginRef("spoofed-generator-origin"),
        )


class _OperationOriginNode(Node):
    """Raise in the body while supplying a finer operation origin hook."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, operation_origin: OriginRef | None) -> None:
        super().__init__(node_id, origin=OriginRef(f"node:{node_id}"))
        self.operation_origin = operation_origin

    def _run_impl(self, inputs):
        raise RuntimeError("operation failed")

    def exception_origin(self, exception: Exception) -> OriginRef | None:
        return self.operation_origin


class _OverrideRaisesStageShaped(Node):
    """Bypass the base wrapper and raise an exception resembling a stage marker."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, failure: Exception) -> None:
        super().__init__(node_id, origin=OriginRef(f"node:{node_id}"))
        self.failure = failure

    def _run_impl(self, inputs):
        return Output("unused")

    def run(self, inputs, **caps):
        raise self.failure


class _DeepcopyBomb:
    """Raise one prebuilt fault when a deferred binder copies an item value."""

    def __init__(self, failure: Exception) -> None:
        self.failure = failure

    def __deepcopy__(self, memo):
        raise self.failure


def test_body_raised_assertion_exception_remains_node_execution() -> None:
    """Semantic classification follows the body stage, not exception class."""
    node = _BodyRaisesAssertion("n", origin=OriginRef("node:n"))

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "node.execute"
    assert result.failure.trace == (node.origin,)
    assert result.failure.cause_type == "NodeAssertFailed"


@pytest.mark.parametrize("node", [_BodyRaisesBinding("body"), _GeneratorRaisesBinding("gen")])
def test_body_raised_binding_exception_remains_node_execution(node: Node) -> None:
    """Only evaluator-owned binding operations may retain binder classification."""
    node.origin = OriginRef(f"node:{node.id}")

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "node.execute"
    assert result.failure.trace == (node.origin,)
    assert result.failure.cause_type == "BindingError"


@pytest.mark.parametrize(
    ("operation_origin", "expected_key"),
    [(OriginRef("operation:n"), "operation:n"), (None, "node:n")],
)
def test_node_execution_prefers_operation_origin_then_node_fallback(
    operation_origin: OriginRef | None,
    expected_key: str,
) -> None:
    """Body diagnostics append the selected local origin after outer ancestry."""
    outer = OriginRef("outer:call")
    node = _OperationOriginNode("n", operation_origin)
    node.origin_context = (outer,)

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.trace == (outer, OriginRef(expected_key))


@pytest.mark.parametrize(
    ("failure", "expected_cause_type"),
    [
        (RuntimeError("ordinary override failure"), "RuntimeError"),
        (BindingError("spoofed", failure_code="binding.evaluate"), "BindingError"),
        (
            NodeAssertFailed(
                "spoofed",
                assertion=Assertion("true", "true", origin=OriginRef("spoofed")),
            ),
            "NodeAssertFailed",
        ),
        (
            _BindingStageError(
                BindingError("wrapped", failure_code="binding.evaluate")
            ),
            "BindingError",
        ),
        (
            _AssertionStageError(
                NodeAssertFailed(
                    "wrapped",
                    assertion=Assertion("true", "true"),
                )
            ),
            "NodeAssertFailed",
        ),
    ],
)
def test_run_override_exception_is_node_execution(
    failure: Exception,
    expected_cause_type: str,
) -> None:
    """A custom run override is one undivided node-body stage."""
    node = _OverrideRaisesStageShaped("n", failure)

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "node.execute"
    assert result.failure.trace == (node.origin,)
    assert result.failure.cause_type == expected_cause_type
    assert "_BindingStageError" not in (result.failure.traceback or "")
    assert "_AssertionStageError" not in (result.failure.traceback or "")


@pytest.mark.parametrize(
    "node",
    [
        CallNode("call", flow_id="child"),
        MapNode("map", flow_id="child"),
        LoopNode("loop", flow_id="child", predicate="true", max_iters=1),
    ],
)
def test_unbaked_composite_is_a_setup_fault(node: Node) -> None:
    """A missing compiled child is malformed setup, not expected node rejection."""
    if isinstance(node, MapNode):
        node._wiring_src = {"over": []}
    node.pre_asserts = [Assertion("false", "must not mask malformed setup")]

    with pytest.raises(RuntimeError, match="baked"):
        drive(node)


def test_missing_node_implementation_is_a_setup_fault() -> None:
    """The base placeholder body is malformed executable setup."""
    with pytest.raises(NotImplementedError, match="defines no _run_impl"):
        drive(Node("n"))


def test_case_without_predicate_is_a_setup_fault() -> None:
    """A compiled CASE branch must have its validated predicate."""
    with pytest.raises(ValueError, match="has no `when` expression"):
        drive(CaseNode("case", [Case("branch")]))


def test_case_with_non_string_predicate_is_a_setup_fault() -> None:
    """Malformed Python-authored CASE state must not look like node execution."""
    malformed = Case("branch", when=42)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="`when` must be a string"):
        drive(CaseNode("case", [malformed]))


def test_case_predicate_failure_selects_the_active_ordered_origin() -> None:
    """The failing arm wins even when an earlier ordered predicate was evaluated."""
    first_origin = OriginRef("case/0/when")
    second_origin = OriginRef("case/1/when")
    node = CaseNode(
        "case",
        [
            Case("first", "1 == 2", origin=first_origin),
            Case("second", "1 / 0 > 0", origin=second_origin),
        ],
    )
    node.origin = OriginRef("case/node")

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.cause_type == "ExpressionError"
    assert result.failure.trace == (second_origin,)


def test_source_free_case_predicate_failure_falls_back_to_the_node_origin() -> None:
    """A predicate without its own site retains the common node fallback."""
    node_origin = OriginRef("case/node")
    node = CaseNode("case", [Case("broken", "1 / 0 > 0")])
    node.origin = node_origin

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.trace == (node_origin,)


def test_case_rejects_a_non_origin_predicate_carrier() -> None:
    """Python-authored CASE provenance must stay source-neutral."""
    with pytest.raises(TypeError, match="CASE predicate origin"):
        Case("branch", "true", origin="flow.yaml:12")  # type: ignore[arg-type]


def test_malformed_assertion_carrier_is_an_ir_fault() -> None:
    """Unexpected wrapper-state faults are outside expected flow failure."""
    node = FuncNode("n", lambda _inputs: "never")
    node.pre_asserts = [object()]  # type: ignore[list-item]

    with pytest.raises(AttributeError):
        drive(node)


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
        assert result.failure.message == MAP_OVER_NOT_LIST_FMT.format(
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


def test_map_deferred_expression_keeps_binding_stage() -> None:
    """A per-item expression failure remains evaluator-owned binding work."""
    binding_origin = OriginRef("map:x:binding")
    node = MapNode("m", flow_id="child", child=object())
    node.params = [Parameter("x")]
    node._wiring_src = {
        "over": [{}],
        "x": Binding('${item.missing :? "x is required"}', binding_origin),
    }

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "binding.evaluate"
    assert result.failure.trace == (binding_origin,)


def test_map_deferred_unexpected_binder_fault_escapes() -> None:
    """Deferred binding preserves the ordinary binder's host/setup boundary."""
    failure = RuntimeError("copy implementation failed")
    node = MapNode("m", flow_id="child", child=object())
    node.params = [Parameter("x")]
    node._wiring_src = {
        "over": [_DeepcopyBomb(failure)],
        "x": "${item}",
    }

    with pytest.raises(RuntimeError) as caught:
        drive(node)

    assert caught.value is failure


def test_per_item_wrapper_default_keeps_binding_stage() -> None:
    """The base wrapper's per-item default backstop retains default provenance."""
    default_origin = OriginRef("map:x:default")
    node = MapNode("m", flow_id="child", child=object())
    node.params = [
        Parameter(
            "x",
            contract=ValueContract(Type.scalar(ValueKind.INTEGER)),
            has_default=True,
            default="bad",
            default_origin=default_origin,
        )
    ]
    node._wiring_src = {"over": []}

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.code == "binding.evaluate"
    assert result.failure.trace == (default_origin,)


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
