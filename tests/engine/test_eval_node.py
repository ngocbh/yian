"""The raw ``eval_node`` read/execution boundary.

The evaluator binds from flow-owned wiring, runs the node, and returns exactly one
``Output | Route | Grow | EvaluationFailure``. Durable starts and terminal observers
belong to ``FlowEngine`` rather than this pure evaluation seam.
"""

from agent_composer.compile.model import START_ID
from agent_composer.nodes.base import Grow, Node, NodeKind, Output, Route
from agent_composer.nodes.binding import ParamDecl
from agent_composer.nodes.map import MapNode
from agent_composer.runtime.eval_node import EvaluationFailure
from agent_composer.typesys.pool import VariablePool
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
    node.params = [ParamDecl(name="x", required=True)]
    node._wiring_src = {}

    result = drive(node)

    assert isinstance(result, EvaluationFailure)
    assert result.error_type == "BindingError"


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
    node.post_asserts = ["${x} == 1"]

    result = drive(node, pool)

    assert result == Output("done")


def test_map_over_not_a_list_message_is_byte_identical():
    pool = VariablePool()
    pool.set(START_ID, {"bad": "notalist"})
    node = MapNode("m", flow_id="f")
    node._wiring_src = {"over": "${input.bad}"}

    result = drive(node, pool)

    assert isinstance(result, EvaluationFailure)
    assert result.error == MAP_OVER_NOT_LIST_FMT.format(
        id="m",
        src="${input.bad}",
    )
