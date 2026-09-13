"""The base `Node.run` template method fires pre/post asserts (record-only).

``Node.run`` is a concrete template method that fires ``pre_asserts`` against
the bound record, calls the
kind's `forward`, then fires `post_asserts` against `{**record, output}`.
Firing reads ONLY the bound record (never the pool) — preserving node purity.

These tests exercise the wrapper directly. We define a tiny concrete node that
implements `forward` (the FuncNode fake still defines `run`, which would
shadow the wrapper), so the wrapper's own firing is under test.
"""

from typing import Any

import pytest

from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import (
    BackwardContext,
    BackwardResult,
    Node,
    NodeAssertFailed,
    NodeKind,
    Output,
)
from agent_composer.nodes.contracts import Assertion


class _ImplNode(Node):
    """A concrete node that implements `forward` (not `run`) so the base
    `Node.run` template method's assert firing is exercised. `forward` records
    whether it ran, so a pre-assert failure can be shown to short-circuit it."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, value: Any, **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.value = value
        self.ran = False

    def forward(self, inputs: dict, **caps: Any) -> Output:
        self.ran = True
        return Output(value=self.value)


def test_default_backward_stops_feedback_with_empty_channels():
    """Behavioral nodes have no automatic input or behavior feedback rule."""
    node = _ImplNode("n", value=None)

    assert node.backward(BackwardContext({}), ("unused",)) == BackwardResult()


def test_runtime_does_not_drive_backward():
    """Forward execution remains the runtime's only node-body call."""
    from tests.engine._fakes import drive

    class _BackwardTrapNode(_ImplNode):
        def backward(self, *args: Any, **caps: Any) -> None:
            raise AssertionError("backward must remain dormant")

    assert drive(_BackwardTrapNode("n", value="ok")) == Output("ok")


def test_run_returns_forward_output_when_asserts_hold():
    """Both pre and post hold -> `run()` returns the `Output` `forward` produced."""
    node = _ImplNode("n", value={"output": 42})
    node.pre_asserts = [Assertion("${x} > 0", "x is positive")]
    node.post_asserts = [Assertion("${output.output} == 42", "output is 42")]

    result = node.run({"x": 5})

    assert isinstance(result, Output)
    assert result.value == {"output": 42}
    assert node.ran is True


def test_pre_assert_failure_raises_before_forward_runs():
    """A violated pre-assert raises `NodeAssertFailed` before `forward` runs."""
    node = _ImplNode("n", value={"output": 1})
    assertion = Assertion(
        "${x} > 100",
        "x is greater than 100",
        origin=OriginRef("assert:n:0"),
    )
    node.pre_asserts = [assertion]

    with pytest.raises(NodeAssertFailed) as excinfo:
        node.run({"x": 5})

    assert node.ran is False  # short-circuited before the body
    assert excinfo.value.assertion is assertion
    assert "pre-assert failed" in str(excinfo.value)
    assert "'n'" in str(excinfo.value)


def test_post_assert_failure_on_output_raises():
    """A violated post-assert (reading `${output}`) raises `NodeAssertFailed`."""
    node = _ImplNode("n", value={"output": 7})
    node.post_asserts = [Assertion("${output.output} > 100", "output is greater than 100")]

    with pytest.raises(NodeAssertFailed) as excinfo:
        node.run({})

    assert node.ran is True  # body ran; post fired on its Output
    assert "post-assert failed" in str(excinfo.value)
    assert "'n'" in str(excinfo.value)


def test_assertion_evaluates_expression_and_reports_display():
    """Display text is presentation-only; it is neither parsed nor evaluated."""
    node = _ImplNode("n", value=None)
    assertion = Assertion(
        "${__assert_calc} > 100",
        "authored text that is not an expression",
    )
    node.pre_asserts = [assertion]

    with pytest.raises(NodeAssertFailed) as excinfo:
        node.run({"__assert_calc": 5})

    assert "authored text that is not an expression" in str(excinfo.value)
    assert excinfo.value.assertion is assertion


def test_assertion_display_is_not_evaluated_when_expression_holds():
    """A deliberately invalid display cannot affect a valid executable expression."""
    node = _ImplNode("n", value=None)
    node.pre_asserts = [Assertion("${x} > 0", "{{ definitely not executable")]

    assert node.run({"x": 1}) == Output(None)


def test_duplicate_assertions_retain_first_occurrence_origin():
    """Equal text retains list order and the first authored semantic origin."""
    node = _ImplNode("n", value=None)
    first = Assertion("1 == 2", "must hold", OriginRef("assert:first"))
    second = Assertion("1 == 2", "must hold", OriginRef("assert:second"))
    node.pre_asserts = [first, second]

    with pytest.raises(NodeAssertFailed) as caught:
        node.run({})

    assert caught.value.assertion is first


# The inherited `Node.run` wrapper is the entry point for real and fake kinds. With no
# assertions, it delegates to the kind's `forward` and returns each result category
# (Output / Route / Grow) unchanged.


def test_wrapper_delegates_to_real_code_node_forward():
    """A real CodeNode (leaf -> Output): its `run` is the INHERITED base wrapper (it no
    longer shadows it), and the wrapper delegates to the kind's `forward`."""
    from agent_composer.nodes.code import CodeNode

    node = CodeNode("c", code="return inputs['v']")
    # The kind defines no `run`, so method resolution selects the base wrapper.
    assert type(node).run is Node.run
    result = node.run({"v": "hi"})

    assert isinstance(result, Output)
    assert result.value == "hi"


def test_wrapper_delegates_for_route_kind():
    """A CASE fake (-> Route): `run` is the inherited wrapper and returns the Route from `forward`."""
    from agent_composer.nodes.base import Route

    from tests.engine._fakes import BranchNode

    node = BranchNode("b", handle="yes")
    assert type(node).run is Node.run
    result = node.run({})

    assert isinstance(result, Route)
    assert result.handle == "yes"


def test_wrapper_delegates_for_grow_kind():
    """A spawner (CALL -> Grow): `run` is the inherited wrapper and returns the Grow from `forward`."""
    from agent_composer.nodes.base import Grow
    from agent_composer.nodes.call import CallNode

    from tests.engine.test_expand import _child_flow

    node = CallNode("r", flow_id="c", child=_child_flow(), child_inputs=[])
    assert type(node).run is Node.run
    result = node.run({"topic": "ACME"})

    assert isinstance(result, Grow)
    assert result.seed == {"topic": "ACME"}


# --- eval_node NO LONGER fires asserts itself: the `run` wrapper owns firing. eval_node drives
# `node.run` (not `forward`), so a node whose `run` is overridden to bypass firing returns
# Output even with a would-be-failing post-assert — proving eval_node does not fire on its own.


def test_eval_node_no_longer_has_first_failing_assert():
    """Structural: the pool-fallback firing function is gone from eval_node — asserts are the
    wrapper's job now."""
    import agent_composer.runtime.eval_node as en

    assert not hasattr(en, "_first_failing_assert")


class _BypassRunNode(Node):
    """A CODE-style node whose `run` bypasses the wrapper's firing entirely (returns Output
    directly). Its declared post-assert WOULD fail if anyone fired it — but nothing does, since
    `run` skips the template method. Drives eval_node to prove the seam delegates firing to `run`."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, value: Any, **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.value = value

    def forward(self, inputs: dict, **caps: Any) -> Output:
        return Output(value=self.value)

    def run(self, inputs: dict, **caps: Any) -> Output:  # bypass the wrapper's firing
        return Output(value=self.value)


class _FireRunNode(_BypassRunNode):
    """The positive control: identical node WITHOUT the `run` override — so it inherits the base
    wrapper, which DOES fire the failing post-assert. Proves the would-fail assert really fails."""

    run = Node.run  # inherit the firing template method


def test_eval_node_does_not_fire_when_run_bypasses_firing():
    """A bypassed wrapper returns Output without independently firing the assert."""

    from tests.engine._fakes import drive, stamp_reads

    node = _BypassRunNode("n", value="v")
    node.post_asserts = [
        Assertion("${output} == 'never'", "output must equal never")
    ]  # would FAIL if fired
    stamp_reads(node, {})

    result = drive(node)
    assert result == Output("v")


def test_eval_node_fires_via_wrapper_positive_control():
    """The inherited wrapper converts its failed assert to EvaluationFailure."""
    from agent_composer.runtime.eval_node import EvaluationFailure

    from tests.engine._fakes import drive, stamp_reads

    node = _FireRunNode("n", value="v")
    node.post_asserts = [Assertion("${output} == 'never'", "output must equal never")]
    stamp_reads(node, {})

    result = drive(node)
    assert isinstance(result, EvaluationFailure)
    assert "post-assert" in result.failure.message
