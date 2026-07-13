"""The base `Node.run` template method fires pre/post asserts (record-only).

Step 2 of the node-owned-asserts refactor: `Node.run` becomes a concrete
template method that fires `pre_asserts` against the bound record, calls the
kind's `_run_impl`, then fires `post_asserts` against `{**record, output}`.
Firing reads ONLY the bound record (never the pool) — preserving node purity.

These tests exercise the wrapper directly (NOT through `eval_node`, which keeps
firing asserts itself until a later step). We define a tiny concrete node that
implements `_run_impl` (the FuncNode fake still defines `run`, which would
shadow the wrapper), so the wrapper's own firing is under test.
"""

from typing import Any

import pytest

from agent_composer.nodes.base import (
    Node,
    NodeAssertFailed,
    NodeKind,
    Output,
)


class _ImplNode(Node):
    """A concrete node that implements `_run_impl` (not `run`) so the base
    `Node.run` template method's assert firing is exercised. `_run_impl` records
    whether it ran, so a pre-assert failure can be shown to short-circuit it."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, value: Any, **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.value = value
        self.ran = False

    def _run_impl(self, inputs: dict, **caps: Any) -> Output:
        self.ran = True
        return Output(value=self.value)


def test_run_returns_impl_output_when_asserts_hold():
    """Both pre and post hold -> `run()` returns the `Output` `_run_impl` produced."""
    node = _ImplNode("n", value={"output": 42})
    node.pre_asserts = ["${x} > 0"]
    node.post_asserts = ["${output.output} == 42"]

    result = node.run({"x": 5})

    assert isinstance(result, Output)
    assert result.value == {"output": 42}
    assert node.ran is True


def test_pre_assert_failure_raises_before_impl_runs():
    """A violated pre-assert raises `NodeAssertFailed` before `_run_impl` runs."""
    node = _ImplNode("n", value={"output": 1})
    node.pre_asserts = ["${x} > 100"]

    with pytest.raises(NodeAssertFailed) as excinfo:
        node.run({"x": 5})

    assert node.ran is False  # short-circuited before the body
    assert "pre-assert failed" in str(excinfo.value)
    assert "'n'" in str(excinfo.value)


def test_post_assert_failure_on_output_raises():
    """A violated post-assert (reading `${output}`) raises `NodeAssertFailed`."""
    node = _ImplNode("n", value={"output": 7})
    node.post_asserts = ["${output.output} > 100"]

    with pytest.raises(NodeAssertFailed) as excinfo:
        node.run({})

    assert node.ran is True  # body ran; post fired on its Output
    assert "post-assert failed" in str(excinfo.value)
    assert "'n'" in str(excinfo.value)


def test_assert_texts_used_in_message():
    """The failure message shows the author's original text via `assert_texts`."""
    node = _ImplNode("n", value=None)
    node.pre_asserts = ["${__assert_calc} > 100"]
    node.assert_texts = {"${__assert_calc} > 100": "calc.output > 100"}

    with pytest.raises(NodeAssertFailed) as excinfo:
        node.run({"__assert_calc": 5})

    assert "calc.output > 100" in str(excinfo.value)


# --- After the Step 3 rename (each kind's `run` -> `_run_impl`), the INHERITED base
# `Node.run` wrapper is the entry point for real/fake kinds: calling `node.run(...)`
# (with no asserts set) must delegate to the kind's `_run_impl` and return its result
# unchanged, once per result category (Output / Route / Pause / Grow). These prove the
# wrapper correctly finds the renamed `_run_impl` — they FAIL before the rename because
# the kinds still shadow `run` (so `run` is the kind body, not the wrapper), and the base
# `_run_impl` stub raises NotImplementedError.


def test_wrapper_delegates_to_real_code_node_run_impl():
    """A real CodeNode (leaf -> Output): its `run` is the INHERITED base wrapper (it no
    longer shadows it), and the wrapper delegates to the kind's `_run_impl`."""
    from agent_composer.nodes.code import CodeNode

    node = CodeNode("c", code="return inputs['v']", code_line=1)
    # After the rename the kind defines no `run`, so `run` resolves to the base wrapper.
    assert type(node).run is Node.run
    result = node.run({"v": "hi"})

    assert isinstance(result, Output)
    assert result.value == "hi"


def test_wrapper_delegates_for_route_kind():
    """A CASE fake (-> Route): `run` is the inherited wrapper and returns the Route from `_run_impl`."""
    from agent_composer.nodes.base import Route

    from tests.engine._fakes import BranchNode

    node = BranchNode("b", handle="yes")
    assert type(node).run is Node.run
    result = node.run({})

    assert isinstance(result, Route)
    assert result.handle == "yes"


def test_wrapper_delegates_for_pause_kind():
    """A HUMAN_INPUT fake (-> Pause): `run` is the inherited wrapper and returns the Pause."""
    from agent_composer.nodes.base import Pause

    from tests.engine._fakes import PauseNode

    node = PauseNode("p", reason="needs-input")
    assert type(node).run is Node.run
    result = node.run({})

    assert isinstance(result, Pause)
    assert result.reason == "needs-input"


def test_wrapper_delegates_for_grow_kind():
    """A spawner (CALL -> Grow): `run` is the inherited wrapper and returns the Grow from `_run_impl`."""
    from agent_composer.nodes.base import Grow
    from agent_composer.nodes.call import CallNode

    from tests.engine.test_expand import _child_flow

    node = CallNode("r", flow_id="c", child=_child_flow(), child_inputs=[])
    assert type(node).run is Node.run
    result = node.run({"topic": "ACME"})

    assert isinstance(result, Grow)
    assert result.seed == {"topic": "ACME"}


# --- eval_node NO LONGER fires asserts itself: the `run` wrapper owns firing. eval_node drives
# `node.run` (not `_run_impl`), so a node whose `run` is overridden to bypass firing produces NO
# NodeFailed even with a would-be-failing post-assert — proving eval_node does not fire on its own.


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

    def _run_impl(self, inputs: dict, **caps: Any) -> Output:
        return Output(value=self.value)

    def run(self, inputs: dict, **caps: Any) -> Output:  # bypass the wrapper's firing
        return Output(value=self.value)


class _FireRunNode(_BypassRunNode):
    """The positive control: identical node WITHOUT the `run` override — so it inherits the base
    wrapper, which DOES fire the failing post-assert. Proves the would-fail assert really fails."""

    run = Node.run  # inherit the firing template method


def test_eval_node_does_not_fire_when_run_bypasses_firing():
    """Behavioral: a would-be-failing post-assert produces NO NodeFailed when `run` bypasses
    firing — eval_node delegates to `run()` and does not fire independently."""
    from agent_composer.events import NodeFailed, NodeSucceeded

    from tests.engine._fakes import drive, stamp_reads

    node = _BypassRunNode("n", value="v")
    node.post_asserts = ["${output} == 'never'"]  # would FAIL if fired
    stamp_reads(node, {})

    events = list(drive(node))
    assert any(isinstance(e, NodeSucceeded) for e in events)
    assert not any(isinstance(e, NodeFailed) for e in events)


def test_eval_node_fires_via_wrapper_positive_control():
    """Positive control: the SAME node WITHOUT the `run` override inherits the wrapper, which
    fires the failing post-assert -> NodeFailed. Confirms the assert genuinely fails."""
    from agent_composer.events import NodeFailed

    from tests.engine._fakes import drive, stamp_reads

    node = _FireRunNode("n", value="v")
    node.post_asserts = ["${output} == 'never'"]
    stamp_reads(node, {})

    events = list(drive(node))
    failed = [e for e in events if isinstance(e, NodeFailed)]
    assert failed and "post-assert" in failed[0].error
