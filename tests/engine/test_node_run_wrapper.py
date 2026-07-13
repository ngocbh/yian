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
