"""The failure funnel resolves a symbolic locator KEY to a concrete SourceRange.

Dual-path: the funnel keeps stamping the symbolic `SourceSpan` locator AND, alongside
it, looks up `node.field_ranges.get(locator.key)` to attach a concrete `SourceRange` to
the emitted `NodeFailed`. The lookup is kind-BLIND — it keys only on the symbolic locator
key (an assert-expr string, an input name), never on `node.kind`. The builder stamps
`field_ranges` in a later step; here the test stamps it directly.
"""

from agent_composer.events import NodeFailed, SourceRange
from agent_composer.nodes.binding import ParamDecl
from tests.engine._fakes import FuncNode, drive


def _drive(node, pool=None):
    return list(drive(node, pool))


def _node_failed(evs):
    fs = [e for e in evs if isinstance(e, NodeFailed)]
    return fs[-1] if fs else None


def test_assert_failure_carries_range():
    # A failing pre-assert flows to a synthesized NodeFailed with an `assert` locator whose
    # key IS the assert-expr string. With that key recorded in `field_ranges`, the funnel
    # must attach the concrete range ALONGSIDE the unchanged symbolic locator.
    expr = "1 == 2"  # a constant, always-false assert -> pre-assert fails
    rng = SourceRange(3, 5, 3, 11)
    node = FuncNode("n", lambda i: {"output": "ok"})
    node.pre_asserts = [expr]
    node.field_ranges = {expr: rng}
    nf = _node_failed(_drive(node))
    assert nf is not None
    # symbolic locator unchanged (dual-path — we ADD range, do not replace the locator)
    assert nf.locator is not None
    assert nf.locator.node == "n" and nf.locator.kind == "assert" and nf.locator.key == expr
    # concrete range resolved from field_ranges[key]
    assert nf.range == rng


def test_binding_failure_carries_range():
    # A required input with no wiring edge raises BindingError, which stamps a node-less
    # `input` locator (key == param name); the funnel fills the node id AND resolves the
    # concrete range from field_ranges[param]. Same generic key->range path as asserts —
    # this proves the kind-blind funnel already covers input keys (no per-kind branch).
    rng = SourceRange(7, 3, 7, 20)
    node = FuncNode("n", lambda i: i)
    node.params = [ParamDecl(name="x", required=True)]
    node._wiring_src = {}  # x omitted -> BindingError through the funnel
    node.field_ranges = {"x": rng}
    nf = _node_failed(_drive(node))
    assert nf is not None
    assert nf.locator is not None
    assert nf.locator.node == "n" and nf.locator.kind == "input" and nf.locator.key == "x"
    assert nf.range == rng
