"""The failure funnel resolves a symbolic locator to a concrete SourceRange.

Dual-path: the funnel keeps stamping the symbolic `SourceSpan` locator AND, alongside it,
attaches a concrete ``SourceRange`` to ``EvaluationFailure``. For an input/binding failure it
keys `node.field_ranges.get((locator.kind, locator.key))` — kind-BLIND (it keys on the locator's
`(kind, key)`, never on `node.kind`), and `(kind, key)`-tagged so an input never collides with a
like-named field. For an assert it is located BY POSITION: `run()` raises `NodeAssertFailed` with
`range = pre/post_assert_ranges[i]` and the funnel carries that range through. The builder stamps
these in a later step; here the test stamps them directly.
"""

from agent_composer.events import SourceRange
from agent_composer.nodes.binding import ParamDecl
from agent_composer.nodes.contracts import Assertion
from agent_composer.runtime.eval_node import EvaluationFailure
from tests.engine._fakes import FuncNode, drive


def _drive(node, pool=None):
    return drive(node, pool)


def _failure(result):
    assert isinstance(result, EvaluationFailure)
    return result


def test_assert_failure_carries_range():
    # A failing pre-assert flows to EvaluationFailure with an `assert` locator whose
    # key IS the assert-expr string. The concrete range is located BY POSITION — the failing
    # assert's own `pre_assert_ranges[i]` — and carried ALONGSIDE the unchanged symbolic locator.
    expr = "1 == 2"  # a constant, always-false assert -> pre-assert fails
    rng = SourceRange(3, 5, 3, 11)
    node = FuncNode("n", lambda i: {"output": "ok"})
    node.pre_asserts = [Assertion(expr, expr)]
    node.pre_assert_ranges = [rng]  # positional, aligned 1:1 with pre_asserts
    nf = _failure(_drive(node))
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
    node.field_ranges = {("input", "x"): rng}  # keyed (kind, key) — the input namespace
    nf = _failure(_drive(node))
    assert nf.locator is not None
    assert nf.locator.node == "n" and nf.locator.kind == "input" and nf.locator.key == "x"
    assert nf.range == rng
