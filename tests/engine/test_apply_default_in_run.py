"""The base `Node.run` template method fills declared defaults BEFORE pre-asserts.

Step 4 of the node-owned-asserts refactor: `Node.run` runs `apply_defaults`
(from `typesys.seeding`) as its FIRST step — before the pre-asserts — so a
pre-assert may read an input the caller omitted but the node declares a default
for. Defaults MUST precede pre-asserts (a pre-assert may read an omitted-then-
defaulted input).

`f(x=None)` semantics: a caller who BINDS `null` to an input SHADOWS the default
(the key is present with value None) — the wrapper must NOT re-fill it. Only a
truly OMITTED input (key absent from the record) gets its default.

These tests exercise the wrapper directly (NOT through `eval_node`). We build
nodes with ParamDecls that carry a `default` by hand: the loader only ever mints
`default`-bearing ParamDecls on START (from the flow-level `input:` block); a body
node's `input:` block is pure wiring, so its params carry no default.
"""

from typing import Any

from agent_composer.nodes.base import Node, NodeKind, Output
from agent_composer.nodes.binding import ParamDecl
from agent_composer.nodes.contracts import Assertion


class _ImplNode(Node):
    """A concrete node implementing `_run_impl` so the base `Node.run` template
    method (default-fill + assert firing) is exercised. `_run_impl` records the
    input record it saw so a test can assert what the wrapper filled."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.seen: dict | None = None

    def _run_impl(self, inputs: dict, **caps: Any) -> Output:
        self.seen = dict(inputs)
        return Output(value=inputs)


def test_omitted_input_defaulted_before_pre_assert():
    """An input OMITTED from the bound record gets its declared default filled by
    the wrapper — and it is VISIBLE to a pre-assert reading that input (defaults
    precede pre-asserts). Uses an int default already an int (no coercion needed),
    so it passes without Step 5's type_str threading."""
    node = _ImplNode("n")
    node.params = [ParamDecl(name="x", type_str="int", required=False, default=5)]
    node.pre_asserts = [Assertion("${x} > 0", "${x} > 0")]

    result = node.run({})  # x omitted

    assert isinstance(result, Output)
    assert node.seen == {"x": 5}
    assert result.value == {"x": 5}


def test_bound_null_shadows_default():
    """An input BOUND to null (key present, value None) is NOT re-filled — the
    caller's explicit null SHADOWS the default (`f(x=None)`)."""
    node = _ImplNode("n")
    node.params = [ParamDecl(name="x", type_str="int", required=False, default=5)]

    result = node.run({"x": None})  # x present, explicitly null

    assert node.seen == {"x": None}
    assert result.value == {"x": None}


def test_string_default_coerced_to_declared_type():
    """A ParamDecl carrying BOTH `type_str="int"` and a string `default="30"`,
    OMITTED -> the wrapper's default-fill coerces it to the INTEGER 30 (via
    `apply_defaults` -> `coerce_param`, which reads `type_str`), not the string "30".

    This pins the wrapper's coerce-on-fill capability directly. Note the loader
    never mints a BODY-node ParamDecl carrying a default (body `input:` blocks are
    pure wiring, no `TYPE = default` syntax — that lives only on the flow-level
    `input:` block, i.e. START's params), and START's params deliberately carry
    `default` WITHOUT `type_str` (`start/node.py`: run() is the sole type authority,
    so `bind_params` must not type-check first). So the type_str+default pairing is
    exercised here against the wrapper as a unit, not via a loader path."""
    node = _ImplNode("n")
    node.params = [ParamDecl(name="x", type_str="int", required=False, default="30")]

    result = node.run({})  # x omitted

    assert node.seen == {"x": 30}
    assert result.value == {"x": 30}


def test_apply_default_idempotent_no_double_fill():
    """Running the wrapper's default-fill does not corrupt an already-present key:
    a START-like node whose `_run_impl` also runs apply_defaults would default the
    same key twice, but apply_defaults only fills OMITTED keys, so the second call
    is a no-op — the default is filled exactly once, not corrupted."""
    from agent_composer.typesys.seeding import apply_defaults

    node = _ImplNode("n")
    node.params = [ParamDecl(name="x", type_str="int", required=False, default=5)]

    # First fill (what the wrapper does), then a second fill (what a START _run_impl
    # would redundantly do): the value stays 5, not double-processed.
    once = apply_defaults(node.params, {})
    twice = apply_defaults(node.params, once)
    assert once == {"x": 5}
    assert twice == {"x": 5}

    # And end-to-end through the wrapper the default is filled exactly once.
    result = node.run({})
    assert result.value == {"x": 5}
