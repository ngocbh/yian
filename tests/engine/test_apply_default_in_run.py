"""The base `Node.run` template fills canonical parameter defaults before pre-asserts.

`Node.run` calls `apply_defaults` before checking pre-asserts, so a pre-assert
may read an input omitted by the caller but supplied by its `Parameter`.

`f(x=None)` semantics: a caller who BINDS `null` to an input SHADOWS the default
(the key is present with value None) — the wrapper must NOT re-fill it. Only a
truly OMITTED input (key absent from the record) gets its default.

These tests exercise the wrapper directly rather than through `eval_node`.
"""

from typing import Any

import pytest

from yian.nodes.base import Node, NodeKind, Output
from yian.nodes.binding import BindingError
from yian.nodes.contracts import Assertion, Parameter, ValueContract
from yian.typesys import Type, ValueKind


INTEGER = ValueContract(Type.scalar(ValueKind.INTEGER))
NUMBER = ValueContract(Type.scalar(ValueKind.NUMBER))
NULLABLE_INTEGER = ValueContract(Type(kind=ValueKind.INTEGER, nullable=True))


class _ParameterSubclass(Parameter):
    """Near-canonical fixture proving direct runs reject carrier subclasses."""


class _ImplNode(Node):
    """A concrete node implementing `forward` so the base `Node.run` template
    method (default-fill + assert firing) is exercised. `forward` records the
    input record it saw so a test can assert what the wrapper filled."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, **kw: Any) -> None:
        super().__init__(node_id, **kw)
        self.seen: dict | None = None

    def forward(self, inputs: dict, **caps: Any) -> Output:
        self.seen = dict(inputs)
        return Output(value=inputs)


def test_omitted_input_defaulted_before_pre_assert():
    """An input OMITTED from the bound record gets its declared default filled by
    the wrapper — and it is VISIBLE to a pre-assert reading that input (defaults
    precede pre-asserts). The stored integer already satisfies its contract."""
    node = _ImplNode("n")
    node.params = [
        Parameter(
            name="x",
            contract=INTEGER,
            has_default=True,
            default=5,
        )
    ]
    node.pre_asserts = [Assertion("x > 0", "x > 0")]

    result = node.run({})  # x omitted

    assert isinstance(result, Output)
    assert node.seen == {"x": 5}
    assert result.value == {"x": 5}


def test_bound_null_shadows_default():
    """An input BOUND to null (key present, value None) is NOT re-filled — the
    caller's explicit null SHADOWS the default (`f(x=None)`)."""
    node = _ImplNode("n")
    node.params = [
        Parameter(
            name="x",
            contract=INTEGER,
            has_default=True,
            default=5,
        )
    ]

    result = node.run({"x": None})  # x present, explicitly null

    assert node.seen == {"x": None}
    assert result.value == {"x": None}


def test_explicit_null_default_fills_when_omitted():
    """`has_default=True` makes a nullable `None` default semantically present."""
    node = _ImplNode("n")
    node.params = [
        Parameter(
            name="x",
            contract=NULLABLE_INTEGER,
            has_default=True,
            default=None,
        )
    ]

    result = node.run({})  # x omitted

    assert node.seen == {"x": None}
    assert result.value == {"x": None}


def test_compatible_stored_default_is_not_replaced_by_validation():
    """Validation may accept numeric widening without replacing the stored payload."""
    node = _ImplNode("n")
    parameter = Parameter(
        name="x",
        contract=NUMBER,
        has_default=True,
        default=30,
    )
    node.params = [parameter]

    result = node.run({})

    assert type(parameter.default) is int
    assert type(result.value["x"]) is int


def test_incompatible_stored_default_fails_without_string_coercion():
    """Caller-string coercion policy never rewrites an already-stored default."""
    node = _ImplNode("n")
    parameter = Parameter(
        name="x",
        contract=INTEGER,
        has_default=True,
        default="30",
        coerce_strings=True,
    )
    node.params = [parameter]

    with pytest.raises(BindingError) as caught:
        node.run({})

    assert caught.value.failure_code == "binding.evaluate"
    assert parameter.default == "30"
    assert node.seen is None


def test_apply_default_idempotent_no_double_fill():
    """Running the wrapper's default-fill does not corrupt an already-present key:
    a START-like node whose `forward` also runs apply_defaults would default the
    same key twice, but apply_defaults only fills OMITTED keys, so the second call
    is a no-op — the default is filled exactly once, not corrupted."""
    from yian.nodes.binding import apply_defaults

    node = _ImplNode("n")
    node.params = [
        Parameter(
            name="x",
            contract=INTEGER,
            has_default=True,
            default=5,
        )
    ]

    # First fill (what the wrapper does), then a second fill (what a START forward
    # would redundantly do): the value stays 5, not double-processed.
    once = apply_defaults(node.params, {})
    twice = apply_defaults(node.params, once)
    assert once == {"x": 5}
    assert twice == {"x": 5}

    # And end-to-end through the wrapper the default is filled exactly once.
    result = node.run({})
    assert result.value == {"x": 5}


@pytest.mark.parametrize("params", [(), (Parameter("x"),)])
def test_node_run_rejects_noncanonical_parameter_containers(params):
    node = _ImplNode("n")
    node.params = params  # type: ignore[assignment]

    with pytest.raises(TypeError, match="exact list"):
        node.run({})


@pytest.mark.parametrize("params", [["x"], [_ParameterSubclass("x")]])
def test_node_run_rejects_noncanonical_parameter_elements(params):
    node = _ImplNode("n")
    node.params = params  # type: ignore[assignment]

    with pytest.raises(TypeError, match="Parameter values"):
        node.run({})
