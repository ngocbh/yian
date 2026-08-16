"""Conservative static type inference over unified expression trees."""

from __future__ import annotations

import pytest

from agent_composer.expr import infer_expr_type, parse_expr
from agent_composer.typesys.values import Type, ValueKind


def _unexpected_ref(path: str) -> Type | None:
    pytest.fail(f"inference unexpectedly resolved reference {path!r}")


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("1", ValueKind.INTEGER),
        ("1.5", ValueKind.NUMBER),
        ('"text"', ValueKind.STRING),
        ("true", ValueKind.BOOLEAN),
        ("false", ValueKind.BOOLEAN),
        ("null", ValueKind.NONE),
        ("none", ValueKind.NONE),
    ],
)
def test_literal_tokens_have_proven_scalar_types(source: str, kind: ValueKind) -> None:
    assert infer_expr_type(parse_expr(source), _unexpected_ref) == Type.scalar(kind)


def test_exact_reference_uses_injected_dotted_path_type() -> None:
    expected = Type.scalar(ValueKind.STRING)
    paths: list[str] = []

    def ref_type(path: str) -> Type | None:
        paths.append(path)
        return expected

    inferred = infer_expr_type(parse_expr("node#0.output.answer"), ref_type)

    assert inferred is expected
    assert paths == ["node#0.output.answer"]


def test_wrapped_reference_uses_the_same_injected_type() -> None:
    """The grammar's back-compat wrapped-ref token remains a typed reference."""
    expected = Type.scalar(ValueKind.INTEGER)
    paths: list[str] = []

    def ref_type(path: str) -> Type | None:
        paths.append(path)
        return expected

    inferred = infer_expr_type(parse_expr("${producer.output}"), ref_type)

    assert inferred is expected
    assert paths == ["producer.output"]


def test_unknown_exact_reference_stays_unknown() -> None:
    assert infer_expr_type(parse_expr("missing.output"), lambda _path: None) is None


class _ReferenceFailure(RuntimeError):
    pass


def test_reference_callback_exception_propagates() -> None:
    def fail(_path: str) -> Type | None:
        raise _ReferenceFailure("lookup failed")

    with pytest.raises(_ReferenceFailure, match="lookup failed"):
        infer_expr_type(parse_expr("producer.output"), fail)


@pytest.mark.parametrize(
    "source",
    [
        "left == right",
        "item in items",
        "item not in items",
        "not value",
        "left and right",
        "left or right",
    ],
)
def test_boolean_result_roots_are_non_nullable_bool(source: str) -> None:
    inferred = infer_expr_type(parse_expr(source), _unexpected_ref)

    assert inferred == Type.scalar(ValueKind.BOOLEAN)
    assert inferred.nullable is False


@pytest.mark.parametrize(
    "source",
    [
        "left + right",
        "left - right",
        "left * right",
        "left / right",
        "left % right",
        "left ** right",
        "-value",
        "[left, right]",
        "left | right",
        "left :- right",
        'left :? "required"',
        "upper(value)",
    ],
)
def test_unsupported_expression_forms_stay_unknown(source: str) -> None:
    assert infer_expr_type(parse_expr(source), _unexpected_ref) is None


def test_builtin_call_is_not_mistaken_for_a_reference() -> None:
    calls: list[str] = []

    def ref_type(path: str) -> Type | None:
        calls.append(path)
        return Type.scalar(ValueKind.STRING)

    assert infer_expr_type(parse_expr("upper(value)"), ref_type) is None
    assert calls == []
