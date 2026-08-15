"""Unit tests for the pure Node.run/eval_node contract pipeline."""

import pytest

from agent_composer.diagnostics import OriginRef
from agent_composer.events import StreamChunk
from agent_composer.nodes.base import Output, Route
from agent_composer.nodes.contracts import ContractOrigins, ValueContract
from agent_composer.runtime.eval_node import EvaluationFailure
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import Type, ValueKind
from tests.engine._fakes import (
    BranchNode,
    FailNode,
    FuncNode,
    ReturnsFailedNode,
    StreamNode,
    drive,
)


class _ValueContractSubclass(ValueContract):
    """Near-canonical fixture proving nodes require the exact IR carrier."""


def test_result_node_returns_output():
    assert drive(FuncNode("n", lambda _inputs: {"output": "ok"})) == Output(
        {"output": "ok"}
    )


def test_streaming_node_forwards_chunks_and_returns_output():
    chunks = []

    result = drive(
        StreamNode("n", ["a", "b", "c"]),
        emit_stream=chunks.append,
    )

    assert [chunk.chunk for chunk in chunks] == ["a", "b", "c"]
    assert all(isinstance(chunk, StreamChunk) for chunk in chunks)
    assert result == Output({"output": "abc"})


def test_raising_node_returns_evaluation_failure():
    result = drive(FailNode("n", "kaboom"))

    assert isinstance(result, EvaluationFailure)
    assert (result.error, result.error_type) == ("kaboom", "RuntimeError")


def test_policy_exception_returns_typed_evaluation_failure():
    result = drive(ReturnsFailedNode("n"))

    assert isinstance(result, EvaluationFailure)
    assert (result.error, result.error_type) == ("declined", "Policy")


def test_branch_node_returns_route_handle():
    assert drive(BranchNode("n", "case_a")) == Route("case_a")


def test_node_does_not_write_pool():
    pool = VariablePool()

    assert drive(FuncNode("n", lambda _inputs: {"output": "x"}), pool) == Output(
        {"output": "x"}
    )
    assert pool.get("n") is None


def test_node_stores_one_canonical_output_contract():
    """The semantic carrier is stored; ``output_type`` is only a read view."""
    untyped = FuncNode("untyped", lambda _inputs: {})
    output_type = Type.scalar(ValueKind.NUMBER)
    contract = ValueContract(
        output_type,
        ContractOrigins(default=OriginRef("node/output")),
    )

    node = FuncNode("n", lambda _inputs: {}, output_contract=contract)

    assert untyped.output_contract is None
    assert untyped.output_type is None
    assert "output_contract" in vars(untyped)
    assert "output_type" not in vars(untyped)
    assert node.output_contract is contract
    assert node.output_type is output_type
    assert "output_contract" in vars(node)
    assert "output_type" not in vars(node)


def test_legacy_output_type_constructor_normalizes_to_a_contract():
    """The temporary constructor bridge cannot recreate parallel type storage."""
    output_type = Type.scalar(ValueKind.NUMBER)

    node = FuncNode("n", lambda _inputs: {}, output_type=output_type)

    assert type(node.output_contract) is ValueContract
    assert node.output_contract.type is output_type
    assert node.output_contract.origins == ContractOrigins()
    assert node.output_type is output_type


def test_legacy_output_type_setter_replaces_only_the_contract_type():
    """A front-end type rewrite retains root and nested contract provenance."""
    original_type = Type.scalar(ValueKind.INTEGER)
    replacement_type = Type.scalar(ValueKind.NUMBER)
    origins = ContractOrigins(
        default=OriginRef("node/output"),
        entries=((("answer",), OriginRef("node/output/answer")),),
    )
    original = ValueContract(original_type, origins)
    node = FuncNode("n", lambda _inputs: {}, output_contract=original)

    node.output_type = replacement_type

    assert type(node.output_contract) is ValueContract
    assert node.output_contract is not original
    assert node.output_contract.type is replacement_type
    assert node.output_contract.origins is origins
    assert node.output_type is replacement_type


def test_legacy_output_type_none_clears_the_contract():
    """An explicitly absent type means there is no output contract or origin."""
    node = FuncNode(
        "n",
        lambda _inputs: {},
        output_contract=ValueContract(Type.scalar(ValueKind.INTEGER)),
    )

    node.output_type = None

    assert node.output_contract is None
    assert node.output_type is None


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        (
            "output_contract",
            Type.scalar(ValueKind.INTEGER),
            "output_contract.*ValueContract",
        ),
        (
            "output_contract",
            _ValueContractSubclass(Type.scalar(ValueKind.INTEGER)),
            "output_contract.*ValueContract",
        ),
        ("output_type", "int", "output_type.*Type"),
    ],
)
def test_node_rejects_invalid_output_contract_inputs(keyword, value, message):
    """Neither canonical nor compatibility constructors accept the other's shape."""
    with pytest.raises(TypeError, match=message):
        FuncNode("n", lambda _inputs: {}, **{keyword: value})


def test_node_rejects_conflicting_canonical_and_legacy_output_inputs():
    """Two supplied declarations cannot compete for the single contract authority."""
    with pytest.raises(ValueError, match="output_contract.*output_type"):
        FuncNode(
            "n",
            lambda _inputs: {},
            output_contract=ValueContract(Type.scalar(ValueKind.INTEGER)),
            output_type=Type.scalar(ValueKind.NUMBER),
        )


def test_legacy_output_type_setter_rejects_a_non_type():
    """The compatibility property accepts only a ``Type`` or ``None``."""
    node = FuncNode("n", lambda _inputs: {})

    with pytest.raises(TypeError, match="output_type.*Type"):
        node.output_type = ValueContract(Type.scalar(ValueKind.INTEGER))
