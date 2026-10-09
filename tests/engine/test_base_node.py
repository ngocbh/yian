"""Unit tests for the pure Node.run/eval_node contract pipeline."""

import pytest

from yian.diagnostics import OriginRef
from yian.events import StreamChunk
from yian.nodes.base import Node, Output, Route
from yian.nodes.contracts import ContractOrigins, ValueContract
from yian.runtime.eval_node import EvaluationFailure
from yian.typesys.pool import VariablePool
from yian.typesys.values import Type, ValueKind
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
    assert (result.failure.message, result.failure.cause_type) == (
        "kaboom",
        "RuntimeError",
    )


def test_policy_exception_returns_typed_evaluation_failure():
    result = drive(ReturnsFailedNode("n"))

    assert isinstance(result, EvaluationFailure)
    assert (result.failure.message, result.failure.cause_type) == (
        "declined",
        "Policy",
    )


def test_branch_node_returns_route_handle():
    assert drive(BranchNode("n", "case_a")) == Route("case_a")


def test_node_does_not_write_pool():
    pool = VariablePool()

    assert drive(FuncNode("n", lambda _inputs: {"output": "x"}), pool) == Output(
        {"output": "x"}
    )
    assert pool.get("n") is None


def test_node_stores_one_canonical_output_contract():
    """The semantic carrier is the node's only output declaration."""
    untyped = FuncNode("untyped", lambda _inputs: {})
    contract_type = Type.scalar(ValueKind.NUMBER)
    contract = ValueContract(
        contract_type,
        ContractOrigins(default=OriginRef("node/output")),
    )

    node = FuncNode("n", lambda _inputs: {}, output_contract=contract)

    assert untyped.output_contract is None
    assert "output_contract" in vars(untyped)
    assert "output_type" not in vars(untyped)
    assert node.output_contract is contract
    assert node.output_contract.type is contract_type
    assert "output_contract" in vars(node)
    assert "output_type" not in vars(node)
    assert not hasattr(Node, "output_type")
    assert not hasattr(node, "output_type")


def test_node_rejects_removed_output_type_constructor_keyword():
    """The deleted compatibility keyword cannot recreate parallel type storage."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'output_type'"):
        FuncNode(
            "n",
            lambda _inputs: {},
            output_type=Type.scalar(ValueKind.NUMBER),
        )


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
    ],
)
def test_node_rejects_invalid_output_contract_inputs(keyword, value, message):
    """The canonical constructor accepts only the exact carrier shape."""
    with pytest.raises(TypeError, match=message):
        FuncNode("n", lambda _inputs: {}, **{keyword: value})
