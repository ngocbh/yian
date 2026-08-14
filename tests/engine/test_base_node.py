"""Unit tests for the pure Node.run/eval_node contract pipeline."""

from agent_composer.events import StreamChunk
from agent_composer.nodes.base import Output, Route
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


def test_node_carries_declared_output_type():
    assert FuncNode("n", lambda _inputs: {}).output_type is None
    output_type = Type.scalar(ValueKind.NUMBER)
    node = FuncNode("n", lambda _inputs: {}, output_type=output_type)

    assert node.output_type == output_type
