"""Unit tests for the effect-node parser arms — human_input + wait."""

from agent_composer.compose.parser import HumanInputDescriptor, WaitDescriptor
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.stages.parse import parse_nodes
from agent_composer.compose.errors import LoadError
import pytest


def test_parse_human_input_and_wait():
    nodes = parse_nodes(NodeReader.spanless({
        "approve": {"kind": "human_input", "prompt": "ok? ${a}",
                    "inputs": {"a": "${p.output}"}, "outputs": "Approval"},
        "settle": {"kind": "wait", "until": "${input.settle_at}"},
    }))
    assert isinstance(nodes["approve"], HumanInputDescriptor)
    assert nodes["approve"].prompt == "ok? ${a}"
    assert isinstance(nodes["settle"], WaitDescriptor)
    assert nodes["settle"].until == "${input.settle_at}"


def test_wait_rejects_inputs_field():
    with pytest.raises(LoadError):
        parse_nodes(NodeReader.spanless({"w": {"kind": "wait", "until": "${input.x}", "inputs": {"a": "1"}}}))


def test_human_input_missing_prompt_loud():
    with pytest.raises(LoadError):
        parse_nodes(NodeReader.spanless({"h": {"kind": "human_input"}}))
