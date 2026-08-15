"""Unit tests for the effect-node parser arms — human_input + wait."""

from agent_composer.compose.parse.file import HumanInputDescriptor, WaitDescriptor
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.source import SourceCursor
import pytest


def test_parse_human_input_and_wait():
    nodes = parse_nodes(SourceCursor.virtual({
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
        parse_nodes(SourceCursor.virtual({"w": {"kind": "wait", "until": "${input.x}", "inputs": {"a": "1"}}}))


def test_human_input_missing_prompt_loud():
    with pytest.raises(LoadError):
        parse_nodes(SourceCursor.virtual({"h": {"kind": "human_input"}}))
