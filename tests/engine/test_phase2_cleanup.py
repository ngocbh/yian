import importlib

import pytest


def test_update_variables_command_is_gone():
    cmds = importlib.import_module("agent_composer.suspension.commands")
    assert not hasattr(cmds, "UpdateVariablesCommand")
    assert not hasattr(cmds, "VariableUpdate")
    assert hasattr(cmds, "AbortCommand") and hasattr(cmds, "DeliverAnswerCommand")


def test_channels_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer.suspension.channels")


def test_effect_nodes_drop_scratch_keys():
    from agent_composer.nodes.human_input.node import HumanInputNode
    from agent_composer.nodes.wait.node import WaitNode

    hi = HumanInputNode("h", prompt="?")
    assert not hasattr(hi, "answer_key")
    w = WaitNode("w", is_timed=True)
    assert not hasattr(w, "release_key") and not hasattr(w, "payload_key")


def test_agent_continuation_packages_and_builders_are_gone():
    agent = importlib.import_module("agent_composer.nodes.agent")
    expand = importlib.import_module("agent_composer.compile.expand")
    modes = importlib.import_module("agent_composer.nodes.agent.modes.tool_calling")
    suspension = importlib.import_module("agent_composer.suspension")
    descriptor = importlib.import_module("agent_composer.compose.nodes.agent")
    base = importlib.import_module("agent_composer.nodes.base")
    engine = importlib.import_module("agent_composer.runtime.engine")

    assert not hasattr(agent, "AgentEntry")
    assert not hasattr(agent, "Fresh")
    assert not hasattr(agent, "Resume")
    assert not hasattr(expand, "agent_segment_subgraph")
    assert not hasattr(expand, "clone_continuation_pair")
    assert not hasattr(expand, "ask_resume_edge_id")
    assert not hasattr(modes, "agent_step")
    assert not hasattr(suspension, "AgentSegment")
    assert not hasattr(base.Node, "grow_restamps_self")
    assert not hasattr(engine.FlowEngine, "_apply_origin_commit_as")
    assert "controls" not in descriptor.AgentDescriptor.__dataclass_fields__
    with pytest.raises(ValueError, match="cannot build graph growth"):
        expand.build_grow(agent.AgentNode("agent"), {})
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer.nodes.agent.controls")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer._growth")
