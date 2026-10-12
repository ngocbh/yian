import ast
from dataclasses import fields
import importlib
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest


def _pause_references(package_name: str) -> list[str]:
    """Return source-level references to ``Pause`` under one shipped node package."""
    package = importlib.import_module(package_name)
    root = Path(package.__file__).parent
    references: list[str] = []
    for source in sorted(root.rglob("*.py")):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            references_pause = (
                isinstance(node, ast.Name) and node.id == "Pause"
            ) or (
                isinstance(node, ast.Attribute) and node.attr == "Pause"
            ) or (
                isinstance(node, (ast.Import, ast.ImportFrom))
                and any(alias.name == "Pause" for alias in node.names)
            )
            if references_pause:
                references.append(f"{source.relative_to(root)}:{node.lineno}")
    return references


@pytest.mark.parametrize(
    "module_name",
    [
        "yian.suspension",
        "yian.suspension.channels",
        "yian.suspension.checkpoint",
        "yian.suspension.commands",
        "yian.suspension.expansions",
        "yian.suspension.pause",
        "yian.runtime.state_manager",
    ],
)
def test_checkpoint_and_suspension_modules_are_gone(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_effect_nodes_drop_scratch_keys():
    from yian.nodes.human_input.node import HumanInputNode
    from yian.nodes.wait.node import WaitNode

    hi = HumanInputNode("h", prompt="?")
    assert not hasattr(hi, "answer_key")
    w = WaitNode("w")
    assert not hasattr(w, "release_key") and not hasattr(w, "payload_key")


def test_wait_event_mode_and_the_watchdog_are_gone():
    from yian.nodes.wait.node import WaitNode

    wait = WaitNode("wait")
    assert not hasattr(wait, "is_timed")
    assert not hasattr(wait, "event_spec")
    assert not hasattr(wait, "poll")
    with pytest.raises(TypeError):
        WaitNode("wait", event_spec={"topic": "alerts"})
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("yian.runtime.watchdog")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("yian.nodes.code.watchdog")


@pytest.mark.parametrize(
    "package_name",
    [
        "yian.nodes.agent",
        "yian.nodes.human_input",
        "yian.nodes.wait",
    ],
)
def test_shipped_blocking_node_packages_cannot_construct_pause(package_name):
    assert _pause_references(package_name) == []


def test_shipped_blocking_node_bodies_are_typed_to_output():
    from yian.nodes.agent import AgentNode
    from yian.nodes.agent.loop import run_agent
    from yian.nodes.base import Output
    from yian.nodes.human_input import HumanInputNode
    from yian.nodes.wait import WaitNode

    assert get_type_hints(AgentNode.forward)["return"] is Output
    assert get_type_hints(HumanInputNode.forward)["return"] is Output
    assert get_type_hints(WaitNode.forward)["return"] is Output
    assert get_type_hints(run_agent)["return"] is Output


@pytest.mark.parametrize("field", ["entry", "controls"])
def test_retired_agent_authoring_fields_are_rejected(field):
    from yian.compose import LoadError, load_flow
    from yian.nodes.agent import AgentNode

    assert field not in inspect.signature(AgentNode).parameters
    with pytest.raises(TypeError):
        AgentNode("agent", **{field: "retired"})

    text = f"""
id: retired-agent-{field}
name: retired_agent_{field}
input: {{}}
nodes:
  agent:
    kind: agent
    prompt: hi
    output: str
    {field}: retired
output: ${{agent.output}}
"""
    with pytest.raises(LoadError, match=rf"field '{field}' is not allowed"):
        load_flow(text)


@pytest.mark.parametrize("field", ["event", "event_spec", "poll", "is_timed"])
def test_retired_wait_authoring_fields_are_rejected(field):
    from yian.compose import LoadError, load_flow

    text = f"""
id: retired-wait-{field}
name: retired_wait_{field}
input: {{}}
nodes:
  wait:
    kind: wait
    until: 2000-01-01
    {field}: retired
output: ${{wait.output}}
"""
    with pytest.raises(LoadError, match=rf"field '{field}' is not allowed"):
        load_flow(text)


def test_final_log_only_runtime_and_public_surface():
    import yian
    import yian.events as events
    import yian.nodes.base as base
    import yian.run as run
    from yian.runtime import FlowEngine

    for name in ("PauseRequested", "RunPaused", "RunAborted", "NodeStreamEvent"):
        assert not hasattr(events, name)
    assert not hasattr(base, "Pause")

    for method in ("snapshot", "restore", "resume", "request_abort"):
        assert not hasattr(FlowEngine, method)
    assert not hasattr(yian, "FlowEngine")
    assert "FlowEngine" not in yian.__all__

    assert not hasattr(run, "resume_command")
    assert list(inspect.signature(run.run_flow).parameters) == [
        "flow",
        "inputs",
        "log",
        "run_id",
        "num_workers",
        "on_event",
        "on_stream",
        "llm_config",
        "memory",
        "learnable",
        "tools",
        "skills",
        "steering",
    ]
    assert list(inspect.signature(run.resume_flow).parameters) == [
        "flow",
        "log",
        "num_workers",
        "on_event",
        "on_stream",
        "memory",
        "tools",
        "skills",
        "steering",
        "api_key",
    ]
    assert [field.name for field in fields(run.RunResult)] == [
        "input",
        "status",
        "run_id",
        "log",
        "output",
        "failure",
        "events",
        "learning",
        "backward_outcome",
        "_backward_handle",
    ]


def test_agent_continuation_packages_and_builders_are_gone():
    agent = importlib.import_module("yian.nodes.agent")
    expand = importlib.import_module("yian.compile.expand")
    loop = importlib.import_module("yian.nodes.agent.loop")
    descriptor = importlib.import_module("yian.compose.nodes.agent")
    base = importlib.import_module("yian.nodes.base")
    engine = importlib.import_module("yian.runtime.engine")

    assert not hasattr(agent, "AgentEntry")
    assert not hasattr(agent, "Fresh")
    assert not hasattr(agent, "Resume")
    assert not hasattr(expand, "agent_segment_subgraph")
    assert not hasattr(expand, "clone_continuation_pair")
    assert not hasattr(expand, "ask_resume_edge_id")
    assert not hasattr(loop, "agent_step")
    assert not hasattr(base.Node, "grow_restamps_self")
    assert not hasattr(engine.FlowEngine, "_apply_origin_commit_as")
    assert "entry" not in descriptor.AgentDescriptor.__dataclass_fields__
    assert "controls" not in descriptor.AgentDescriptor.__dataclass_fields__
    with pytest.raises(ValueError, match="cannot build graph growth"):
        expand.build_grow(agent.AgentNode("agent"), {})
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("yian.nodes.agent.controls")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("yian._growth")
