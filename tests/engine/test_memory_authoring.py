"""Authored flow and AGENT memory-policy semantics."""

from __future__ import annotations

import copy
import json

import pytest

from agent_composer.compile.manifest import expansion_digest, flow_fingerprint, flow_manifest
from agent_composer.compile.model import CompiledFlow
from agent_composer.compose import LoadError, load_flow
from agent_composer.diagnostics import OriginRef, SourceRange
from agent_composer.memory import MemoryPolicySpec, MemorySnapshotEnvelope
from agent_composer.nodes.agent import AgentNode


_PRIVATE_CONFIG_CASES = (
    {"nested": {"clientSecret": "private"}},
    {"nested": {"secretKey": "private"}},
    {"nested": {"PASSWORD": "private"}},
    {"nested": {"passphrase": "private"}},
    {"nested": {"passwd": "private"}},
    {"nested": {"pwd": "private"}},
    {"nested": {"access-token": "private"}},
    {"nested": {"refreshToken": "private"}},
    {"nested": {"session_key": "private"}},
    {"nested": {"subscriptionKey": "private"}},
    {"nested": {"authToken": "private"}},
    {"nested": {"bearer-key": "private"}},
    {"nested": {"api-key": "private"}},
    {"nested": {"apiKey": "private"}},
    {"nested": {"backend_token": "private"}},
    {"nested": {"Authorization": "private"}},
    {"nested": {"private-key": "private"}},
    {"nested": {"connectionString": "private"}},
    {"nested": {"credentials": "private"}},
    {"nested": {"webhookURL": "private"}},
    {"nested": {"webhookSignature": "private"}},
    {"webhook": {"signature": "private"}},
    {"nested": {"tenant_id": "private"}},
    {"nested": {"orgTenantId": "private"}},
    {"nested": {"pythonImport": "package.policy"}},
    {"nested": {"modulePath": "package.policy"}},
    {"nested": {"class-path": "package.Policy"}},
    {"backend": {"path": "/private/backend"}},
    {"backend": {"connection": "private-connection"}},
    {"storage": {"location": "/private/storage"}},
    {"database": {"dsn": "private-dsn"}},
    {"persistence": {"uri": "private://state"}},
)


def _full_flow(policy: str) -> str:
    """Return one full source with ``policy`` inserted at the flow level."""
    return (
        "id: memory-root\n"
        "name: memory_root\n"
        f"{policy}"
        "nodes:\n"
        "  echo:\n"
        "    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input: {topic: hello}\n"
        "    output: str\n"
        "output: ${echo.output}\n"
    )


@pytest.mark.parametrize(
    ("policy", "expected_config", "line"),
    [
        ("memory_policy: reflective\n", {}, 3),
        (
            "memory_policy:\n"
            "  name: reflective\n"
            "  config:\n"
            "    max_items: 20\n"
            "    nested: [true, null, 3.5]\n",
            {"max_items": 20, "nested": [True, None, 3.5]},
            4,
        ),
    ],
    ids=["string", "object"],
)
def test_full_root_retains_flow_owned_policy_with_one_origin(
    policy: str,
    expected_config: dict,
    line: int,
) -> None:
    loaded = load_flow(_full_flow(policy), source_label="memory.yaml")
    spec = loaded.compiled.memory_policy

    assert loaded.compiled.flow_id == "memory-root"
    assert type(spec) is MemoryPolicySpec
    assert spec.name == "reflective"
    assert spec.config == expected_config
    assert type(spec.origin) is OriginRef
    assert loaded.compiled.nodes["echo"].__dict__.get("memory_policy") is None
    assert loaded.origin_resolver is not None
    frame = loaded.origin_resolver.resolve(spec.origin)
    assert frame is not None
    assert frame.label == "memory.yaml"
    assert frame.range.start_line == line


def test_full_nested_definition_retains_flow_owned_policy_and_semantic_id() -> None:
    loaded = load_flow(
        """id: parent
name: parent
defs:
  child:
    memory_policy:
      name: reflective
      config: {max_items: 8}
    nodes:
      answer:
        kind: code
        code: tests.engine._compose_codefns:echo
        input: {topic: child}
        output: str
    output: ${answer.output}
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""
    )

    child = loaded.compiled.nodes["invoke"].child
    assert child.flow_id == "child"
    assert child.memory_policy.name == "reflective"
    assert child.memory_policy.config == {"max_items": 8}
    assert child.nodes["answer"].__dict__.get("memory_policy") is None


def test_compact_root_projects_policy_to_its_agent() -> None:
    loaded = load_flow(
        """id: compact_agent
name: compact_agent
memory_policy: reflective
output: str
kind: agent
prompt: answer
"""
    )

    assert loaded.compiled.flow_id == "compact_agent"
    assert loaded.compiled.memory_policy is None
    agent = loaded.compiled.nodes["compact_agent"]
    assert type(agent) is AgentNode
    assert agent.memory_policy.name == "reflective"


def test_flat_agent_definition_projects_policy_to_its_agent() -> None:
    loaded = load_flow(
        """id: parent
name: parent
defs:
  child:
    kind: agent
    memory_policy: reflective
    prompt: answer
    output: str
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""
    )

    child = loaded.compiled.nodes["invoke"].child
    assert child.flow_id == "child"
    assert child.memory_policy is None
    assert child.nodes["child"].memory_policy.name == "reflective"


@pytest.mark.parametrize(
    "source",
    [
        """id: compact-code
name: compact_code
memory_policy: reflective
output: str
kind: code
code: tests.engine._compose_codefns:echo
""",
        """id: ordinary-code
name: ordinary_code
nodes:
  echo:
    kind: code
    memory_policy: reflective
    code: tests.engine._compose_codefns:echo
    output: str
output: ${echo.output}
""",
    ],
    ids=["compact", "ordinary"],
)
def test_non_agent_node_rejects_memory_policy_at_the_authored_value(source: str) -> None:
    with pytest.raises(LoadError, match="memory_policy.*not allowed") as caught:
        load_flow(source)

    frame = caught.value.diagnostic.frames[-1]
    assert frame.text.splitlines()[frame.range.start_line - 1].strip().startswith(
        "memory_policy:"
    )


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        ("memory_policy: ''\n", "name must be non-empty"),
        ("memory_policy: null\n", "string or a mapping"),
        ("memory_policy: {name: 3}\n", "name must be a non-empty string"),
        ("memory_policy: {name: reflective, config: []}\n", "config must be a mapping"),
        ("memory_policy: {name: reflective, extra: true}\n", "unknown key"),
        ("memory_policy: {name: reflective, 1: true}\n", "unknown key"),
        ("memory_policy: {config: {}}\n", "requires 'name'"),
    ],
)
def test_malformed_flow_policy_is_located_at_its_authored_occurrence(
    surface: str,
    message: str,
) -> None:
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(_full_flow(surface))

    assert caught.value.diagnostic.frames[-1].range.start_line == 3


def test_inline_body_rejects_flow_policy_but_allows_child_agent_policy() -> None:
    rejected = """id: inline-policy
name: inline_policy
nodes:
  wrapper:
    kind: call
    memory_policy: reflective
    nodes:
      answer:
        kind: agent
        prompt: answer
        output: str
    output: ${answer.output}
output: ${wrapper.output}
"""
    with pytest.raises(LoadError, match="memory_policy.*not allowed") as caught:
        load_flow(rejected)
    assert caught.value.diagnostic.frames[-1].range == SourceRange(6, 20, 6, 30)

    allowed = rejected.replace("    memory_policy: reflective\n", "").replace(
        "        prompt: answer\n",
        "        memory_policy: reflective\n        prompt: answer\n",
    )
    loaded = load_flow(allowed)
    child = loaded.compiled.nodes["wrapper"].child
    assert child.memory_policy is None
    assert child.nodes["answer"].memory_policy.name == "reflective"


def test_python_constructors_default_and_validate_authored_memory_fields() -> None:
    flow = CompiledFlow({}, [])
    agent = AgentNode("agent")

    assert flow.flow_id is None
    assert flow.memory_policy is None
    assert agent.memory_policy is None

    spec = MemoryPolicySpec("reflective", {})
    assert CompiledFlow({}, [], flow_id="flow", memory_policy=spec).memory_policy is spec
    assert AgentNode("agent", memory_policy=spec).memory_policy is spec

    for invalid in ("", 1, True):
        with pytest.raises((TypeError, ValueError)):
            CompiledFlow({}, [], flow_id=invalid)  # type: ignore[arg-type]

    class DerivedString(str):
        """Invalid semantic identifier subclass."""

    with pytest.raises(TypeError, match="exact string"):
        CompiledFlow({}, [], flow_id=DerivedString("flow"))

    class DerivedPolicy(MemoryPolicySpec):
        """Invalid semantic carrier subclass."""

    derived = DerivedPolicy("reflective", {})
    with pytest.raises(TypeError, match="exact MemoryPolicySpec"):
        CompiledFlow({}, [], memory_policy=derived)
    with pytest.raises(TypeError, match="exact MemoryPolicySpec"):
        AgentNode("agent", memory_policy=derived)


def test_authored_identity_includes_flow_id_and_policy_but_excludes_origin() -> None:
    plain = CompiledFlow({}, [])
    same_plain = copy.deepcopy(plain)
    flow_named = CompiledFlow({}, [], flow_id="named")
    flow_policy = CompiledFlow(
        {},
        [],
        memory_policy=MemoryPolicySpec("reflective", {"max_items": 4}),
    )
    flow_policy_relocated = CompiledFlow(
        {},
        [],
        memory_policy=MemoryPolicySpec(
            "reflective",
            {"max_items": 4},
            OriginRef("other/source"),
        ),
    )
    flow_policy_renamed = CompiledFlow(
        {},
        [],
        memory_policy=MemoryPolicySpec("episodic", {"max_items": 4}),
    )
    flow_policy_reconfigured = CompiledFlow(
        {},
        [],
        memory_policy=MemoryPolicySpec("reflective", {"max_items": 5}),
    )
    agent_policy = CompiledFlow(
        {"agent": AgentNode("agent", memory_policy=MemoryPolicySpec("reflective", {}))},
        [],
    )

    plain_manifest = flow_manifest(plain)
    assert "flow_id" not in plain_manifest
    assert "memory_policy" not in plain_manifest
    assert flow_manifest(same_plain) == plain_manifest
    assert flow_fingerprint(same_plain) == flow_fingerprint(plain)
    assert flow_fingerprint(flow_named) != flow_fingerprint(plain)
    assert flow_fingerprint(flow_policy) != flow_fingerprint(plain)
    assert flow_fingerprint(flow_policy_relocated) == flow_fingerprint(flow_policy)
    assert flow_fingerprint(flow_policy_renamed) != flow_fingerprint(flow_policy)
    assert flow_fingerprint(flow_policy_reconfigured) != flow_fingerprint(flow_policy)
    assert flow_fingerprint(agent_policy) != flow_fingerprint(plain)
    assert flow_manifest(flow_policy)["memory_policy"] == {
        "config": {
            "entries": [
                {
                    "key": {"type": "str", "value": "max_items"},
                    "value": {"type": "int", "value": "4"},
                }
            ],
            "type": "mapping",
        },
        "name": "reflective",
    }
    assert flow_manifest(agent_policy)["nodes"]["agent"]["memory_policy"] == {
        "config": {"entries": [], "type": "mapping"},
        "name": "reflective",
    }
    assert expansion_digest(flow_policy) == expansion_digest(plain)


@pytest.mark.parametrize("config", _PRIVATE_CONFIG_CASES)
def test_policy_config_rejects_host_private_fields(config: dict) -> None:
    with pytest.raises(ValueError, match="forbidden private key"):
        MemoryPolicySpec("reflective", config)


@pytest.mark.parametrize("config", _PRIVATE_CONFIG_CASES)
def test_authored_policy_config_rejects_host_private_fields(config: dict) -> None:
    source = _full_flow(
        "memory_policy:\n"
        "  name: reflective\n"
        f"  config: {json.dumps(config)}\n"
    )

    with pytest.raises(LoadError, match="forbidden private key"):
        load_flow(source, source_label="memory.yaml")


def test_policy_config_keeps_public_behavior_options() -> None:
    config = {
        "max_items": 20,
        "recall_limit": 5,
        "query_options": {"strategy": "semantic", "path_filter": "docs/**"},
        "backend": {"mode": "memory"},
        "database_query_options": {"timeout": 3},
    }

    assert MemoryPolicySpec("reflective", config).config == config


def test_manifest_redaction_remains_defense_in_depth_for_forged_policy() -> None:
    policy = MemoryPolicySpec("reflective", {})
    flow = CompiledFlow({}, [], memory_policy=policy)
    private = MemorySnapshotEnvelope(
        "opaque",
        {"nested": {"api_key": "must-not-appear"}},
    )
    object.__setattr__(policy, "_config", private._snapshot)

    first_manifest = flow_manifest(flow)
    rendered = json.dumps(first_manifest, sort_keys=True)
    assert "must-not-appear" not in rendered
    replacement = MemorySnapshotEnvelope(
        "opaque",
        {"nested": {"api_key": "different-secret"}},
    )
    object.__setattr__(policy, "_config", replacement._snapshot)
    assert flow_manifest(flow) == first_manifest


def test_expansion_digest_excludes_flow_id_and_agent_memory_policy() -> None:
    plain_flow = CompiledFlow({}, [])
    named_flow = CompiledFlow({}, [], flow_id="named")
    plain_agent = CompiledFlow({"agent": AgentNode("agent")}, [])
    memory_agent = CompiledFlow(
        {
            "agent": AgentNode(
                "agent",
                memory_policy=MemoryPolicySpec(
                    "reflective",
                    {"max_items": 4},
                ),
            )
        },
        [],
    )

    assert expansion_digest(named_flow) == expansion_digest(plain_flow)
    assert expansion_digest(memory_agent) == expansion_digest(plain_agent)


def test_compose_loaded_flow_id_changes_the_previous_memoryless_identity() -> None:
    first = load_flow("id: first\nname: same\nnodes: {}\n").compiled
    second = load_flow("id: second\nname: same\nnodes: {}\n").compiled

    assert first.flow_id == "first"
    assert second.flow_id == "second"
    assert flow_fingerprint(first) != flow_fingerprint(second)
