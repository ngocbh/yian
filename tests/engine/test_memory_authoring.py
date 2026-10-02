"""Authored flow and AGENT long-term-memory semantics."""

from __future__ import annotations

import copy
import json

import pytest

from agent_composer.compile.manifest import expansion_digest, flow_fingerprint, flow_manifest
from agent_composer.compile.model import CompiledFlow
from agent_composer.compose import LoadError, load_flow
from agent_composer.diagnostics import OriginRef, SourceRange
from agent_composer.memory import DeclSite, MemoryDecl, MemoryPolicySpec, MemorySnapshotEnvelope
from agent_composer.nodes.agent import AgentNode


_PLURAL_PRIVATE_CONFIG_CASES = (
    {"nested": {"secretTokens": ["private"]}},
    {"nested": {"apiKeys": ["private"]}},
    {"nested": {"backendPaths": ["/private/backend"]}},
    {"nested": {"pythonModules": ["package.policy"]}},
)


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
    {"nested": {"pythonImports": ["package.policy"]}},
    {"python": {"imports": ["package.policy"]}},
    {"nested": {"modulePath": "package.policy"}},
    {"nested": {"class-path": "package.Policy"}},
    {"nested": {"secretToken": "private"}},
    *_PLURAL_PRIVATE_CONFIG_CASES,
    {"nested": {"credentialFile": "/private/credentials"}},
    {"nested": {"apiKeyPath": "/private/api-key"}},
    {"nested": {"orgtenantid": "private"}},
    {"nested": {"backendDir": "/private/backend"}},
    {"nested": {"storageDirectory": "/private/storage"}},
    {"backend": {"path": "/private/backend"}},
    {"backend": {"directory": "/private/backend"}},
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
        ("long_term_memory: reflective\n", {}, 3),
        (
            "long_term_memory:\n"
            "  name: reflective\n"
            "  config:\n"
            "    max_items: 20\n"
            "    recall_limit: 5\n",
            {"max_items": 20, "recall_limit": 5},
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
    (decl,) = loaded.compiled.memories
    assert loaded.compiled.flow_id == "memory-root"
    assert type(decl) is MemoryDecl
    assert (decl.name, decl.type, decl.key) == ("reflective", "reflective", "reflective")
    assert decl.site == DeclSite((), ("root",), None)
    assert decl.config == expected_config
    assert type(decl.origin) is OriginRef
    assert loaded.compiled.nodes["echo"].__dict__.get("memories") is None
    spec = decl
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
    long_term_memory:
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
    assert child.memories[0].name == "reflective"
    assert child.memories[0].config == {"max_items": 8}
    assert child.nodes["answer"].__dict__.get("memories") is None


def test_compact_root_projects_policy_to_its_agent() -> None:
    loaded = load_flow(
        """id: compact_agent
name: compact_agent
long_term_memory: reflective
output: str
kind: agent
prompt: answer
"""
    )

    assert loaded.compiled.flow_id == "compact_agent"
    assert loaded.compiled.memories == ()
    agent = loaded.compiled.nodes["compact_agent"]
    assert type(agent) is AgentNode
    assert agent.memories[0].name == "reflective"


def test_flat_agent_definition_projects_policy_to_its_agent() -> None:
    loaded = load_flow(
        """id: parent
name: parent
defs:
  child:
    kind: agent
    long_term_memory: reflective
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
    assert child.memories == ()
    assert child.nodes["child"].memories[0].name == "reflective"


def test_compose_stamps_decl_sites_for_root_def_and_inline_owners() -> None:
    loaded = load_flow(
        """id: parent
name: parent
long_term_memory: outer
defs:
  child:
    kind: agent
    long_term_memory: inner
    prompt: answer
    output: str
nodes:
  top:
    kind: agent
    long_term_memory: local
    prompt: answer
    output: str
  invoke:
    kind: call
    call: child
  wrapper:
    kind: call
    nodes:
      answer:
        kind: agent
        long_term_memory: inline
        prompt: answer
        output: str
    output: ${answer.output}
output: ${top.output}
"""
    )
    compiled = loaded.compiled
    assert [d.site for d in compiled.memories] == [DeclSite((), ("root",), None)]
    assert compiled.nodes["top"].memories[0].site == DeclSite((), ("root",), "top")
    child = compiled.nodes["invoke"].child
    assert child.memories == ()
    assert child.nodes["child"].memories[0].site == DeclSite((), ("def", "child"), "child")
    inline = compiled.nodes["wrapper"].child
    assert inline.nodes["answer"].memories[0].site == DeclSite(
        (), ("root", "node", "wrapper", "inline"), "answer"
    )


@pytest.mark.parametrize(
    "source",
    [
        """id: compact-code
name: compact_code
long_term_memory: reflective
output: str
kind: code
code: tests.engine._compose_codefns:echo
""",
        """id: ordinary-code
name: ordinary_code
nodes:
  echo:
    kind: code
    long_term_memory: reflective
    code: tests.engine._compose_codefns:echo
    output: str
output: ${echo.output}
""",
    ],
    ids=["compact", "ordinary"],
)
def test_non_agent_node_rejects_long_term_memory_at_the_authored_value(source: str) -> None:
    with pytest.raises(LoadError, match="long_term_memory.*not allowed") as caught:
        load_flow(source)

    frame = caught.value.diagnostic.frames[-1]
    assert frame.text.splitlines()[frame.range.start_line - 1].strip().startswith(
        "long_term_memory:"
    )


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        ("long_term_memory: ''\n", "name must be non-empty"),
        ("long_term_memory: null\n", "string or a mapping"),
        ("long_term_memory: {name: 3}\n", "name must be a non-empty string"),
        ("long_term_memory: {name: reflective, config: []}\n", "config must be a mapping"),
        ("long_term_memory: {name: reflective, extra: true}\n", "unknown key"),
        ("long_term_memory: {name: reflective, 1: true}\n", "unknown key"),
        ("long_term_memory: {config: {}}\n", "requires 'name'"),
    ],
)
def test_malformed_flow_policy_is_located_at_its_authored_occurrence(
    surface: str,
    message: str,
) -> None:
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(_full_flow(surface))

    assert caught.value.diagnostic.frames[-1].range.start_line == 3


@pytest.mark.parametrize(
    ("source", "line"),
    [
        (
            """id: full
name: full
memory_policy: reflective
nodes: {}
""",
            3,
        ),
        (
            """id: compact
name: compact
memory_policy: reflective
kind: agent
prompt: answer
output: str
""",
            3,
        ),
        (
            """id: ordinary
name: ordinary
nodes:
  answer:
    kind: agent
    memory_policy: reflective
    prompt: answer
    output: str
""",
            6,
        ),
        (
            """id: parent
name: parent
defs:
  child:
    memory_policy: reflective
    nodes: {}
nodes: {}
""",
            5,
        ),
        (
            """id: parent
name: parent
defs:
  child:
    kind: agent
    memory_policy: reflective
    prompt: answer
    output: str
nodes: {}
""",
            6,
        ),
        (
            """id: both-old-first
name: both_old_first
memory_policy: reflective
long_term_memory: episodic
nodes: {}
""",
            3,
        ),
        (
            """id: both-new-first
name: both_new_first
long_term_memory: episodic
memory_policy: reflective
nodes: {}
""",
            4,
        ),
    ],
    ids=[
        "full-root",
        "compact-root",
        "ordinary-agent",
        "multi-node-def",
        "flat-agent-def",
        "both-old-first",
        "both-new-first",
    ],
)
def test_retired_memory_policy_is_a_located_migration_error(
    source: str,
    line: int,
) -> None:
    with pytest.raises(
        LoadError,
        match="memory_policy.*renamed.*long_term_memory",
    ) as caught:
        load_flow(source)

    frame = caught.value.diagnostic.frames[-1]
    assert frame.range.start_line == line
    assert frame.text.splitlines()[line - 1].strip().startswith("memory_policy:")


def test_inline_body_rejects_flow_policy_but_allows_child_agent_policy() -> None:
    rejected = """id: inline-policy
name: inline_policy
nodes:
  wrapper:
    kind: call
    long_term_memory: reflective
    nodes:
      answer:
        kind: agent
        prompt: answer
        output: str
    output: ${answer.output}
output: ${wrapper.output}
"""
    with pytest.raises(LoadError, match="long_term_memory.*not allowed") as caught:
        load_flow(rejected)
    assert caught.value.diagnostic.frames[-1].range == SourceRange(6, 23, 6, 33)

    allowed = rejected.replace("    long_term_memory: reflective\n", "").replace(
        "        prompt: answer\n",
        "        long_term_memory: reflective\n        prompt: answer\n",
    )
    loaded = load_flow(allowed)
    child = loaded.compiled.nodes["wrapper"].child
    assert child.memories == ()
    assert child.nodes["answer"].memories[0].name == "reflective"


def test_python_constructors_default_and_validate_authored_memory_fields() -> None:
    flow = CompiledFlow({}, [])
    agent = AgentNode("agent")

    assert flow.flow_id is None
    assert flow.memories == ()
    assert agent.memories == ()

    flow_decl = MemoryDecl("reflective", "reflective", {}, DeclSite((), ("root",), None))
    agent_decl = MemoryDecl("reflective", "reflective", {}, DeclSite((), ("root",), "agent"))
    assert CompiledFlow({}, [], flow_id="flow", memories=(flow_decl,)).memories == (flow_decl,)
    assert CompiledFlow({}, [], memories=(flow_decl,)).memories[0] is flow_decl
    assert AgentNode("agent", memories=(agent_decl,)).memories[0] is agent_decl
    with pytest.raises(ValueError, match="flow DeclSites"):
        CompiledFlow({}, [], memories=(agent_decl,))
    with pytest.raises(ValueError, match="AGENT DeclSites"):
        AgentNode("agent", memories=(flow_decl,))

    for invalid in ("", 1, True):
        with pytest.raises((TypeError, ValueError)):
            CompiledFlow({}, [], flow_id=invalid)  # type: ignore[arg-type]

    class DerivedString(str):
        """Invalid semantic identifier subclass."""

    with pytest.raises(TypeError, match="exact string"):
        CompiledFlow({}, [], flow_id=DerivedString("flow"))

    class DerivedDecl(MemoryDecl):
        """Invalid semantic carrier subclass."""

    derived_flow = DerivedDecl("reflective", "reflective", {}, DeclSite((), ("root",), None))
    derived_agent = DerivedDecl("reflective", "reflective", {}, DeclSite((), ("root",), "agent"))
    with pytest.raises(TypeError, match="exact MemoryDecl"):
        CompiledFlow({}, [], memories=(derived_flow,))
    with pytest.raises(TypeError, match="exact MemoryDecl"):
        AgentNode("agent", memories=(derived_agent,))
    with pytest.raises(TypeError, match="exact MemoryDecl"):
        CompiledFlow({}, [], memories=[flow_decl])  # type: ignore[arg-type]


def test_authored_identity_includes_flow_id_and_policy_but_excludes_origin() -> None:
    plain = CompiledFlow({}, [])
    same_plain = copy.deepcopy(plain)
    flow_named = CompiledFlow({}, [], flow_id="named")
    root = DeclSite((), ("root",), None)
    flow_policy = CompiledFlow(
        {},
        [],
        memories=(MemoryDecl("reflective", "reflective", {"max_items": 4}, root),),
    )
    flow_policy_relocated = CompiledFlow(
        {},
        [],
        memories=(
            MemoryDecl(
                "reflective",
                "reflective",
                {"max_items": 4},
                root,
                OriginRef("other/source"),
            ),
        ),
    )
    flow_policy_renamed = CompiledFlow(
        {},
        [],
        memories=(MemoryDecl("custom", "custom", {"max_items": 4}, root),),
    )
    flow_policy_reconfigured = CompiledFlow(
        {},
        [],
        memories=(MemoryDecl("reflective", "reflective", {"max_items": 5}, root),),
    )
    flow_policy_resited = CompiledFlow(
        {},
        [],
        memories=(
            MemoryDecl(
                "reflective", "reflective", {"max_items": 4}, DeclSite((), ("def", "x"), None)
            ),
        ),
    )
    agent_policy = CompiledFlow(
        {
            "agent": AgentNode(
                "agent",
                memories=(
                    MemoryDecl("reflective", "reflective", {}, DeclSite((), ("root",), "agent")),
                ),
            )
        },
        [],
    )

    plain_manifest = flow_manifest(plain)
    assert "flow_id" not in plain_manifest
    assert "memories" not in plain_manifest
    assert flow_manifest(same_plain) == plain_manifest
    assert flow_fingerprint(same_plain) == flow_fingerprint(plain)
    assert flow_fingerprint(flow_named) != flow_fingerprint(plain)
    assert flow_fingerprint(flow_policy) != flow_fingerprint(plain)
    assert flow_fingerprint(flow_policy_relocated) == flow_fingerprint(flow_policy)
    assert flow_fingerprint(flow_policy_renamed) != flow_fingerprint(flow_policy)
    assert flow_fingerprint(flow_policy_reconfigured) != flow_fingerprint(flow_policy)
    assert flow_fingerprint(flow_policy_resited) != flow_fingerprint(flow_policy)
    assert flow_fingerprint(agent_policy) != flow_fingerprint(plain)
    assert flow_manifest(flow_policy)["memories"] == [{
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
        "site": {"node": None, "owner": ["root"], "scope": []},
        "type": "reflective",
    }]
    assert flow_manifest(agent_policy)["nodes"]["agent"]["memories"] == [{
        "config": {"entries": [], "type": "mapping"},
        "name": "reflective",
        "site": {"node": "agent", "owner": ["root"], "scope": []},
        "type": "reflective",
    }]
    assert expansion_digest(flow_policy) == expansion_digest(plain)


@pytest.mark.parametrize("config", _PRIVATE_CONFIG_CASES)
def test_policy_config_rejects_host_private_fields(config: dict) -> None:
    with pytest.raises(ValueError, match="forbidden private key"):
        MemoryPolicySpec("reflective", config)


@pytest.mark.parametrize("config", _PRIVATE_CONFIG_CASES)
def test_authored_policy_config_rejects_host_private_fields(config: dict) -> None:
    source = _full_flow(
        "long_term_memory:\n"
        "  name: reflective\n"
        f"  config: {json.dumps(config)}\n"
    )

    with pytest.raises(LoadError, match="forbidden private key"):
        load_flow(source, source_label="memory.yaml")


def test_policy_config_keeps_public_behavior_options() -> None:
    config = {
        "max_items": 20,
        "recall_limit": 5,
        "token_budget": 4096,
        "stop_token": "semantic-stop-marker",
        "continuationToken": "semantic-continuation-marker",
        "query_options": {"strategy": "semantic", "path_filter": "docs/**"},
        "backend": {"mode": "memory"},
        "database_query_options": {"timeout": 3},
    }

    policy = MemoryDecl("custom", "custom", config, DeclSite((), ("root",), None))

    assert policy.config == config
    rendered = json.dumps(
        flow_manifest(CompiledFlow({}, [], memories=(policy,))),
        sort_keys=True,
    )
    assert "semantic-stop-marker" in rendered
    assert "semantic-continuation-marker" in rendered


def test_manifest_redaction_remains_defense_in_depth_for_forged_policy() -> None:
    policy = MemoryDecl("reflective", "reflective", {}, DeclSite((), ("root",), None))
    flow = CompiledFlow({}, [], memories=(policy,))
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
                memories=(
                    MemoryDecl(
                        "reflective",
                        "reflective",
                        {"max_items": 4},
                        DeclSite((), ("root",), "agent"),
                    ),
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
