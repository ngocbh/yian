"""Pure static memory-layout derivation and atomic scope application."""

from __future__ import annotations

import copy

import pytest

from agent_composer.compile.manifest import flow_fingerprint, flow_manifest
from agent_composer.compile.memory_layout import (
    apply_memory_layout,
    check_session_compatibility,
    memory_stacks,
    resolve_memory_layout,
)
from agent_composer.compile.model import END_ID, START_ID, CompiledFlow
from agent_composer.compose import LoadError, load_flow
from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    DeclSite,
    MemoryDecl,
    MemoryScope,
    MemorySetupError,
    MemoryStack,
)
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.code import CodeNode
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode


_NESTED = """id: root
name: root
long_term_memory: outer
defs:
  child:
    long_term_memory:
      inner: {type: inner, config: {limit: 3}}
    nodes:
      answer:
        kind: agent
        long_term_memory: local
        prompt: answer
        output: str
      normalize:
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


def _target(layout, *locator: str):
    """Return one resolved target by its structural locator."""
    return next(target for target in layout.targets if target.locator == locator)


def test_disabled_layout_needs_no_flow_id_and_leaves_every_scope_none() -> None:
    flow = CompiledFlow(
        {"plain": CodeNode("plain", code="return 1")},
        [],
        wiring={"plain": {}},
    )
    flow.nodes["plain"].memory_scope = MemoryScope(("plain",))

    layout = resolve_memory_layout(flow)

    assert layout.enabled is False
    assert layout.memories == ()
    assert [(target.locator, target.scope) for target in layout.targets] == [
        (("plain",), None)
    ]
    # Resolution is pure and ignores an existing derived scope.
    assert flow.nodes["plain"].memory_scope == MemoryScope(("plain",))
    with pytest.raises(MemorySetupError, match="different memory scope"):
        apply_memory_layout(flow, layout)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-both", "missing direct START and END"),
        ("missing-start", "missing direct START"),
        ("missing-end", "missing direct END"),
        ("start-subclass", "start_id must be a non-empty exact string"),
        ("end-empty", "end_id must be a non-empty exact string"),
        ("same", "distinct START and END"),
        ("wrong-start-id", "start_id must be canonical '__start__'"),
        ("wrong-end-id", "end_id must be canonical '__end__'"),
        ("wrong-start-class", "START boundary must be an exact StartNode"),
        ("wrong-end-class", "END boundary must be an exact EndNode"),
    ],
)
def test_memory_owning_python_flow_requires_exact_direct_boundaries(
    case: str,
    message: str,
) -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical boundary identifier."""

    origin = OriginRef("flow/memories/x")
    nodes = {
        START_ID: StartNode(START_ID, params=[]),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    if case == "missing-both":
        nodes = {"body": CodeNode("body", code="return 1")}
    elif case == "missing-start":
        nodes.pop(START_ID)
    elif case == "missing-end":
        nodes.pop(END_ID)
    elif case == "wrong-start-id":
        nodes["entry"] = StartNode("entry", params=[])
    elif case == "wrong-end-id":
        nodes["result"] = EndNode.record("result", output_names=[])
    elif case == "wrong-start-class":
        nodes[START_ID] = CodeNode(START_ID, code="return None")
    elif case == "wrong-end-class":
        nodes[END_ID] = CodeNode(END_ID, code="return None")
    flow = CompiledFlow(
        nodes,
        [],
        wiring={node_id: {} for node_id in nodes},
        flow_id="root",
        memories=(MemoryDecl("reflective", "reflective", {}, DeclSite((), ("root",), None), origin),),
    )
    if case == "start-subclass":
        flow.start_id = StringSubclass(START_ID)
    elif case == "end-empty":
        flow.end_id = ""
    elif case == "same":
        flow.end_id = START_ID
    elif case == "wrong-start-id":
        flow.start_id = "entry"
    elif case == "wrong-end-id":
        flow.end_id = "result"

    with pytest.raises(MemorySetupError, match=message) as caught:
        resolve_memory_layout(flow)

    assert caught.value.origin is origin
    assert all(node.memory_scope is None for node in nodes.values())


def test_flow_start_and_end_scopes_carry_the_enclosing_keys() -> None:
    without_child_policy = load_flow(
        _NESTED.replace(
            "    long_term_memory:\n      inner: {type: inner, config: {limit: 3}}\n",
            "",
        )
    ).compiled
    inherited = resolve_memory_layout(without_child_policy)

    for node_id in (START_ID, END_ID):
        assert _target(inherited, "invoke", node_id).scope == MemoryScope(
            ("invoke", node_id), ("outer",)
        )
    assert _target(inherited, "invoke", "answer").scope.memories == ("outer", "local")
    assert [memory.key for memory in inherited.memories] == ["local", "outer"]


def test_composite_drivers_see_only_enclosing_keys() -> None:
    layout = resolve_memory_layout(load_flow(_NESTED).compiled)

    # The CALL sees only its enclosing flow's keys; the child flow's keys
    # begin inside the called child.
    assert _target(layout, "invoke").scope.memories == ("outer",)
    assert [(memory.key, memory.placements) for memory in layout.memories] == [
        ("inner", ()),
        ("local", (("invoke", "answer"),)),
        ("outer", ()),
    ]


def test_repeated_definition_shares_one_owner_and_inline_children_own_their_keys() -> None:
    repeated_def = load_flow(
        """id: root
name: root
defs:
  child:
    long_term_memory: reflective
    nodes:
      answer:
        kind: code
        code: tests.engine._compose_codefns:echo
        input: {topic: child}
        output: str
    output: ${answer.output}
nodes:
  left: {kind: call, call: child}
  right: {kind: call, call: child}
output: ${right.output}
"""
    ).compiled
    definition_layout = resolve_memory_layout(repeated_def)
    (memory,) = definition_layout.memories
    assert memory.owners == (DeclSite((), ("def", "child"), None),)
    stacks = {stack.locator: stack.memories for stack in memory_stacks(definition_layout)}
    assert stacks[("left",)] == ()
    assert stacks[("left", "answer")] == ("reflective",)
    assert stacks[("right", "answer")] == ("reflective",)

    inline = load_flow(
        """id: root
name: root
nodes:
  left:
    kind: call
    nodes:
      answer:
        kind: agent
        long_term_memory: reflective
        prompt: left
        output: str
    output: ${answer.output}
  right:
    kind: call
    nodes:
      answer:
        kind: agent
        long_term_memory: reflective
        prompt: right
        output: str
    output: ${answer.output}
output: ${right.output}
"""
    ).compiled
    (memory,) = resolve_memory_layout(inline).memories
    assert memory.owners == (
        DeclSite((), ("root", "node", "left", "inline"), "answer"),
        DeclSite((), ("root", "node", "right", "inline"), "answer"),
    )
    assert memory.placements == (("left", "answer"), ("right", "answer"))
    assert memory.conversation_eligible is False


def test_apply_is_idempotent_and_does_not_change_authored_identity() -> None:
    flow = load_flow(_NESTED).compiled
    manifest = flow_manifest(flow)
    fingerprint = flow_fingerprint(flow)
    layout = resolve_memory_layout(flow)

    assert resolve_memory_layout(flow) == layout
    apply_memory_layout(flow, layout)
    assert resolve_memory_layout(flow) == layout
    first_scopes = tuple(target.node.memory_scope for target in layout.targets)
    apply_memory_layout(flow, layout)

    assert tuple(target.node.memory_scope for target in layout.targets) == first_scopes
    assert flow_manifest(flow) == manifest
    assert flow_fingerprint(flow) == fingerprint


def test_apply_scope_mismatch_is_atomic() -> None:
    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    final = layout.targets[-1]
    final.node.memory_scope = MemoryScope(("different",))

    with pytest.raises(MemorySetupError, match="different memory scope"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets[:-1])
    assert final.node.memory_scope == MemoryScope(("different",))


def test_apply_revalidates_forged_nested_layout_scope_before_stamping() -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical memory key."""

    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    forged = layout.targets[-1].scope
    object.__setattr__(
        forged, "memories", tuple(StringSubclass(key) for key in forged.memories)
    )

    with pytest.raises(TypeError, match="memories components must be a string"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets)


def test_apply_revalidates_a_forged_key_table_before_stamping() -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical memory key."""

    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    object.__setattr__(layout.memories[0], "key", StringSubclass("inner"))

    with pytest.raises(TypeError, match="key must be a string"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets)


def test_apply_revalidates_forged_equal_existing_scope_atomically() -> None:
    class StringSubclass(str):
        """Equality-compatible but noncanonical root identity."""

    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    final = layout.targets[-1]
    forged = copy.deepcopy(final.scope)
    object.__setattr__(forged, "memories", tuple(StringSubclass(key) for key in forged.memories))
    final.node.memory_scope = forged

    with pytest.raises(MemorySetupError, match="noncanonical memory scope"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets[:-1])
    assert final.node.memory_scope is forged


def test_apply_stale_object_target_is_atomic() -> None:
    flow = load_flow(_NESTED).compiled
    layout = resolve_memory_layout(flow)
    replacement = copy.deepcopy(flow.nodes["invoke"])
    flow.nodes["invoke"] = replacement

    with pytest.raises(MemorySetupError, match="same node"):
        apply_memory_layout(flow, layout)

    assert replacement.memory_scope is None
    assert all(target.node.memory_scope is None for target in layout.targets)


def test_apply_rejects_shared_child_targets_before_stamping() -> None:
    child = load_flow(
        """id: child
name: child
long_term_memory: reflective
nodes:
  answer:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: {topic: child}
    output: str
output: ${answer.output}
"""
    ).compiled
    flow = CompiledFlow(
        {
            "left": CallNode("left", flow_id="child", child=child),
            "right": CallNode("right", flow_id="child", child=child),
        },
        [],
        wiring={"left": {}, "right": {}},
        flow_id="root",
    )
    layout = resolve_memory_layout(flow)

    with pytest.raises(MemorySetupError, match="multiple static paths"):
        apply_memory_layout(flow, layout)

    assert all(target.node.memory_scope is None for target in layout.targets)


def test_recursive_static_child_cycle_is_rejected() -> None:
    flow = CompiledFlow({}, [], wiring={}, flow_id="root")
    cycle = CallNode("cycle", flow_id="root", child=flow)
    flow.nodes["cycle"] = cycle
    flow.wiring["cycle"] = {}

    with pytest.raises(MemorySetupError, match="recursive child-flow cycle"):
        resolve_memory_layout(flow)

    assert cycle.memory_scope is None


def test_key_table_retains_config_and_origin_outside_node_scopes() -> None:
    flow = load_flow(_NESTED, source_label="memory.yaml").compiled
    layout = resolve_memory_layout(flow)
    inner = next(memory for memory in layout.memories if memory.key == "inner")

    assert inner.config == {"limit": 3}
    assert inner.owners == (DeclSite((), ("def", "child"), None),)
    assert inner.origin == flow.nodes["invoke"].child.memories[0].origin
    assert inner.origin is not None
    assert not hasattr(_target(layout, "invoke", "answer").scope, "origin")


def test_layout_records_key_table_with_sorted_owners_and_placements() -> None:
    source = _NESTED.replace("long_term_memory: local", "long_term_memory: outer")
    layout = resolve_memory_layout(load_flow(source).compiled)

    assert [
        (m.key, m.type, m.config, m.owners, m.conversation_eligible, m.placements)
        for m in layout.memories
    ] == [
        ("inner", "inner", {"limit": 3}, (DeclSite((), ("def", "child"), None),), False, ()),
        (
            "outer",
            "outer",
            {},
            (DeclSite((), ("def", "child"), "answer"), DeclSite((), ("root",), None)),
            False,
            (("invoke", "answer"),),
        ),
    ]
    assert layout.memories[1].spec.owners == layout.memories[1].owners


def test_memory_stacks_dedupe_a_repeated_key_at_its_outermost_position() -> None:
    source = _NESTED.replace("long_term_memory: local", "long_term_memory: outer")
    layout = resolve_memory_layout(load_flow(source).compiled)

    stacks = memory_stacks(layout)
    by_locator = {stack.locator: stack.memories for stack in stacks}
    assert [stack.locator for stack in stacks] == sorted(by_locator)
    assert set(by_locator) == {target.locator for target in layout.targets}
    assert by_locator[("invoke",)] == ("outer",)
    assert by_locator[("invoke", "normalize")] == ("outer", "inner")
    assert by_locator[("invoke", "answer")] == ("outer", "inner")


def test_single_agent_owner_is_conversation_eligible() -> None:
    flow = load_flow(
        "id: one\nname: one\nnodes:\n  ask:\n    kind: agent\n"
        "    long_term_memory: chat\n    prompt: hi\n    output: str\n"
        "output: ${ask.output}\n"
    ).compiled
    (memory,) = resolve_memory_layout(flow).memories
    assert memory.owners == (DeclSite((), ("root",), "ask"),)
    assert memory.conversation_eligible is True
    assert memory.placements == (("ask",),)


def test_one_key_with_two_configs_is_a_load_error_at_the_first_site() -> None:
    source = _NESTED.replace(
        "long_term_memory: local", "long_term_memory: {inner: {type: inner, config: {limit: 4}}}"
    )
    with pytest.raises(LoadError, match="memory 'inner'.*different type or config") as caught:
        load_flow(source)
    notes = caught.value.diagnostic.notes
    assert notes[0] == "also declared at def/child AGENT 'answer'"
    assert len(notes) == 2 and notes[1].startswith("related declaration at <flow>:")


@pytest.mark.parametrize("other", [{"limit": True}, {"limit": 1.0}, {"limit": 2}])
@pytest.mark.parametrize("same_site", [False, True])
def test_memory_table_compares_configs_as_canonical_json(other, same_site) -> None:
    from agent_composer.compile.memory_layout import _memory_table

    root = DeclSite((), ("root",), None)
    first = MemoryDecl("notes", "notes", {"limit": 1}, root)
    clash = MemoryDecl("notes", "notes", other, root if same_site else DeclSite((), ("def", "x"), None))
    with pytest.raises(MemorySetupError, match="memory 'notes' is declared with different type or config") as caught:
        _memory_table([(first, None), (clash, None)])
    assert len(caught.value.__notes__) == 1


def test_memory_table_merges_identical_declarations_at_one_site() -> None:
    from agent_composer.compile.memory_layout import _memory_table

    site = DeclSite((), ("root",), None)
    first = MemoryDecl("notes", "notes", {"limit": 1}, site)
    again = MemoryDecl("notes", "notes", {"limit": 1}, site)
    (memory,) = _memory_table([(first, None), (again, None), (first, None)])
    assert memory.owners == (site,)


def test_disabled_layout_has_an_empty_key_table() -> None:
    flow = CompiledFlow({"plain": CodeNode("plain", code="return 1")}, [], wiring={"plain": {}})
    layout = resolve_memory_layout(flow)
    assert layout.memories == ()
    assert memory_stacks(layout) == ()


_SESSION_CALLS = """id: calls
name: calls
defs:
  body:
    nodes:
      ask:
        kind: agent
        long_term_memory: session
        prompt: answer
        output: str
    output: ${ask.output}
nodes:
  a: {kind: call, call: body}
  b: {kind: call, call: body}
output: ${[a.output, b.output]}
"""


def test_session_def_called_twice_has_one_owner_and_two_placements() -> None:
    (memory,) = resolve_memory_layout(load_flow(_SESSION_CALLS).compiled).memories
    assert memory.owners == (DeclSite((), ("def", "body"), "ask"),)
    assert memory.conversation_eligible is True
    assert memory.placements == (("a", "ask"), ("b", "ask"))


def test_flow_level_session_is_a_load_error() -> None:
    source = (
        "id: flat\nname: flat\nlong_term_memory: session\nnodes:\n"
        "  ask:\n    kind: agent\n    prompt: hi\n    output: str\n"
        "output: ${ask.output}\n"
    )
    with pytest.raises(
        LoadError, match="session memory 'session' cannot be declared on a flow"
    ) as caught:
        load_flow(source)
    assert caught.value.diagnostic.frames[-1].range.start_line == 3


def test_session_key_with_two_agent_owners_is_a_load_error() -> None:
    source = (
        "id: pair\nname: pair\nnodes:\n"
        "  left:\n    kind: agent\n    long_term_memory: session\n"
        "    prompt: hi\n    output: str\n"
        "  right:\n    kind: agent\n    long_term_memory: session\n"
        "    prompt: hi\n    output: str\n"
        "output: ${[left.output, right.output]}\n"
    )
    with pytest.raises(
        LoadError, match="session memory 'session' has more than one owner"
    ) as caught:
        load_flow(source)
    diagnostic = caught.value.diagnostic
    assert "AGENT 'left'" in diagnostic.message and "AGENT 'right'" in diagnostic.message
    assert diagnostic.frames[-1].range.start_line == 6
    assert diagnostic.notes[0] == "also owned at root AGENT 'right'"
    assert len(diagnostic.notes) == 2
    assert diagnostic.notes[1].startswith("related declaration at <flow>:11:")


def test_config_clash_carries_every_other_site_as_a_related_origin() -> None:
    from agent_composer.compile.memory_layout import _memory_table

    # DeclSite.sort_key() order: ("def", "x") < ("def", "y") < ("root",).
    first = MemoryDecl("notes", "notes", {"limit": 1}, DeclSite((), ("def", "x"), None), OriginRef("a"))
    second = MemoryDecl("notes", "notes", {"limit": 2}, DeclSite((), ("def", "y"), None), OriginRef("b"))
    third = MemoryDecl("notes", "notes", {"limit": 1}, DeclSite((), ("root",), None), OriginRef("c"))
    with pytest.raises(MemorySetupError) as caught:
        _memory_table([(third, None), (second, None), (first, None)])
    assert caught.value.origin == OriginRef("a")
    assert caught.value.related == (OriginRef("b"), OriginRef("c"))
    assert len(caught.value.__notes__) == 2


@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("repeated", ["first", "other"])
def test_config_clash_lists_a_repeated_declaration_once(repeated, copied) -> None:
    from agent_composer.compile.memory_layout import _memory_table

    # A def called from two nodes yields its declaration once per call: the
    # same object, or an equal copy when run_flow deep-copies the child per call.
    in_def = MemoryDecl("notes", "notes", {"limit": 3}, DeclSite((), ("def", "x"), None), OriginRef("a"))
    in_root = MemoryDecl("notes", "notes", {"limit": 2}, DeclSite((), ("root",), None), OriginRef("b"))
    twice, once = (in_def, in_root) if repeated == "first" else (in_root, in_def)
    again = copy.deepcopy(twice) if copied else twice
    assert (again is twice) is not copied
    with pytest.raises(MemorySetupError) as caught:
        _memory_table([(twice, ("p1",)), (again, ("p2",)), (once, None)])
    assert caught.value.origin == OriginRef("a")
    assert caught.value.related == (OriginRef("b"),)
    assert caught.value.__notes__ == ["also declared at root"]


def test_flow_level_session_in_an_uncalled_def_is_not_an_error() -> None:
    source = (
        "id: spare\nname: spare\ndefs:\n"
        "  unused:\n    long_term_memory: session\n    nodes:\n"
        "      ask:\n        kind: agent\n        prompt: hi\n        output: str\n"
        "    output: ${ask.output}\n"
        "nodes:\n  ask:\n    kind: agent\n    long_term_memory: episodic\n"
        "    prompt: hi\n    output: str\n"
        "output: ${ask.output}\n"
    )
    layout = resolve_memory_layout(load_flow(source).compiled)
    assert [memory.key for memory in layout.memories] == ["episodic"]


def test_session_placements_must_resolve_one_compatibility() -> None:
    layout = resolve_memory_layout(load_flow(_SESSION_CALLS).compiled)
    check_session_compatibility(
        layout,
        {"a/ask": {"provider": "openai", "model": "one"}, "b/ask": {"provider": "OpenAI", "model": "two"}},
    )
    with pytest.raises(
        MemorySetupError,
        match=(
            r"session memory 'session' resolves provider 'openai' at 'a/ask' "
            r"but 'anthropic' at 'b/ask'; pin a provider on the AGENT"
        ),
    ):
        check_session_compatibility(
            layout,
            {"a/ask": {"provider": "openai", "model": "m"}, "b/ask": {"provider": "anthropic", "model": "m"}},
        )


def test_session_compatibility_ignores_other_types_and_disabled_layouts() -> None:
    episodic = _SESSION_CALLS.replace("long_term_memory: session", "long_term_memory: episodic")
    layout = resolve_memory_layout(load_flow(episodic).compiled)
    check_session_compatibility(
        layout, {"a/ask": {"provider": "openai"}, "b/ask": {"provider": "anthropic"}}
    )
    plain = CompiledFlow({"plain": CodeNode("plain", code="return 1")}, [], wiring={"plain": {}})
    check_session_compatibility(resolve_memory_layout(plain), {})


def test_scopes_are_root_relative_and_carry_outermost_first_keys() -> None:
    layout = resolve_memory_layout(load_flow(_NESTED).compiled)

    assert layout.enabled is True
    assert not hasattr(layout, "frames")
    assert not hasattr(layout, "root_flow_id")
    assert _target(layout, START_ID).scope == MemoryScope((START_ID,), ("outer",))
    assert _target(layout, "invoke").scope == MemoryScope(("invoke",), ("outer",))
    assert _target(layout, "invoke", "normalize").scope == MemoryScope(
        ("invoke", "normalize"), ("outer", "inner")
    )
    assert _target(layout, "invoke", "answer").scope == MemoryScope(
        ("invoke", "answer"), ("outer", "inner", "local")
    )
    for node_id in (START_ID, END_ID):
        assert _target(layout, "invoke", node_id).scope == MemoryScope(
            ("invoke", node_id), ("outer", "inner")
        )
    assert memory_stacks(layout) == tuple(
        MemoryStack(target.locator, target.scope.memories)
        for target in sorted(layout.targets, key=lambda target: target.locator)
    )


def test_memory_flow_without_an_id_resolves() -> None:
    site = DeclSite((), ("root",), "agent")
    agent = AgentNode(
        "agent", memories=(MemoryDecl("reflective", "reflective", {}, site),)
    )
    flow = CompiledFlow({"agent": agent}, [], wiring={"agent": {}})
    assert flow.flow_id is None

    layout = resolve_memory_layout(flow)

    assert layout.enabled is True
    assert [memory.key for memory in layout.memories] == ["reflective"]
    assert _target(layout, "agent").scope == MemoryScope(("agent",), ("reflective",))


def test_one_agent_stacks_several_keys_in_authored_order() -> None:
    site = DeclSite((), ("root",), "agent")
    agent = AgentNode(
        "agent",
        memories=(
            MemoryDecl("reflective", "reflective", {}, site),
            MemoryDecl("episodic", "episodic", {}, site),
        ),
    )
    flow = CompiledFlow({"agent": agent}, [], wiring={"agent": {}})

    layout = resolve_memory_layout(flow)

    assert _target(layout, "agent").scope.memories == ("reflective", "episodic")
    assert [memory.key for memory in layout.memories] == ["episodic", "reflective"]
    agent.memories = (*agent.memories, MemoryDecl("reflective", "reflective", {}, site))
    with pytest.raises(
        MemorySetupError, match="declares memory 'reflective' more than once"
    ):
        resolve_memory_layout(flow)


def test_agent_declaration_must_name_its_own_node() -> None:
    # A programmatic flow could otherwise make two AGENTs share one session owner.
    forged = DeclSite((), ("root",), "a")
    flow = CompiledFlow(
        {
            "a": AgentNode("a", memories=(MemoryDecl("chat", "session", {}, forged),)),
            "b": AgentNode("b", memories=(MemoryDecl("chat", "session", {}, forged),)),
        },
        [],
        wiring={"a": {}, "b": {}},
    )
    with pytest.raises(MemorySetupError, match="AGENT node 'b' memory 'chat' names site node 'a'"):
        resolve_memory_layout(flow)


def test_flow_declaration_must_not_name_a_node() -> None:
    flow = CompiledFlow({"a": AgentNode("a")}, [], wiring={"a": {}})
    flow.memories = (MemoryDecl("notes", "episodic", {}, DeclSite((), ("root",), "a")),)
    with pytest.raises(MemorySetupError, match="flow memory 'notes' names site node 'a'"):
        resolve_memory_layout(flow)


def test_agent_with_two_session_memories_is_a_setup_error() -> None:
    site = DeclSite((), ("root",), "agent")
    agent = AgentNode(
        "agent",
        memories=(
            MemoryDecl("session", "session", {}, site),
            MemoryDecl("chat", "session", {}, site),
        ),
    )
    flow = CompiledFlow({"agent": agent}, [], wiring={"agent": {}})

    with pytest.raises(
        MemorySetupError,
        match="AGENT 'agent' declares more than one session memory: 'chat', 'session'",
    ):
        resolve_memory_layout(flow)


def test_one_def_under_two_parents_has_one_stack_per_placement_and_one_owner() -> None:
    flow = load_flow(
        """id: root
name: root
long_term_memory: outer
defs:
  leaf:
    long_term_memory: leaf_notes
    nodes:
      work:
        kind: code
        code: tests.engine._compose_codefns:echo
        input: {topic: leaf}
        output: str
    output: ${work.output}
  left_parent:
    long_term_memory: left_notes
    nodes:
      leaf_call: {kind: call, call: leaf}
    output: ${leaf_call.output}
  right_parent:
    long_term_memory: right_notes
    nodes:
      leaf_call: {kind: call, call: leaf}
    output: ${leaf_call.output}
nodes:
  left: {kind: call, call: left_parent}
  right: {kind: call, call: right_parent}
output: ${right.output}
"""
    ).compiled

    layout = resolve_memory_layout(flow)
    stacks = {stack.locator: stack.memories for stack in memory_stacks(layout)}

    assert stacks[("left", "leaf_call", "work")] == ("outer", "left_notes", "leaf_notes")
    assert stacks[("right", "leaf_call", "work")] == ("outer", "right_notes", "leaf_notes")
    assert stacks[("left",)] == stacks[("right",)] == ("outer",)
    by_key = {memory.key: memory.spec for memory in layout.memories}
    assert list(by_key) == ["leaf_notes", "left_notes", "outer", "right_notes"]
    assert by_key["leaf_notes"].owners == (DeclSite((), ("def", "leaf"), None),)
    assert by_key["outer"].owners == (DeclSite((), ("root",), None),)
