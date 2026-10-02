"""Derive and atomically stamp source-neutral static memory scopes.

Memory declarations remain authored identity on ``CompiledFlow`` and
``AgentNode``. This module resolves them into a sorted memory key table and
one exact ``MemoryScope`` for every statically reachable node. A scope names
the node's root-relative locator and the keys it recalls and learns into,
outermost first. Resolution is pure; applying a resolved layout is a
separate, all-or-nothing mutation over a private compiled graph.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_composer.compile.model import END_ID, START_ID, CompiledFlow, Flow
from agent_composer.diagnostics import OriginRef
from agent_composer.memory.contracts import (
    DeclSite,
    JsonObject,
    MemoryDecl,
    MemoryOccurrenceSegment,
    MemoryScope,
    MemorySetupError,
    MemorySpec,
    MemoryStack,
)
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.agent.memory import compatibility
from agent_composer.nodes.base import Node
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode


def _revalidate_memory_occurrence_path(
    value: tuple[MemoryOccurrenceSegment, ...],
    *,
    name: str,
) -> tuple[MemoryOccurrenceSegment, ...]:
    """Replay every exact nested invariant of one occurrence path."""
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    for index, segment in enumerate(value):
        item_name = f"{name}[{index}]"
        if type(segment) is not MemoryOccurrenceSegment:
            raise TypeError(
                f"{item_name} must be an exact MemoryOccurrenceSegment"
            )
        MemoryOccurrenceSegment(
            kind=segment.kind,
            static_path=segment.static_path,
            index=segment.index,
        )
    return value


def _revalidate_memory_scope(
    value: MemoryScope,
    *,
    name: str = "memory_scope",
) -> MemoryScope:
    """Replay every exact invariant nested below one frozen memory scope.

    Frozen dataclasses can still be forged through ``object.__setattr__``.
    Reconstructing each layer keeps string subclasses and equality-collapsing
    values such as ``True`` for an occurrence index from crossing the compile
    boundary unnoticed.
    """
    if type(value) is not MemoryScope:
        raise TypeError(f"{name} must be an exact MemoryScope")
    _revalidate_memory_occurrence_path(
        value.occurrence_path,
        name=f"{name}.occurrence_path",
    )
    MemoryScope(
        static_path=value.static_path,
        memories=value.memories,
        occurrence_path=value.occurrence_path,
    )
    return value


@dataclass(frozen=True, slots=True)
class ResolvedMemory:
    """One reachable memory key with its owners and AGENT placements.

    ``placements`` lists the locators of AGENT targets whose own declarations
    include the key; the run-setup session check reads it. ``origins`` aligns
    with ``owners`` and serves diagnostics only.
    """

    key: str
    type: str
    config: JsonObject
    owners: tuple[DeclSite, ...]
    conversation_eligible: bool
    placements: tuple[tuple[str, ...], ...]
    origins: tuple[OriginRef | None, ...] = field(compare=False)

    def __post_init__(self) -> None:
        """Validate through ``MemorySpec`` and pin a private config copy."""
        spec = MemorySpec(
            self.key, self.type, self.config, self.owners, self.conversation_eligible
        )
        object.__setattr__(self, "config", spec.config)
        object.__setattr__(self, "owners", spec.owners)
        if type(self.placements) is not tuple or any(
            type(locator) is not tuple
            or not locator
            or any(type(part) is not str or not part for part in locator)
            for locator in self.placements
        ):
            raise ValueError("placements must be a tuple of non-empty string locators")
        if list(self.placements) != sorted(set(self.placements)):
            raise ValueError("placements must be sorted and unique")
        if type(self.origins) is not tuple or len(self.origins) != len(self.owners):
            raise ValueError("origins must align with owners")
        if any(o is not None and not isinstance(o, OriginRef) for o in self.origins):
            raise TypeError("origins must hold OriginRef values or None")

    @property
    def spec(self) -> MemorySpec:
        """Return the record-facing table entry."""
        return MemorySpec(
            self.key, self.type, self.config, self.owners, self.conversation_eligible
        )

    @property
    def origin(self) -> OriginRef | None:
        """Return the first owner's origin, where key-level errors point."""
        return self.origins[0]


@dataclass(frozen=True, slots=True)
class ResolvedMemoryTarget:
    """One exact static node target and the scope it must receive."""

    locator: tuple[str, ...]
    scope: MemoryScope | None
    node: Node = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the immutable locator, scope, and target class."""
        if type(self.locator) is not tuple or not self.locator:
            raise ValueError("target locator must be a non-empty tuple")
        if any(type(part) is not str or not part for part in self.locator):
            raise TypeError("target locator components must be non-empty strings")
        if self.scope is not None and type(self.scope) is not MemoryScope:
            raise TypeError("target scope must be an exact MemoryScope or None")
        if self.scope is not None:
            _revalidate_memory_scope(self.scope, name="target scope")
        if not isinstance(self.node, Node):
            raise TypeError("target node must be a Node")


@dataclass(frozen=True, slots=True)
class ResolvedMemoryLayout:
    """Immutable static memory layout ready for atomic application.

    ``enabled`` is true when a reachable flow or AGENT declares a memory.
    Then every target has a scope whose ``static_path`` is its locator and
    whose keys are all in ``memories``. A disabled layout has no scopes and
    no memories.
    """

    enabled: bool
    targets: tuple[ResolvedMemoryTarget, ...]
    memories: tuple[ResolvedMemory, ...] = ()

    def __post_init__(self) -> None:
        """Validate exact containers and that every scope agrees with the table."""
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool")
        if type(self.targets) is not tuple or any(
            type(target) is not ResolvedMemoryTarget for target in self.targets
        ):
            raise TypeError("targets must be a tuple of ResolvedMemoryTarget values")
        if type(self.memories) is not tuple or any(
            type(memory) is not ResolvedMemory for memory in self.memories
        ):
            raise TypeError("memories must be a tuple of ResolvedMemory values")
        for target in self.targets:
            ResolvedMemoryTarget(
                locator=target.locator,
                scope=target.scope,
                node=target.node,
            )
        for memory in self.memories:
            ResolvedMemory(
                memory.key,
                memory.type,
                memory.config,
                memory.owners,
                memory.conversation_eligible,
                memory.placements,
                memory.origins,
            )
        keys = [memory.key for memory in self.memories]
        if keys != sorted(set(keys)):
            raise ValueError("memories must be sorted and unique by key")
        if not self.enabled:
            if self.memories:
                raise ValueError("a disabled memory layout cannot contain memories")
            if any(target.scope is not None for target in self.targets):
                raise ValueError("a disabled memory layout cannot contain scopes")
            return
        if not self.memories:
            raise ValueError("an enabled memory layout must contain memories")
        known = set(keys)
        for index, target in enumerate(self.targets):
            if target.scope is None:
                raise ValueError("an enabled memory layout must scope every target")
            if target.scope.static_path != target.locator:
                raise ValueError(
                    f"targets[{index}] scope static_path must equal its locator"
                )
            if not set(target.scope.memories) <= known:
                raise ValueError(
                    f"targets[{index}] scope names a memory outside the table"
                )


def _setup_error(
    message: str,
    *,
    origin: OriginRef | None = None,
) -> MemorySetupError:
    """Build one layout failure with the best available authored origin."""
    return MemorySetupError(message, origin=origin)


def _revalidate_resolved_layout(layout: ResolvedMemoryLayout) -> None:
    """Replay the complete frozen layout contract after possible mutation."""
    if type(layout) is not ResolvedMemoryLayout:
        raise TypeError("layout must be an exact ResolvedMemoryLayout")
    ResolvedMemoryLayout(
        enabled=layout.enabled,
        targets=layout.targets,
        memories=layout.memories,
    )


def _validate_memory_flow_boundaries(
    flow: Flow,
    *,
    decl: MemoryDecl,
) -> None:
    """Require exact direct START/END targets before a flow owns memory."""
    origin = decl.origin
    if type(flow.nodes) is not dict:
        raise _setup_error(
            "a memory-owning flow requires an exact node dictionary",
            origin=origin,
        )
    if type(flow.start_id) is not str or not flow.start_id:
        raise _setup_error(
            "a memory-owning flow start_id must be a non-empty exact string",
            origin=origin,
        )
    if type(flow.end_id) is not str or not flow.end_id:
        raise _setup_error(
            "a memory-owning flow end_id must be a non-empty exact string",
            origin=origin,
        )
    if flow.start_id == flow.end_id:
        raise _setup_error(
            "a memory-owning flow must have distinct START and END boundaries",
            origin=origin,
        )
    if flow.start_id != START_ID:
        raise _setup_error(
            f"a memory-owning flow start_id must be canonical {START_ID!r}",
            origin=origin,
        )
    if flow.end_id != END_ID:
        raise _setup_error(
            f"a memory-owning flow end_id must be canonical {END_ID!r}",
            origin=origin,
        )
    missing = [
        label
        for label, node_id in (("START", flow.start_id), ("END", flow.end_id))
        if node_id not in flow.nodes
    ]
    if missing:
        raise _setup_error(
            "a memory-owning flow is missing direct "
            + " and ".join(missing)
            + " boundary node(s)",
            origin=origin,
        )
    if type(flow.nodes[START_ID]) is not StartNode:
        raise _setup_error(
            "a memory-owning flow START boundary must be an exact StartNode",
            origin=origin,
        )
    if type(flow.nodes[END_ID]) is not EndNode:
        raise _setup_error(
            "a memory-owning flow END boundary must be an exact EndNode",
            origin=origin,
        )


def _owner_decls(
    decls: object,
    *,
    label: str,
    origin: OriginRef | None,
) -> tuple[MemoryDecl, ...]:
    """Return one owner's declarations in authored order, each key once."""
    if type(decls) is not tuple or any(type(decl) is not MemoryDecl for decl in decls):
        raise _setup_error(
            f"{label} memories must be a tuple of exact MemoryDecl values",
            origin=origin,
        )
    seen: set[str] = set()
    for decl in decls:
        if decl.key in seen:
            raise _setup_error(
                f"{label} declares memory {decl.key!r} more than once",
                origin=decl.origin,
            )
        seen.add(decl.key)
    return decls


def _flow_decls(flow: Flow) -> tuple[MemoryDecl, ...]:
    """Return the declarations owned by an exact compiled flow."""
    if type(flow) is not CompiledFlow:
        return ()
    return _owner_decls(flow.memories, label="flow", origin=None)


def _agent_decls(node: Node) -> tuple[MemoryDecl, ...]:
    """Return the declarations owned by an exact AGENT node."""
    if type(node) is not AgentNode:
        return ()
    return _owner_decls(
        node.memories, label=f"AGENT node {node.id!r}", origin=node.origin
    )


def _extend_stack(
    stack: tuple[str, ...],
    decls: tuple[MemoryDecl, ...],
) -> tuple[str, ...]:
    """Append each declared key not already in ``stack``.

    A repeated key keeps its outermost position, so it yields one recall
    section and enters each learning batch once.
    """
    keys = list(stack)
    for decl in decls:
        if decl.key not in keys:
            keys.append(decl.key)
    return tuple(keys)


def _flow_nodes(flow: Flow, *, origin: OriginRef | None = None) -> list[tuple[str, Node]]:
    """Return a validated deterministic view of one flow's direct nodes."""
    if type(flow) not in (Flow, CompiledFlow):
        raise _setup_error(
            f"memory layout expected an exact Flow, got {type(flow).__name__}",
            origin=origin,
        )
    if type(flow.nodes) is not dict:
        raise _setup_error("flow nodes must be an exact dictionary", origin=origin)
    for node_id in flow.nodes:
        if type(node_id) is not str or not node_id:
            raise _setup_error(
                "flow node ids must be non-empty exact strings",
                origin=origin,
            )
    result: list[tuple[str, Node]] = []
    for node_id in sorted(flow.nodes):
        node = flow.nodes[node_id]
        if not isinstance(node, Node):
            raise _setup_error(
                f"flow node {node_id!r} must be a Node",
                origin=origin,
            )
        if type(node.id) is not str or not node.id:
            raise _setup_error(
                f"flow node {node_id!r} has an invalid node id",
                origin=node.origin,
            )
        if node.id != node_id:
            raise _setup_error(
                f"flow node key {node_id!r} does not match node id {node.id!r}",
                origin=node.origin,
            )
        result.append((node_id, node))
    return result


def _node_child(node: Node) -> Flow | None:
    """Return the node's single exact static child, if present."""
    children = node.origin_children()
    if type(children) is not tuple:
        raise _setup_error(
            f"node {node.id!r} static children must be an exact tuple",
            origin=node.origin,
        )
    if len(children) > 1:
        raise _setup_error(
            f"node {node.id!r} exposes more than one static child",
            origin=node.origin,
        )
    if not children:
        return None
    child = children[0]
    if type(child) not in (Flow, CompiledFlow):
        raise _setup_error(
            f"node {node.id!r} static child must be an exact Flow",
            origin=node.origin,
        )
    return child


def _site_label(site: DeclSite) -> str:
    """Render one declaration site for diagnostics."""
    label = "/".join(site.owner)
    return label if site.node is None else f"{label} AGENT {site.node!r}"


def _note(error: BaseException, text: str) -> None:
    """Attach one note; ``BaseException.add_note`` is new in Python 3.11."""
    notes = getattr(error, "__notes__", None)
    if notes is None:
        notes = []
        error.__notes__ = notes
    notes.append(text)


def _canonical_json(value: object) -> str:
    """Encode one JSON value for exact comparison, so ``1``, ``1.0`` and ``true`` differ."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def _memory_table(
    declared: list[tuple[MemoryDecl, tuple[str, ...] | None]],
) -> tuple[ResolvedMemory, ...]:
    """Group reachable declarations by key and enforce the per-key rules.

    Each key has one type and config. A session key has exactly one owner, an
    AGENT, and an AGENT owns at most one session key.
    """
    by_key: dict[str, list[tuple[MemoryDecl, tuple[str, ...] | None]]] = {}
    for decl, placement in declared:
        by_key.setdefault(decl.key, []).append((decl, placement))
    table: list[ResolvedMemory] = []
    for key in sorted(by_key):
        entries = by_key[key]
        # Stable sort: declarations at one site keep walk order.
        declarations = sorted((decl for decl, _ in entries), key=lambda decl: decl.site.sort_key())
        first = declarations[0]
        first_config = _canonical_json(first.config)
        conflicts = [
            decl for decl in declarations[1:]
            if decl.type != first.type or _canonical_json(decl.config) != first_config
        ]
        if conflicts:
            error = _setup_error(
                f"memory {key!r} is declared with different type or config "
                f"at {_site_label(first.site)} and {_site_label(conflicts[0].site)}",
                origin=first.origin,
            )
            for decl in declarations[1:]:
                _note(error, f"also declared at {_site_label(decl.site)}")
            raise error
        by_site: dict[DeclSite, MemoryDecl] = {}
        for decl in declarations:
            by_site.setdefault(decl.site, decl)
        ordered = list(by_site.values())
        if first.type == "session":
            flow_decl = next((d for d in ordered if d.site.node is None), None)
            if flow_decl is not None:
                raise _setup_error(
                    f"session memory {key!r} cannot be declared on a flow "
                    f"({_site_label(flow_decl.site)}); declare it on one AGENT",
                    origin=flow_decl.origin,
                )
            if len(ordered) > 1:
                error = _setup_error(
                    f"session memory {key!r} has more than one owner: "
                    + ", ".join(_site_label(d.site) for d in ordered)
                    + "; give each AGENT its own session memory",
                    origin=ordered[0].origin,
                )
                for decl in ordered[1:]:
                    _note(error, f"also owned at {_site_label(decl.site)}")
                raise error
        owners = tuple(decl.site for decl in ordered)
        table.append(
            ResolvedMemory(
                key=key,
                type=first.type,
                config=first.config,
                owners=owners,
                conversation_eligible=len(owners) == 1 and owners[0].node is not None,
                placements=tuple(
                    sorted({p for _, p in entries if p is not None})
                ),
                origins=tuple(decl.origin for decl in ordered),
            )
        )
    sessions_by_agent: dict[DeclSite, list[ResolvedMemory]] = {}
    for memory in table:
        if memory.type == "session":
            sessions_by_agent.setdefault(memory.owners[0], []).append(memory)
    for site, sessions in sessions_by_agent.items():
        if len(sessions) > 1:
            error = _setup_error(
                f"{_site_label(site)} declares more than one session memory: "
                + ", ".join(repr(memory.key) for memory in sessions),
                origin=sessions[0].origin,
            )
            for memory in sessions[1:]:
                _note(error, f"also declares {memory.key!r}")
            raise error
    return tuple(table)


def memory_stacks(layout: ResolvedMemoryLayout) -> tuple[MemoryStack, ...]:
    """Return every scoped target's key stack, sorted by locator."""
    _revalidate_resolved_layout(layout)
    if not layout.enabled:
        return ()
    return tuple(
        sorted(
            (MemoryStack(target.locator, target.scope.memories) for target in layout.targets),
            key=lambda stack: stack.locator,
        )
    )


def check_session_compatibility(
    layout: ResolvedMemoryLayout,
    effective_llm: Mapping[str, object],
) -> None:
    """Require every placement of a session key to resolve one compatibility.

    ``effective_llm`` is the run's resolved LLM map keyed by ``"/".join``
    of root-relative node ids. Compatibility is provider plus message
    format; model differences are allowed. Every placement must have an
    entry in ``effective_llm``; a missing entry is a ``KeyError``, because
    the cascade and the layout then disagree about the flow.
    """
    _revalidate_resolved_layout(layout)
    for memory in layout.memories:
        if memory.type != "session" or len(memory.placements) < 2:
            continue
        first_path = "/".join(memory.placements[0])
        first = compatibility(effective_llm[first_path])
        for placement in memory.placements[1:]:
            path = "/".join(placement)
            other = compatibility(effective_llm[path])
            if _canonical_json(other) != _canonical_json(first):
                raise _setup_error(
                    f"session memory {memory.key!r} resolves provider "
                    f"{first['provider']!r} at {first_path!r} but "
                    f"{other['provider']!r} at {path!r}; pin a provider on the AGENT",
                    origin=memory.origin,
                )


def _resolve_flow(
    flow: Flow,
    *,
    locator: tuple[str, ...],
    inherited: tuple[str, ...],
    targets: list[ResolvedMemoryTarget],
    declared: list[tuple[MemoryDecl, tuple[str, ...] | None]],
    active_flows: set[int],
    owner_origin: OriginRef | None = None,
) -> None:
    """Resolve one flow occurrence into node scopes and reachable declarations."""
    flow_identity = id(flow)
    if flow_identity in active_flows:
        raise _setup_error(
            "recursive child-flow cycle is not memory-layout resolvable",
            origin=owner_origin,
        )
    active_flows.add(flow_identity)
    try:
        flow_decls = _flow_decls(flow)
        if flow_decls:
            _validate_memory_flow_boundaries(flow, decl=flow_decls[0])
            declared.extend((decl, None) for decl in flow_decls)
        stack = _extend_stack(inherited, flow_decls)

        for node_id, node in _flow_nodes(flow, origin=owner_origin):
            node_locator = (*locator, node_id)
            agent_decls = _agent_decls(node)
            declared.extend((decl, node_locator) for decl in agent_decls)
            targets.append(
                ResolvedMemoryTarget(
                    locator=node_locator,
                    scope=MemoryScope(
                        static_path=node_locator,
                        memories=_extend_stack(stack, agent_decls),
                    ),
                    node=node,
                )
            )
            child = _node_child(node)
            if child is not None:
                _resolve_flow(
                    child,
                    locator=node_locator,
                    inherited=stack,
                    targets=targets,
                    declared=declared,
                    active_flows=active_flows,
                    owner_origin=node.origin,
                )
    finally:
        active_flows.remove(flow_identity)


def resolve_memory_layout(flow: CompiledFlow) -> ResolvedMemoryLayout:
    """Purely derive the memory key table and static node scopes for ``flow``.

    Existing ``Node.memory_scope`` values are ignored. A graph with no
    reachable flow or AGENT declaration resolves to a disabled layout whose
    targets carry no scope. Otherwise every reachable static node receives a
    scope, including memory-neutral composite drivers and flow boundaries.
    No flow id is required.
    """
    if type(flow) is not CompiledFlow:
        raise TypeError("resolve_memory_layout expects an exact CompiledFlow")

    targets: list[ResolvedMemoryTarget] = []
    declared: list[tuple[MemoryDecl, tuple[str, ...] | None]] = []
    _resolve_flow(
        flow,
        locator=(),
        inherited=(),
        targets=targets,
        declared=declared,
        active_flows=set(),
    )
    if not declared:
        return ResolvedMemoryLayout(
            enabled=False,
            targets=tuple(
                ResolvedMemoryTarget(locator=target.locator, scope=None, node=target.node)
                for target in targets
            ),
        )
    return ResolvedMemoryLayout(
        enabled=True,
        targets=tuple(targets),
        memories=_memory_table(declared),
    )


def _validate_unique_targets(layout: ResolvedMemoryLayout) -> None:
    """Reject duplicate locators or aliased node objects before any mutation."""
    locators: set[tuple[str, ...]] = set()
    identities: set[int] = set()
    for target in layout.targets:
        if target.locator in locators:
            raise _setup_error(
                f"memory layout contains duplicate target path {target.locator!r}",
                origin=target.node.origin,
            )
        locators.add(target.locator)
        identity = id(target.node)
        if identity in identities:
            raise _setup_error(
                "memory layout reaches one node object through multiple static paths; "
                "the run setup must de-alias child flows before applying it",
                origin=target.node.origin,
            )
        identities.add(identity)


def apply_memory_layout(
    flow: CompiledFlow,
    layout: ResolvedMemoryLayout,
) -> None:
    """Atomically apply one previously resolved layout to its exact graph.

    The complete graph, semantic paths, the key table, exact node identities,
    duplicate targets, and every existing scope are validated first. Only after
    that pass succeeds are scopes assigned, so a stale or partially stamped graph
    cannot receive a partial layout. Reapplying the same layout is a no-op.
    """
    if type(flow) is not CompiledFlow:
        raise TypeError("apply_memory_layout expects an exact CompiledFlow")
    _revalidate_resolved_layout(layout)

    _validate_unique_targets(layout)
    current = resolve_memory_layout(flow)
    _validate_unique_targets(current)
    if current != layout:
        raise _setup_error("memory layout no longer matches the compiled flow")

    current_by_locator = {target.locator: target for target in current.targets}
    for target in layout.targets:
        actual = current_by_locator[target.locator]
        if actual.node is not target.node:
            raise _setup_error(
                f"memory layout target {target.locator!r} no longer names the same node",
                origin=actual.node.origin,
            )
        existing = actual.node.memory_scope
        if existing is not None:
            try:
                _revalidate_memory_scope(
                    existing,
                    name=f"node at {target.locator!r} memory_scope",
                )
            except (TypeError, ValueError) as exc:
                raise _setup_error(
                    f"node at {target.locator!r} has a noncanonical memory scope",
                    origin=actual.node.origin,
                ) from exc
        if existing is not None and existing != target.scope:
            raise _setup_error(
                f"node at {target.locator!r} already has a different memory scope",
                origin=actual.node.origin,
            )

    for target in layout.targets:
        target.node.memory_scope = target.scope


__all__ = [
    "ResolvedMemory",
    "ResolvedMemoryLayout",
    "ResolvedMemoryTarget",
    "apply_memory_layout",
    "check_session_compatibility",
    "memory_stacks",
    "resolve_memory_layout",
]
