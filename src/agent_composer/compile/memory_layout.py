"""Derive and atomically stamp source-neutral static memory scopes.

Memory policy declarations remain authored identity on ``CompiledFlow`` and
``AgentNode``. This module resolves them into immutable frame specifications and
one exact ``MemoryScope`` for every statically reachable node. Resolution is
pure; applying a resolved layout is a separate, all-or-nothing mutation over a
private compiled graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator

from agent_composer.compile.model import END_ID, START_ID, CompiledFlow, Flow
from agent_composer.diagnostics import OriginRef
from agent_composer.memory.contracts import (
    DeclSite,
    JsonObject,
    MemoryDecl,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryOccurrenceSegment,
    MemoryOwner,
    MemoryScope,
    MemorySetupError,
    MemorySpec,
    MemoryStack,
)
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Node
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode


def _revalidate_memory_frame_ref(
    value: MemoryFrameRef,
    *,
    name: str,
) -> MemoryFrameRef:
    """Replay every exact nested invariant of one frozen frame identity."""
    if type(value) is not MemoryFrameRef:
        raise TypeError(f"{name} must be an exact MemoryFrameRef")
    MemoryFrameRef(
        root_flow_id=value.root_flow_id,
        static_path=value.static_path,
        owner=value.owner,
    )
    return value


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

    Frozen dataclasses can still be forged through ``object.__setattr__``. This
    validator reconstructs each carrier layer so string/enum subclasses and
    equality-collapsing values such as ``True`` for an occurrence index cannot
    cross the compile boundary unnoticed.
    """
    if type(value) is not MemoryScope:
        raise TypeError(f"{name} must be an exact MemoryScope")
    if type(value.frame_stack) is not tuple:
        raise TypeError(f"{name}.frame_stack must be an exact tuple")
    for index, frame in enumerate(value.frame_stack):
        _revalidate_memory_frame_ref(
            frame,
            name=f"{name}.frame_stack[{index}]",
        )
    _revalidate_memory_occurrence_path(
        value.occurrence_path,
        name=f"{name}.occurrence_path",
    )
    MemoryScope(
        root_flow_id=value.root_flow_id,
        static_path=value.static_path,
        frame_stack=value.frame_stack,
        occurrence_path=value.occurrence_path,
    )
    return value


def _revalidate_memory_frame_spec(
    value: MemoryFrameSpec,
    *,
    name: str,
) -> MemoryFrameSpec:
    """Replay one effective frame spec including its frozen config snapshot."""
    if type(value) is not MemoryFrameSpec:
        raise TypeError(f"{name} must be an exact MemoryFrameSpec")
    _revalidate_memory_frame_ref(value.ref, name=f"{name}.ref")
    MemoryFrameSpec(
        ref=value.ref,
        policy_name=value.policy_name,
        config=value.config,
    )
    return value


@dataclass(frozen=True, slots=True)
class ResolvedMemoryFrame:
    """One effective frame specification plus its diagnostic policy site."""

    spec: MemoryFrameSpec
    origin: OriginRef | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Require exact source-neutral carriers."""
        _revalidate_memory_frame_spec(self.spec, name="spec")
        if self.origin is not None:
            if not isinstance(self.origin, OriginRef):
                raise TypeError("origin must be an OriginRef or None")
            OriginRef(self.origin.key)


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
    """Immutable static memory layout ready for atomic application."""

    root_flow_id: str | None
    frames: tuple[ResolvedMemoryFrame, ...]
    targets: tuple[ResolvedMemoryTarget, ...]
    memories: tuple[ResolvedMemory, ...] = ()

    def __post_init__(self) -> None:
        """Validate exact immutable layout containers."""
        if self.root_flow_id is not None and (
            type(self.root_flow_id) is not str or not self.root_flow_id
        ):
            raise ValueError("root_flow_id must be a non-empty exact string or None")
        if type(self.frames) is not tuple or any(
            type(frame) is not ResolvedMemoryFrame for frame in self.frames
        ):
            raise TypeError("frames must be a tuple of ResolvedMemoryFrame values")
        if type(self.targets) is not tuple or any(
            type(target) is not ResolvedMemoryTarget for target in self.targets
        ):
            raise TypeError("targets must be a tuple of ResolvedMemoryTarget values")
        for frame in self.frames:
            ResolvedMemoryFrame(spec=frame.spec, origin=frame.origin)
        for target in self.targets:
            ResolvedMemoryTarget(
                locator=target.locator,
                scope=target.scope,
                node=target.node,
            )
        if self.root_flow_id is None and self.frames:
            raise ValueError("a disabled memory layout cannot contain frames")
        for target in self.targets:
            if self.root_flow_id is None:
                if target.scope is not None:
                    raise ValueError("a disabled memory layout cannot contain scopes")
            elif target.scope is None:
                raise ValueError("an enabled memory layout must scope every target")
        if self.root_flow_id is not None:
            for index, frame in enumerate(self.frames):
                if frame.spec.ref.root_flow_id != self.root_flow_id:
                    raise ValueError(
                        f"frames[{index}] root_flow_id must match the layout"
                    )
            for index, target in enumerate(self.targets):
                if target.scope.root_flow_id != self.root_flow_id:
                    raise ValueError(
                        f"targets[{index}] scope root_flow_id must match the layout"
                    )
        if type(self.memories) is not tuple or any(
            type(memory) is not ResolvedMemory for memory in self.memories
        ):
            raise TypeError("memories must be a tuple of ResolvedMemory values")
        keys = [memory.key for memory in self.memories]
        if keys != sorted(set(keys)):
            raise ValueError("memories must be sorted and unique by key")
        if self.root_flow_id is None and self.memories:
            raise ValueError("a disabled memory layout cannot contain memories")

    @property
    def enabled(self) -> bool:
        """Return whether at least one reachable memory policy enabled layout."""
        return self.root_flow_id is not None


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
        root_flow_id=layout.root_flow_id,
        frames=layout.frames,
        targets=layout.targets,
        memories=layout.memories,
    )


def _single_decl(
    decls: object,
    *,
    label: str,
    origin: OriginRef | None,
) -> MemoryDecl | None:
    """Return the owner's only declaration while frames still key layout."""
    if type(decls) is not tuple or any(type(d) is not MemoryDecl for d in decls):
        raise _setup_error(f"{label} memories must be a tuple of exact MemoryDecl values", origin=origin)
    if len(decls) > 1:
        raise _setup_error(f"{label} declares more than one memory", origin=decls[1].origin)
    return decls[0] if decls else None


def _flow_policy(flow: Flow) -> MemoryDecl | None:
    """Return the optional declaration owned by an exact compiled flow."""
    if type(flow) is not CompiledFlow:
        return None
    return _single_decl(flow.memories, label="flow", origin=None)


def _validate_memory_flow_boundaries(
    flow: Flow,
    *,
    policy: MemoryDecl,
) -> None:
    """Require exact direct START/END targets before emitting a flow frame."""
    origin = policy.origin
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


def _agent_policy(node: Node) -> MemoryDecl | None:
    """Return the optional declaration owned by an exact AGENT node."""
    if type(node) is not AgentNode:
        return None
    return _single_decl(node.memories, label=f"AGENT node {node.id!r}", origin=node.origin)


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


def _walk_targets(
    flow: Flow,
    *,
    locator: tuple[str, ...] = (),
    active_flows: set[int] | None = None,
    owner_origin: OriginRef | None = None,
) -> Iterator[tuple[tuple[str, ...], Node]]:
    """Yield every reachable static node while allowing non-recursive aliases."""
    active = set() if active_flows is None else active_flows
    flow_identity = id(flow)
    if flow_identity in active:
        raise _setup_error(
            "recursive child-flow cycle is not memory-layout resolvable",
            origin=owner_origin,
        )
    active.add(flow_identity)
    try:
        _flow_policy(flow)
        for node_id, node in _flow_nodes(flow, origin=owner_origin):
            node_locator = (*locator, node_id)
            _agent_policy(node)
            yield node_locator, node
            child = _node_child(node)
            if child is not None:
                yield from _walk_targets(
                    child,
                    locator=node_locator,
                    active_flows=active,
                    owner_origin=node.origin,
                )
    finally:
        active.remove(flow_identity)


def _first_policy(
    flow: Flow,
    *,
    active_flows: set[int] | None = None,
    owner_origin: OriginRef | None = None,
) -> MemoryDecl | None:
    """Return the first reachable policy while validating the static child tree."""
    active = set() if active_flows is None else active_flows
    flow_identity = id(flow)
    if flow_identity in active:
        raise _setup_error(
            "recursive child-flow cycle is not memory-layout resolvable",
            origin=owner_origin,
        )
    active.add(flow_identity)
    try:
        policy = _flow_policy(flow)
        if policy is not None:
            return policy
        for _, node in _flow_nodes(flow, origin=owner_origin):
            policy = _agent_policy(node)
            if policy is not None:
                return policy
            child = _node_child(node)
            if child is not None:
                policy = _first_policy(
                    child,
                    active_flows=active,
                    owner_origin=node.origin,
                )
                if policy is not None:
                    return policy
        return None
    finally:
        active.remove(flow_identity)


def _resolved_frame(
    policy: MemoryDecl,
    ref: MemoryFrameRef,
) -> ResolvedMemoryFrame:
    """Lower one authored policy into its effective source-neutral frame spec."""
    return ResolvedMemoryFrame(
        spec=MemoryFrameSpec(ref, policy.key, policy.config),
        origin=policy.origin,
    )


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
    """Group reachable declarations by key and enforce one type and config."""
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
    return tuple(table)


def memory_stacks(layout: ResolvedMemoryLayout) -> tuple[MemoryStack, ...]:
    """Return every scoped target's key stack, sorted by locator."""
    _revalidate_resolved_layout(layout)
    if not layout.enabled:
        return ()
    key_by_ref = {frame.spec.ref: frame.spec.policy_name for frame in layout.frames}
    stacks = []
    for target in layout.targets:
        keys: list[str] = []
        for ref in target.scope.frame_stack:
            key = key_by_ref[ref]
            if key not in keys:
                keys.append(key)
        stacks.append(MemoryStack(target.locator, tuple(keys)))
    return tuple(sorted(stacks, key=lambda stack: stack.locator))


def _resolve_enabled_flow(
    flow: Flow,
    *,
    root_flow_id: str,
    flow_path: tuple[str, ...],
    locator: tuple[str, ...],
    inherited_frames: tuple[MemoryFrameRef, ...],
    frames: list[ResolvedMemoryFrame],
    targets: list[ResolvedMemoryTarget],
    declared: list[tuple[MemoryDecl, tuple[str, ...] | None]],
    active_flows: set[int],
    owner_origin: OriginRef | None = None,
) -> None:
    """Resolve one enabled flow occurrence into ordered frames and node scopes."""
    flow_identity = id(flow)
    if flow_identity in active_flows:
        raise _setup_error(
            "recursive child-flow cycle is not memory-layout resolvable",
            origin=owner_origin,
        )
    active_flows.add(flow_identity)
    try:
        flow_policy = _flow_policy(flow)
        frame_stack = inherited_frames
        if flow_policy is not None:
            _validate_memory_flow_boundaries(flow, policy=flow_policy)
            flow_frame = MemoryFrameRef(
                root_flow_id=root_flow_id,
                static_path=flow_path,
                owner=MemoryOwner.FLOW,
            )
            frames.append(_resolved_frame(flow_policy, flow_frame))
            declared.append((flow_policy, None))
            frame_stack = (*frame_stack, flow_frame)

        for node_id, node in _flow_nodes(flow, origin=owner_origin):
            node_locator = (*locator, node_id)
            node_path = (*flow_path, node_id)
            node_frames = frame_stack
            agent_policy = _agent_policy(node)
            if agent_policy is not None:
                agent_frame = MemoryFrameRef(
                    root_flow_id=root_flow_id,
                    static_path=node_path,
                    owner=MemoryOwner.AGENT,
                )
                frames.append(_resolved_frame(agent_policy, agent_frame))
                declared.append((agent_policy, node_locator))
                node_frames = (*node_frames, agent_frame)

            targets.append(
                ResolvedMemoryTarget(
                    locator=node_locator,
                    scope=MemoryScope(
                        root_flow_id=root_flow_id,
                        static_path=node_path,
                        frame_stack=node_frames,
                    ),
                    node=node,
                )
            )

            child = _node_child(node)
            if child is not None:
                _resolve_enabled_flow(
                    child,
                    root_flow_id=root_flow_id,
                    flow_path=node_path,
                    locator=node_locator,
                    inherited_frames=frame_stack,
                    frames=frames,
                    targets=targets,
                    declared=declared,
                    active_flows=active_flows,
                    owner_origin=node.origin,
                )
    finally:
        active_flows.remove(flow_identity)


def resolve_memory_layout(flow: CompiledFlow) -> ResolvedMemoryLayout:
    """Purely derive frame specs and exact static node scopes for ``flow``.

    Existing ``Node.memory_scope`` values are deliberately ignored. A graph with
    no reachable flow- or AGENT-owned policy resolves to a disabled layout and
    does not require a root flow id. Once any policy is reachable, the root id is
    mandatory and every reachable static node receives a scope, including
    memory-neutral composite drivers and flow boundaries.
    """
    if type(flow) is not CompiledFlow:
        raise TypeError("resolve_memory_layout expects an exact CompiledFlow")

    first_policy = _first_policy(flow)
    if first_policy is None:
        return ResolvedMemoryLayout(
            root_flow_id=None,
            frames=(),
            targets=tuple(
                ResolvedMemoryTarget(locator=locator, scope=None, node=node)
                for locator, node in _walk_targets(flow)
            ),
        )

    root_flow_id = flow.flow_id
    if type(root_flow_id) is not str or not root_flow_id:
        raise _setup_error(
            "a flow with reachable memory policies requires a non-empty root flow_id",
            origin=first_policy.origin,
        )

    frames: list[ResolvedMemoryFrame] = []
    targets: list[ResolvedMemoryTarget] = []
    declared: list[tuple[MemoryDecl, tuple[str, ...] | None]] = []
    _resolve_enabled_flow(
        flow,
        root_flow_id=root_flow_id,
        flow_path=(root_flow_id,),
        locator=(),
        inherited_frames=(),
        frames=frames,
        targets=targets,
        declared=declared,
        active_flows=set(),
    )
    return ResolvedMemoryLayout(
        root_flow_id=root_flow_id,
        frames=tuple(frames),
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

    The complete graph, semantic paths, frame specs, exact node identities,
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
    "ResolvedMemoryFrame",
    "ResolvedMemoryLayout",
    "ResolvedMemoryTarget",
    "apply_memory_layout",
    "memory_stacks",
    "resolve_memory_layout",
]
