"""Derive and atomically apply memory-independent static execution scopes.

This module assigns root-relative authored paths to private compiled node
templates. Resolution is pure; application validates the complete layout and
its current targets before stamping any node. LOOP templates begin at driver
zero; all other dynamic occurrence paths belong to growth preparation.

Knows about: compile.model, nodes.base, nodes.contracts, nodes.start, nodes.end.
Never imports: memory, compose, runtime, or the package-root run boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Generator

from yian.compile.model import CompiledFlow, Flow
from yian.nodes.base import Node
from yian.nodes.contracts import ExecutionPosition, ExecutionScope
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.trampoline import Descend, flatten


def _revalidate_scope(scope: ExecutionScope) -> None:
    """Replay exact nested invariants before accepting an existing frozen scope."""
    if type(scope) is not ExecutionScope:
        raise TypeError("execution scope must be an exact ExecutionScope")
    ExecutionScope(
        static_path=scope.static_path,
        flow_path=scope.flow_path,
        occurrence_path=scope.occurrence_path,
        boundary=scope.boundary,
    )


@dataclass(frozen=True, slots=True)
class ResolvedExecutionTarget:
    """One exact static node target and its root-relative execution scope."""

    locator: tuple[str, ...]
    scope: ExecutionScope
    node: Node = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the immutable locator, static scope, and live target class."""
        if type(self.locator) is not tuple or not self.locator:
            raise ValueError("target locator must be a non-empty tuple")
        if any(type(part) is not str or not part for part in self.locator):
            raise TypeError("target locator components must be non-empty exact strings")
        _revalidate_scope(self.scope)
        if self.scope.static_path != self.locator:
            raise ValueError("target scope static_path must match its locator")
        if self.scope.flow_path != self.locator[:-1]:
            raise ValueError("target scope flow_path must name its containing flow")
        if not isinstance(self.node, Node):
            raise TypeError("target node must be a Node")
        initial = (ExecutionPosition("loop_driver", self.locator, 0),) if self.node.is_loop else ()
        if self.scope.occurrence_path != initial:
            raise ValueError("static execution layout requires only the initial driver position")


@dataclass(frozen=True, slots=True)
class ResolvedExecutionLayout:
    """Immutable static execution targets ready for atomic application."""

    targets: tuple[ResolvedExecutionTarget, ...]

    def __post_init__(self) -> None:
        """Require exact immutable containers and replay every target invariant."""
        if type(self.targets) is not tuple or any(
            type(target) is not ResolvedExecutionTarget for target in self.targets
        ):
            raise TypeError("targets must be a tuple of ResolvedExecutionTarget values")
        for target in self.targets:
            ResolvedExecutionTarget(target.locator, target.scope, target.node)


def _flow_nodes(flow: Flow) -> list[tuple[str, Node]]:
    """Return a validated deterministic view of one flow's direct nodes."""
    if type(flow) not in (Flow, CompiledFlow):
        raise TypeError("execution layout expects an exact Flow or CompiledFlow")
    if type(flow.nodes) is not dict:
        raise TypeError("flow nodes must be an exact dictionary")
    for name in ("start_id", "end_id"):
        value = getattr(flow, name)
        if type(value) is not str or not value:
            raise ValueError(f"flow {name} must be a non-empty exact string")
    for node_id in flow.nodes:
        if type(node_id) is not str or not node_id:
            raise ValueError("flow node ids must be non-empty exact strings")
    result: list[tuple[str, Node]] = []
    for node_id in sorted(flow.nodes):
        node = flow.nodes[node_id]
        if not isinstance(node, Node):
            raise TypeError(f"flow node {node_id!r} must be a Node")
        if type(node.id) is not str or node.id != node_id:
            raise ValueError(f"flow node key {node_id!r} must match its exact node id")
        result.append((node_id, node))
    return result


def _node_child(node: Node) -> Flow | None:
    """Read the existing node-owned static-child hook without kind dispatch."""
    children = node.origin_children()
    if type(children) is not tuple:
        raise TypeError(f"node {node.id!r} static children must be an exact tuple")
    if len(children) > 1:
        raise ValueError(f"node {node.id!r} exposes more than one static child")
    if not children:
        return None
    if type(children[0]) not in (Flow, CompiledFlow):
        raise TypeError(f"node {node.id!r} static child must be an exact Flow")
    return children[0]


def _walk_targets(
    flow: Flow,
    flow_path: tuple[str, ...],
    active_flows: set[int],
) -> Generator[ResolvedExecutionTarget | Descend, None, None]:
    """Traverse every callsite while detecting recursive child-flow cycles.

    A walker for `flatten`: it yields `Descend` with each child flow's walker
    rather than `yield from` it, so nesting doesn't grow the Python stack.
    """
    identity = id(flow)
    if identity in active_flows:
        raise ValueError("recursive child-flow cycle is not execution-layout resolvable")
    active_flows.add(identity)
    try:
        for node_id, node in _flow_nodes(flow):
            locator = (*flow_path, node_id)
            boundary = None
            if node_id == flow.start_id and type(node) is StartNode:
                boundary = "start"
            elif node_id == flow.end_id and type(node) is EndNode:
                boundary = "end"
            yield ResolvedExecutionTarget(
                locator=locator,
                scope=ExecutionScope(
                    locator, flow_path=flow_path, boundary=boundary,
                    occurrence_path=(ExecutionPosition("loop_driver", locator, 0),) if node.is_loop else (),
                ),
                node=node,
            )
            child = _node_child(node)
            if child is not None:
                yield Descend(_walk_targets(child, locator, active_flows))
    finally:
        active_flows.remove(identity)


def resolve_execution_layout(flow: Flow) -> ResolvedExecutionLayout:
    """Purely derive execution scopes without reading memory declarations or flow IDs.

    Args:
        flow (`Flow | CompiledFlow`): The exact static graph to inspect. Missing
            artificial START/END nodes do not prevent layout derivation.

    Returns:
        `ResolvedExecutionLayout`: Deterministic root-relative node targets.

    Raises:
        `TypeError`: If graph or static-child carriers have noncanonical types.
        `ValueError`: If a graph identifier is malformed or static children cycle.
    """
    return ResolvedExecutionLayout(tuple(flatten(_walk_targets(flow, (), set()))))


def _validate_unique_targets(layout: ResolvedExecutionLayout) -> None:
    """Reject duplicate paths or one node object reached through several paths."""
    locators: set[tuple[str, ...]] = set()
    identities: set[int] = set()
    for target in layout.targets:
        if target.locator in locators:
            raise ValueError(f"execution layout has duplicate target path {target.locator!r}")
        locators.add(target.locator)
        if id(target.node) in identities:
            raise ValueError(
                "execution layout reaches one node object through multiple static paths; "
                "use private child templates before applying it"
            )
        identities.add(id(target.node))


def apply_execution_layout(flow: Flow, layout: ResolvedExecutionLayout) -> None:
    """Apply one validated static layout to its exact graph, all or nothing.

    Args:
        flow (`Flow | CompiledFlow`): The private graph used during resolution.
        layout (`ResolvedExecutionLayout`): The layout to revalidate and stamp.
            Reapplying the same layout is idempotent.

    Raises:
        `TypeError`: If any layout, graph, or existing scope is noncanonical.
        `ValueError`: If targets are stale, aliased, or already scoped differently.
    """
    if type(layout) is not ResolvedExecutionLayout:
        raise TypeError("layout must be an exact ResolvedExecutionLayout")
    ResolvedExecutionLayout(layout.targets)
    _validate_unique_targets(layout)
    current = resolve_execution_layout(flow)
    _validate_unique_targets(current)
    if current != layout:
        raise ValueError("execution layout no longer matches the compiled flow")

    for target, actual in zip(layout.targets, current.targets):
        if actual.node is not target.node:
            raise ValueError(
                f"execution layout target {target.locator!r} no longer names the same node"
            )
        existing = actual.node.execution_scope
        if existing is not None:
            _revalidate_scope(existing)
            if existing != target.scope:
                raise ValueError(
                    f"node at {target.locator!r} already has a different execution scope"
                )

    for target in layout.targets:
        target.node.execution_scope = target.scope


__all__ = [
    "ResolvedExecutionLayout",
    "ResolvedExecutionTarget",
    "apply_execution_layout",
    "resolve_execution_layout",
]
