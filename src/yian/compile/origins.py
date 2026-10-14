"""Identity-safe traversal and rebasing across compiled origin graphs.

The compile layer owns whole-graph shape: flow wiring, declared outputs, node
identity, and recursively baked child flows. Nodes own their local provenance
slots and expose static children through source-neutral hooks. This module joins
those responsibilities without importing Compose, resolving a source, or
inspecting arbitrary semantic payloads.
"""

from __future__ import annotations

import copy
from typing import Callable, Iterator, TypeVar, cast

from yian.diagnostics import OriginRef, OriginResolver, ResolverTable
from yian.memory.contracts import (
    _iter_memory_decl_origins,
    _rewrite_memory_decl_origins,
)
from yian.nodes.base import Node
from yian.nodes.contracts import (
    _iter_origin_carrier,
    _rewrite_origin_carrier,
)

from .model import CompiledFlow, Flow, FlowOutput


_FlowT = TypeVar("_FlowT", bound=Flow)


def iter_flow_origins(flow: Flow) -> Iterator[OriginRef]:
    """Iterate declared provenance slots in one static flow graph.

    Traversal follows wiring and outputs, then nodes and their baked children in
    deterministic structural order. Flow, node, and frozen-carrier identities are
    visited once, so shared children and deliberate object cycles terminate. Equal
    origins in distinct carriers remain distinct occurrences. Binding sources
    other than value templates, parameter defaults, node environments, and render
    metadata stay opaque.

    Args:
        flow (`Flow`):
            Compiled topology or compile-owned fragment to traverse.

    Returns:
        `Iterator[OriginRef]`:
            Present semantic origins. Source-free graphs produce an empty iterator.

    Raises:
        TypeError:
            Raised when `flow`, a node, or a non-empty static child is not a
            compiled graph value.
    """
    if not isinstance(flow, Flow):
        raise TypeError(f"flow must be a Flow, got {type(flow).__name__}")
    return _iter_flow_origins(
        flow,
        seen_flows=set(),
        seen_nodes=set(),
        seen_carriers=set(),
    )


def _iter_flow_origins(
    flow: Flow,
    *,
    seen_flows: set[int],
    seen_nodes: set[int],
    seen_carriers: set[int],
) -> Iterator[OriginRef]:
    """Yield one flow identity and recursively visit each unseen static child."""
    flow_identity = id(flow)
    if flow_identity in seen_flows:
        return
    seen_flows.add(flow_identity)

    if isinstance(flow, CompiledFlow):
        yield from _iter_memory_decl_origins(flow.memories, seen_carriers)

    for node_wiring in flow.wiring.values():
        for binding in node_wiring.values():
            yield from _iter_origin_carrier(binding, seen_carriers)

    if isinstance(flow, CompiledFlow):
        for output in flow.outputs:
            if not isinstance(output, FlowOutput):
                raise TypeError(
                    "compiled flow outputs must contain FlowOutput values, "
                    f"got {type(output).__name__}"
                )
            yield from _iter_origin_carrier(output.binding, seen_carriers)

    for node in flow.nodes.values():
        if not isinstance(node, Node):
            raise TypeError(
                "flow nodes must contain Node values, "
                f"got {type(node).__name__}"
            )
        node_identity = id(node)
        if node_identity in seen_nodes:
            continue
        seen_nodes.add(node_identity)
        yield from node.iter_local_origins(_seen_carriers=seen_carriers)
        for child in node.origin_children():
            if not isinstance(child, Flow):
                raise TypeError(
                    f"node {node.id!r} origin child must be a Flow, "
                    f"got {type(child).__name__}"
                )
            yield from _iter_flow_origins(
                child,
                seen_flows=seen_flows,
                seen_nodes=seen_nodes,
                seen_carriers=seen_carriers,
            )


def rewrite_flow_origins(
    flow: _FlowT,
    mapper: Callable[[OriginRef], OriginRef],
) -> _FlowT:
    """Return an identity-preserving deep copy with every origin transformed.

    The authored input graph is never mutated. An isolated working copy retains
    shared topology and cycles while provenance fields are rewritten once per
    frozen carrier identity. A final copy rebuilds hash containers after those
    frozen values change. The mapper is still called once for each origin reached
    through a unique carrier or node slot. Ordinary payloads stay opaque,
    including back-references to graph objects.

    Args:
        flow (`Flow`):
            Compiled topology or compile-owned fragment to copy and rewrite.
        mapper (`Callable[[OriginRef], OriginRef]`):
            Transform for every present semantic origin. It must return an
            `OriginRef`; it may raise to abort the rewrite.

    Returns:
        `Flow`:
            A graph of the same concrete flow type with preserved shared-object
            topology and rewritten origins.

    Raises:
        TypeError:
            Raised when `flow` is not a `Flow`, `mapper` is not callable, a
            mapper result is not an `OriginRef`, or a static child is invalid.
    """
    if not isinstance(flow, Flow):
        raise TypeError(f"flow must be a Flow, got {type(flow).__name__}")
    if not callable(mapper):
        raise TypeError(f"origin mapper must be callable, got {type(mapper).__name__}")

    # Validate static shape before copying so an invalid child cannot run an
    # arbitrary payload's __deepcopy__ ahead of the structural TypeError.
    for _origin in iter_flow_origins(flow):
        pass

    working = copy.deepcopy(flow)
    _rewrite_flow_origin_graph(
        working,
        mapper=mapper,
        seen_flows=set(),
        seen_nodes=set(),
        seen_carriers=set(),
    )
    # Frozen carriers may also be keys or set members inside opaque payloads.
    # Re-copying the finalized working graph rebuilds those hash containers with
    # mapped hashes while preserving every alias and cycle.
    return cast(_FlowT, copy.deepcopy(working))


def rebase_flow_origins(
    flow: _FlowT,
    mapper: Callable[[OriginRef], OriginRef],
    *,
    resolver: OriginResolver | None,
) -> tuple[_FlowT, ResolverTable]:
    """Atomically rewrite reachable origins and retain their resolution routes.

    Equal old keys are one namespace identity even when their `OriginRef` values
    have distinct Python classes, so `mapper` runs once per distinct opaque key.
    The returned table routes each mapped identity back through `resolver` with
    its exact old identity. Keys remain opaque to this operation.

    Args:
        flow (`Flow`): Compiled topology or compile-owned fragment to copy.
        mapper (`Callable[[OriginRef], OriginRef]`):
            Namespace transform called once per distinct reachable old key.
        resolver (`OriginResolver | None`):
            Authority for the old identities. Required for an originful graph and
            required to be `None` for a source-free graph.

    Returns:
        `tuple[Flow, ResolverTable]`:
            Independent graph copy of the same concrete type and exact routes for
            all of its mapped origins. A source-free graph returns an empty table.

    Raises:
        TypeError:
            Raised when the flow, mapper, resolver, mapper result, or static graph
            shape is invalid.
        ValueError:
            Raised when an origin subclass carries an invalid key, resolver
            presence disagrees with graph provenance, or distinct old keys map
            to the same new key.
    """
    if not isinstance(flow, Flow):
        raise TypeError(f"flow must be a Flow, got {type(flow).__name__}")
    if not callable(mapper):
        raise TypeError(f"origin mapper must be callable, got {type(mapper).__name__}")
    if resolver is not None and not callable(getattr(resolver, "resolve", None)):
        raise TypeError("origin resolver must implement resolve")

    # Materializing validates the entire static graph before a user mapper can
    # have side effects. Dict insertion order retains the first structural visit,
    # while the single-key origin contract stays independent of Python subclasses.
    old_by_key: dict[str, OriginRef] = {}
    for old_origin in iter_flow_origins(flow):
        old_key = OriginRef(old_origin.key).key
        old_by_key.setdefault(old_key, old_origin)
    old_origins = tuple(old_by_key.values())
    if old_origins and resolver is None:
        raise ValueError("originful flow requires an origin resolver")
    if not old_origins and resolver is not None:
        raise ValueError("source-free flow cannot have an origin resolver")

    mapped_by_old_key: dict[str, OriginRef] = {}
    mapped_keys: set[str] = set()
    for old_origin in old_origins:
        mapped_origin = mapper(old_origin)
        if not isinstance(mapped_origin, OriginRef):
            raise TypeError(
                "origin mapper must return an OriginRef, "
                f"got {type(mapped_origin).__name__}"
            )
        new_origin = OriginRef(mapped_origin.key)
        if new_origin.key in mapped_keys:
            raise ValueError(f"mapped origin collision {new_origin.key!r}")
        mapped_by_old_key[old_origin.key] = new_origin
        mapped_keys.add(new_origin.key)

    table = (
        ResolverTable()
        if resolver is None
        else ResolverTable.from_entries(
            (mapped_by_old_key[old_origin.key], resolver, old_origin)
            for old_origin in old_origins
        )
    )
    rewritten = rewrite_flow_origins(
        flow,
        lambda origin: mapped_by_old_key[origin.key],
    )
    return rewritten, table


def _rewrite_flow_origin_graph(
    flow: Flow,
    *,
    mapper: Callable[[OriginRef], OriginRef],
    seen_flows: set[int],
    seen_nodes: set[int],
    seen_carriers: set[int],
) -> None:
    """Rewrite one copied flow in place and recurse through unseen children."""
    flow_identity = id(flow)
    if flow_identity in seen_flows:
        return
    seen_flows.add(flow_identity)

    if isinstance(flow, CompiledFlow):
        _rewrite_memory_decl_origins(flow.memories, mapper, seen_carriers)

    for node_wiring in flow.wiring.values():
        for binding in node_wiring.values():
            _rewrite_origin_carrier(binding, mapper, seen_carriers)

    if isinstance(flow, CompiledFlow):
        for output in flow.outputs:
            if not isinstance(output, FlowOutput):
                raise TypeError(
                    "compiled flow outputs must contain FlowOutput values, "
                    f"got {type(output).__name__}"
                )
            _rewrite_origin_carrier(output.binding, mapper, seen_carriers)

    for node in flow.nodes.values():
        if not isinstance(node, Node):
            raise TypeError(
                "flow nodes must contain Node values, "
                f"got {type(node).__name__}"
            )
        node_identity = id(node)
        if node_identity in seen_nodes:
            continue
        seen_nodes.add(node_identity)
        node.rewrite_local_origins(mapper, _seen_carriers=seen_carriers)
        for child in node.origin_children():
            if not isinstance(child, Flow):
                raise TypeError(
                    f"node {node.id!r} origin child must be a Flow, "
                    f"got {type(child).__name__}"
                )
            _rewrite_flow_origin_graph(
                child,
                mapper=mapper,
                seen_flows=seen_flows,
                seen_nodes=seen_nodes,
                seen_carriers=seen_carriers,
            )


__all__ = [
    "iter_flow_origins",
    "rebase_flow_origins",
    "rewrite_flow_origins",
]
