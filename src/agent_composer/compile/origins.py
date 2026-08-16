"""Identity-safe traversal of opaque origins across compiled flow graphs.

The compile layer owns whole-graph shape: flow wiring, declared outputs, node
identity, and recursively baked child flows. Nodes own their local provenance
slots and expose static children through source-neutral hooks. This module joins
those responsibilities without importing Compose, resolving a source, or
inspecting arbitrary semantic payloads.
"""

from __future__ import annotations

import copy
from typing import Callable, Iterator, TypeVar, cast

from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import (
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
    origins in distinct carriers remain distinct occurrences. Binding sources,
    parameter defaults, node environments, and render metadata stay opaque.

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


__all__ = ["iter_flow_origins", "rewrite_flow_origins"]
