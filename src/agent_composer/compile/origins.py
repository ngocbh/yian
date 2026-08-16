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

    The authored input graph is never mutated. One root `deepcopy` retains shared
    topology and cycles; rewriting then memoizes every frozen carrier replacement,
    including nested contracts. The mapper is called once for each origin reached
    through a unique carrier or node slot. Ordinary payloads stay opaque.

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

    rewritten = copy.deepcopy(flow)
    _rewrite_flow_origin_graph(
        rewritten,
        mapper=mapper,
        seen_flows=set(),
        seen_nodes=set(),
        carrier_memo={},
        output_memo={},
    )
    return cast(_FlowT, rewritten)


def _rewrite_flow_origin_graph(
    flow: Flow,
    *,
    mapper: Callable[[OriginRef], OriginRef],
    seen_flows: set[int],
    seen_nodes: set[int],
    carrier_memo: dict[int, object],
    output_memo: dict[int, FlowOutput],
) -> None:
    """Rewrite one copied flow in place and recurse through unseen children."""
    flow_identity = id(flow)
    if flow_identity in seen_flows:
        return
    seen_flows.add(flow_identity)

    for node_wiring in flow.wiring.values():
        for parameter, binding in node_wiring.items():
            node_wiring[parameter] = _rewrite_origin_carrier(
                binding, mapper, carrier_memo
            )

    if isinstance(flow, CompiledFlow):
        for index, output in enumerate(flow.outputs):
            if not isinstance(output, FlowOutput):
                raise TypeError(
                    "compiled flow outputs must contain FlowOutput values, "
                    f"got {type(output).__name__}"
                )
            output_identity = id(output)
            replacement = output_memo.get(output_identity)
            if replacement is None:
                replacement = FlowOutput(
                    name=output.name,
                    binding=_rewrite_origin_carrier(
                        output.binding, mapper, carrier_memo
                    ),
                )
                output_memo[output_identity] = replacement
                output_memo[id(replacement)] = replacement
            flow.outputs[index] = replacement

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
        node.rewrite_local_origins(mapper, _carrier_memo=carrier_memo)
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
                carrier_memo=carrier_memo,
                output_memo=output_memo,
            )


__all__ = ["iter_flow_origins", "rewrite_flow_origins"]
