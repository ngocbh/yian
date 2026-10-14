"""Graph-stage: stamp each TOOL node with the inferred types of its input bindings.

A TOOL node hands its bound input record to a host tool whose declared input types
are known only when the run supplies its tools. This load pass records, on the node,
the type each input binding is proven to produce, so that the run-start tool check
can compare them with the tool's declared input types without the authoring source.

Each binding is inferred over the producer and flow-input contracts as narrowed by
the CASE branches that route the node; an unproven binding is left out. A TOOL has
no item scope (`${item}` is legal only in a MAP's own `input:`), so none is used.

Imports flow down only: `nodes` (the TOOL node), `compose.nodes` (build results), and
the sibling narrowing view and typed-binding inference. Only the loader (and the compose
package facade) import this back.
"""

from __future__ import annotations

from typing import Any

from yian.compose.helpers.output_contracts import TypeFn
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.narrowing import Narrowing
from yian.compose.validate.typed_bindings import infer_binding_type
from yian.nodes.contracts import OutputContract, ValueContract
from yian.nodes.tool.node import ToolNode


def stamp_tool_input_types(
    flow_wiring: dict[str, dict[str, Any]],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    build_results: dict[str, BuildResult],
    *,
    narrowing: Narrowing | None = None,
    type_fn: TypeFn | None = None,
) -> None:
    """Set each TOOL node's `input_types` to its bindings' proven inferred types.

    Args:
        flow_wiring (`dict[str, dict[str, Any]]`): Final binding maps keyed by node id
            and parameter name.
        producers (`dict[str, OutputContract]`): Exact or partial node output
            contracts keyed by producer id.
        flow_inputs (`dict[str, ValueContract]`): Exact flow input contracts keyed by
            input name.
        build_results (`dict[str, BuildResult]`): Built nodes keyed by node id.
        narrowing (`Narrowing | None`, *optional*, defaults to `None`):
            The union narrowing each param reads against, from the CASE branches
            that route its node; `None` reads every param against the unchanged maps.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`): Resolves a
            constructor name to its type.
    """
    for node_id, result in build_results.items():
        node = result.node
        if not isinstance(node, ToolNode):
            continue
        input_types = {}
        for name, binding in flow_wiring.get(node_id, {}).items():
            param_producers, param_inputs = (
                narrowing.for_param(node_id, name)
                if narrowing is not None
                else (producers, flow_inputs)
            )
            inferred = infer_binding_type(
                binding, param_producers, param_inputs, type_fn=type_fn
            )
            if inferred is not None:
                input_types[name] = inferred
        node.input_types = input_types
