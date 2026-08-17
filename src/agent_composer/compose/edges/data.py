"""Data-edge inference: a data `Edge` per `${<id>.output[.…]}` reference.

The analogue of scope/dependency analysis: a sink binding that reads a free
variable `${<id>.output[.…]}` depends on the producing node `<id>`, so we emit a
data `Edge(from_=<id>, to=<consumer>, input_group=<sink key>)`. `expr_refs_of`
already flattens a coalesce `${a | b}` to BOTH ref paths, so the two alternatives
of one sink share that sink's `input_group` (per-input readiness). Refs whose
head is not `outputs` (`inputs`/`system`/`item`) are NOT data edges.

Boundary assembly owns root/terminal `__start__`/`__end__` synthesis and runs after
`case` control-edge construction; this pass emits only real-node data edges.
"""

from __future__ import annotations

from typing import Any

from agent_composer.compile.model import Edge
from agent_composer.expr import binding_co_skips
from agent_composer.compose.edges.producers import _binding_producers
from agent_composer.compose.parse.file import CaseDescriptor, NodeDescriptor, WaitDescriptor
from agent_composer.nodes.contracts import binding_source


def infer_data_edges(
    descriptors: dict[str, NodeDescriptor],
    flow_wiring: dict[str, dict[str, Any]],
) -> list[Edge]:
    """Infer the flow's data edges from the flow-owned wiring + descriptors (no sentinels).

    For each consuming node, scan its sources and emit a data `Edge` per `${<id>.output[.…]}`
    ref, tagged with the `input_group` that identifies WHICH sink it feeds (the param/arg name
    for a leaf/call input; `over` for a map's `over:`; `until` for WAIT). A coalesce
    sink's alternatives share one group. Edges are the **projection of `flow.wiring`** for
    leaf + WAIT + call/map.

    `flow_wiring[node_id]` is the `{param -> source}` map (WAIT carries its `until`, a
    MAP its `over`, under the reserved keys). A `case` node is SKIPPED here — `CaseDescriptor.build`
    owns the case's data + control edges (returned in its `BuildResult.edges`), so no provisional
    into-case edge is inferred.
    """
    edges: list[Edge] = []
    counts: dict[tuple[str, str], int] = {}  # (from, to) -> next edge index

    def emit(producer: str, consumer: str, group: str, optional: bool) -> None:
        i = counts.get((producer, consumer), 0)
        counts[(producer, consumer)] = i + 1
        edges.append(
            Edge(
                id=f"{producer}->{consumer}#{i}",
                from_=producer,
                to=consumer,
                input_group=group,
                optional=optional,
            )
        )

    def emit_for(value: Any, consumer: str, group: str) -> None:
        # the whole group's co-skip stance is a property of its binding.
        optional = not binding_co_skips(binding_source(value))
        for producer in _binding_producers(value):
            emit(producer, consumer, group, optional)

    for node_id, desc in descriptors.items():
        if isinstance(desc, CaseDescriptor):
            # A `case` owns its edges: `CaseDescriptor.build` returns them (data + control) in
            # its `BuildResult.edges`, so this inference pass emits NOTHING for a case node
            # (no provisional into-case edges to reconcile).
            continue

        if isinstance(desc, WaitDescriptor):
            # WAIT orders after any node its required `until:` reads. The source is relocated
            # to the flow wiring under the reserved key; `${input.X}`/`${system.X}` yield no edge.
            wiring = flow_wiring.get(node_id, {})
            emit_for(wiring["until"], node_id, "until")
            continue

        # leaf + call/map: sources are the flow-owned wiring. A map's
        # reserved `over` source rides the `"over"` key (over-then-inputs order); `${item}` is
        # body-local and yields no edge. CASE/WAIT were handled above.
        for param, src in flow_wiring.get(node_id, {}).items():
            emit_for(src, node_id, param)

    return edges
