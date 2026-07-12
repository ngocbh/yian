"""Data-edge inference: a data `Edge` per `${<id>.output[.…]}` reference.

The analogue of scope/dependency analysis: a sink binding that reads a free
variable `${<id>.output[.…]}` depends on the producing node `<id>`, so we emit a
data `Edge(from_=<id>, to=<consumer>, input_group=<sink key>)`. `expr_refs_of`
already flattens a coalesce `${a | b}` to BOTH ref paths, so the two alternatives
of one sink share that sink's `input_group` (per-input readiness). Refs whose
head is not `outputs` (`inputs`/`system`/`item`) are NOT data edges.

Root/terminal `__start__`/`__end__` synthesis is a later pass (it must run after the
`case` control edges): this pass emits ONLY real-node data edges.
"""

from __future__ import annotations

from typing import Any

from agent_composer.compile.model import Edge
from agent_composer.expr import binding_co_skips
from agent_composer.compose.edges.producers import _binding_producers, _condition_producers
from agent_composer.compose.parse.file import CaseDescriptor, NodeDescriptor, WaitDescriptor


def infer_data_edges(
    descriptors: dict[str, NodeDescriptor],
    flow_wiring: dict[str, dict[str, Any]],
) -> list[Edge]:
    """Infer the flow's data edges from the flow-owned wiring + descriptors (no sentinels).

    For each consuming node, scan its sources and emit a data `Edge` per `${<id>.output[.…]}`
    ref, tagged with the `input_group` that identifies WHICH sink it feeds (the param/arg name
    for a leaf/REF input; `over` for a MAP's `over:`; `until` for a timed WAIT; a provisional
    `<case>:<n>` key for a `case` `on:`/`when:` ref). A coalesce sink's alternatives share one
    group. Edges are the **projection of `flow.wiring`** for leaf + WAIT + REF/MAP;
    CASE sources are still read off `descriptors`.

    `flow_wiring[node_id]` is the `{param -> source}` map (a timed WAIT carries its `until`, a
    MAP its `over`, under the reserved keys). `descriptors` supplies the CASE `on:`/`when:` refs
    (a `case` desugars to an `CaseNode`, which reconciles these provisional groups
    to its `__rN`/`__on` names).
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
        optional = not binding_co_skips(value)
        for producer in _binding_producers(value):
            emit(producer, consumer, group, optional)

    for node_id, desc in descriptors.items():
        if isinstance(desc, CaseDescriptor):
            # case has no built node (the CaseNode is built later). Its data refs
            # live in `on:` + each `when:`. Provisional per-ref groups (`<case>:<n>`)
            # — the case desugar reconciles to `__on`/`__rN`. The edge EXISTENCE is the
            # invariant this pass must satisfy (so `score -> gate` is inferred).
            ref_index = 0
            sources = [desc.on] if desc.on is not None else []
            sources += [c.get("when") for c in desc.cases]
            for src in sources:
                for producer in _condition_producers(src):
                    # provisional case edges (reconciliation drops them); optional
                    # is recomputed there from the desugared __rN/__on binding.
                    emit(producer, node_id, f"{node_id}:{ref_index}", False)
                    ref_index += 1
            continue

        if isinstance(desc, WaitDescriptor):
            # A timed `wait` orders after any node its `until:` reads. The `until` SOURCE is
            # relocated to the flow wiring (reserved key); an event WAIT has no `until`
            # key. `${input.X}`/`${system.X}` yield no edge.
            wiring = flow_wiring.get(node_id, {})
            if "until" in wiring:
                emit_for(wiring["until"], node_id, "until")
            continue

        # leaf + REF/MAP: sources are the flow-owned wiring. A MAP's
        # reserved `over` source rides the `"over"` key (over-then-inputs order); `${item}` is
        # body-local and yields no edge. CASE/WAIT were handled above.
        for param, src in flow_wiring.get(node_id, {}).items():
            emit_for(src, node_id, param)

    return edges
