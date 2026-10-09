"""Graph-stage: `depends_on:` / `runs_after:` -> run-ordering edges.

Cross-node assembly pass. Each node may declare ordering-only dependencies on other
nodes: `depends_on:` (co-skip — if the source is SKIPPED the dependent dies) and
`runs_after:` (pure order — the dependent waits for the source to SETTLE but survives
its skip). Both lower to zero-data `Edge(ordering=True)` gates; the `optional` flag is
the only difference (`depends_on` -> False, `runs_after` -> True). An unknown target is
a located `LoadError` (at the declaring node's `range`). Cycle rejection is the caller's
job — these edges join the `reject_cycles` set.

Located parse metadata: the unknown-target `LoadError` locates at the declaring node
descriptor's `range`.

Imports flow down/peer only: `compile.model` (Edge), `compose.parse.file` (NodeDescriptor),
`compose.helpers.errors`. Only the loader imports this back.
"""

from __future__ import annotations

from yian.compile.model import Edge
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.parse.file import NodeDescriptor


def infer_ordering_edges(
    descriptors: dict[str, NodeDescriptor],
    node_ids: "set[str]",
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> list[Edge]:
    """Run-ordering edges from `depends_on:` (co-skip) and `runs_after:` (pure order).

    An ordering edge carries no data (`ordering=True`); it gates the dependent on the
    source SETTLING. `depends_on` co-skips (source SKIPPED -> dependent dies, `optional=
    False`); `runs_after` does not (`optional=True`). An unknown target is a located
    `LoadError` (at the declaring node's `range`); cycle rejection is the caller's (the
    edges join the `reject_cycles` set). Distinct `~>` ids keep an ordering edge from
    colliding with a data edge on the same pair.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    edges: list[Edge] = []
    counts: dict[tuple[str, str], int] = {}
    for node_id, desc in descriptors.items():
        for key, sources, optional in (
            ("depends_on", desc.depends_on, False),
            ("runs_after", desc.runs_after, True),
        ):
            for src in sources:
                if src not in node_ids:
                    raise diagnostics.error(
                        f"node {node_id!r}: `{key}` references unknown node {src!r}",
                        range=getattr(desc, "range", None),
                    )
                i = counts.get((src, node_id), 0)
                counts[(src, node_id)] = i + 1
                edges.append(
                    Edge(
                        id=f"{src}~>{node_id}#{i}",
                        from_=src,
                        to=node_id,
                        ordering=True,
                        optional=optional,
                    )
                )
    return edges
