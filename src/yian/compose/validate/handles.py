"""Validate: a `CaseNode`'s handles align with its outgoing control edges.

The handle-alignment check ports `compile.validation._check_case_handles`'s rule
to the desugared `CaseNode`s + their control edges (the Compose path has no
`FlowSpec`). Runs in the validate phase over the built node map + derived edges.

Imports flow DOWN only: `compile.model` (Edge), `nodes.base` (Node) + `nodes.case`
(CaseNode/DEFAULT_HANDLE), `compose.helpers.errors` (LoadError). Nothing in
helpers/nodes/parse/desugar/edges imports this back.
"""

from __future__ import annotations

from yian.compile.model import Edge
from yian.nodes.base import Node
from yian.nodes.case import DEFAULT_HANDLE, CaseNode
from yian.compose.helpers.errors import LoadDiagnostics


def check_case_handles(
    nodes: dict[str, Node],
    edges: list[Edge],
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Every desugared `CaseNode`'s handles align with its outgoing control edges.

    Each outgoing control edge's handle must be a declared case handle or `default`, and
    every declared case must have an outgoing edge. A handle-less edge (`depends_on`,
    `runs_after`, a data read) is not a branch, so it is not checked. The `case` desugar emits the
    `gate -> <then|else>` control edges and the `CaseNode.cases`; this pins them
    consistent. Raises `LoadError` on the first misalignment.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for node_id, node in nodes.items():
        if not isinstance(node, CaseNode):
            continue
        case_handles = {c.handle for c in node.cases}
        out_handles = {
            e.source_handle
            for e in edges
            if e.from_ == node_id and e.source_handle is not None
        }
        for handle in out_handles:
            if handle != DEFAULT_HANDLE and handle not in case_handles:
                raise diagnostics.error(
                    f"case node {node_id!r} has edge handle {handle!r} with no matching case"
                )
        for handle in case_handles:
            if handle not in out_handles:
                raise diagnostics.error(
                    f"case node {node_id!r} case {handle!r} has no outgoing edge"
                )
