"""Graph-stage: load-time `flow.wiring` / `params` key parity for every node.

A loader post-build pass. Every node's `flow.wiring` keys must equal its declared `params`
names PLUS the kind's reserved wiring keys (`until` for WAIT, `over` for a mapped
`call`). Its range anchor comes from the node's transient Compose
`BuildResult.field_ranges[("field", "inputs")]` (the coarse `input:` mapping span, else
the authored node range).

Imports flow down only: `compose.helpers.errors`, `compose.nodes`, `diagnostics`, and `typing`.
Only the loader (and the compose package facade) imports this back.
"""

from __future__ import annotations

from typing import Any, Optional

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.base import BuildResult
from yian.diagnostics import SourceRange


def check_wiring_parity(
    build_results: dict[str, BuildResult],
    flow_wiring: dict[str, dict[str, Any]],
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Load-time key parity: every node's `flow.wiring` keys must equal its declared
    `params` names PLUS the kind's reserved keys (`until` for WAIT, `over` for a mapped
    `call`).

    Closes the duplicate-param hole by construction (a nested-dict wiring cannot hold a duplicate
    param) and turns an orphan source (a wiring key with no param), a missing source (a param with no
    wiring), or a node absent from `flow.wiring` entirely (silent {} bind -> lost input) into a
    located `LoadError`, anchored on the node's Compose-owned `input:` mapping range,
    else its authored node range. Engine bind-time
    reserved keys (the mapped-call `over` / WAIT `until` the eval_node seam pre-resolves) are never
    author wiring and do not appear here."""
    diagnostics = diagnostics or LoadDiagnostics()
    for nid, result in build_results.items():
        node = result.node
        expected = {p.name for p in (node.params or [])} | node.reserved_wiring_keys()
        actual = set(flow_wiring.get(nid, {}))
        if actual != expected:
            rng: Optional[SourceRange] = (
                result.field_ranges.get(("field", "inputs"))
                or result.node_range
            )
            raise diagnostics.error(
                f"node {nid!r}: flow.wiring keys {sorted(actual)} != declared params + reserved "
                f"{sorted(expected)} (wiring/params key parity)",
                range=rng,
            )
