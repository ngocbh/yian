"""Graph-stage: load-time `flow.wiring` / `params` key parity for every node.

A loader post-build pass. Every node's `flow.wiring` keys must equal its declared `params`
names PLUS the kind's reserved wiring keys (`until` for WAIT, `over` for a mapped
`call`). It relocates `build.check_wiring_parity` verbatim, adding a `.range` anchor off the
built node's `field_ranges[("field", "inputs")]` (the coarse `input:` mapping span, else
the node's own range).

Imports flow down only: `compose.helpers.errors`, `events` (SourceRange), `typing`. Only the loader
(and the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.events import SourceRange


def check_wiring_parity(
    nodes: dict,
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
    located `LoadError`, anchored on the node's `input:` mapping range (`field_ranges[("field", "inputs")]`,
    else the node's own range). Engine bind-time
    reserved keys (the mapped-call `over` / WAIT `until` the eval_node seam pre-resolves) are never
    author wiring and do not appear here."""
    diagnostics = diagnostics or LoadDiagnostics()
    for nid, node in nodes.items():
        expected = {p.name for p in (node.params or [])} | node.reserved_wiring_keys()
        actual = set(flow_wiring.get(nid, {}))
        if actual != expected:
            rng: Optional[SourceRange] = node.field_ranges.get(("field", "inputs")) or node.range
            raise diagnostics.error(
                f"node {nid!r}: flow.wiring keys {sorted(actual)} != declared params + reserved "
                f"{sorted(expected)} (wiring/params key parity)",
                range=rng,
            )
