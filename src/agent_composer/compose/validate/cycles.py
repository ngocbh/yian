"""Graph-stage: reject a directed cycle in the inferred (real-node) graph.

Cross-node assembly pass. After all data + ordering edges are inferred, a flow graph
must be a DAG — the scheduler resolves nodes by settling their producers, so a loop
(`a -> b -> a`) can never make progress. This pass runs Kahn's algorithm over the
non-sentinel edges and raises a located `LoadError` naming the stuck nodes when a cycle
remains. The boundary sentinels (`__start__`/`__end__`) are acyclic by construction, so
passing the fully-assembled edge set is fine.

Located parse metadata: the caller passes `node_ranges` (node id -> `SourceRange`); the
cycle error anchors its `range` on the first (sorted) stuck node that has one. Transitionally
it also passes `node_lines` (node id -> 1-based key line) so the CLI can highlight *every*
stuck node (`flow.yaml:4,9`) via `LoadError.lines`; the compatibility `.line` still derives
from the authoritative range. The error also carries a "why"
legend (`LoadError.notes`): one line per dependency edge inside the loop, so the author sees
which `${...}` reference closes the cycle.

Imports flow down/peer only: `compile.model` (Edge + boundary ids), `compile.validation`
(the shared Kahn check), `compose.helpers.errors`, `events` (SourceRange). Only the loader (and
the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Optional

from agent_composer.compile.model import END_ID, START_ID, Edge
from agent_composer.compile.validation import FlowValidationError, _reject_cycles
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange


def reject_cycles(
    edges: list[Edge],
    node_ids: "set[str]",
    node_lines: "dict[str, int] | None" = None,
    node_ranges: "dict[str, SourceRange] | None" = None,
) -> None:
    """Raise a `LoadError` if the inferred (real-node) graph has a directed cycle.

    Delegates to the shared `_reject_cycles` (Kahn's algorithm over the non-sentinel
    edges); a cyclic flow (`a -> b -> a`) is loud and names the stuck nodes. Sentinel
    `__start__`/`__end__` edges can't close a cycle, so passing the fully-assembled edge
    set is fine.

    `node_ranges` (node id -> `SourceRange`) lets the error anchor on a node in the loop:
    the error's `.range` is the first (sorted) stuck node's range. Absent it, `.range` is
    None. `node_lines` (node id -> 1-based key line) is the transitional line map used to
    put every stuck node in `LoadError.lines`, so the frame highlights the whole loop. When
    a primary range exists, the compatibility `.line` follows that range rather than the
    smallest secondary line.

    The error also carries a "why" legend (`LoadError.notes`): one line per dependency edge
    *inside* the loop — "<consumer> depends on <producer> (<consumer>.input.<group>)" — so the
    author sees which reference closes the cycle, not just which nodes are in it.
    """
    lines_map = node_lines or {}
    ranges = node_ranges or {}
    try:
        _reject_cycles(edges, node_ids)
    except FlowValidationError as exc:
        stuck = _stuck_nodes(edges, node_ids)
        lines = [lines_map[nid] for nid in stuck if nid in lines_map]
        rng: Optional[SourceRange] = next(
            (ranges[nid] for nid in stuck if nid in ranges), None
        )
        notes = _cycle_notes(edges, set(stuck))
        raise LoadError(
            str(exc),
            line=rng.start_line if rng is not None else None,
            lines=(lines or None),
            range=rng,
            notes=notes or None,
        ) from exc


def _cycle_notes(edges: list[Edge], stuck: "set[str]") -> list[str]:
    """The "why" legend for a cycle: one line per dependency edge between stuck nodes.

    Each loop-internal edge `from_ -> to` reads as "`to` consumes `from_`", i.e. the consumer
    depends on the producer; we name the consumer's input group so the author finds the exact
    `${...}` reference that closes the loop. A data edge cites `to.input.<group>`; a pure
    ordering edge (`depends_on`/`runs_after`, no data) is labelled as such. Sorted + de-duped
    for a deterministic legend.
    """
    seen: set[tuple[str, str, str]] = set()
    notes: list[str] = []
    for e in edges:
        if e.from_ in stuck and e.to in stuck and e.from_ != e.to:
            via = f"{e.to}.input.{e.input_group}" if e.input_group else "ordering"
            key = (e.to, e.from_, via)
            if key not in seen:
                seen.add(key)
                notes.append(f"{e.to} depends on {e.from_} ({via})")
    return sorted(notes)


def _stuck_nodes(edges: list[Edge], node_ids: "set[str]") -> list[str]:
    """The nodes left with in-degree > 0 after a Kahn pass (the cycle members), sorted.

    Mirrors the stuck-set `_reject_cycles` reports, so the cycle error can anchor on the
    first stuck node's range. Sentinel `__start__`/`__end__` edges are ignored.
    """
    in_degree = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for edge in edges:
        if edge.from_ in (START_ID, END_ID) or edge.to in (START_ID, END_ID):
            continue
        adjacency[edge.from_].append(edge.to)
        in_degree[edge.to] += 1
    queue = [nid for nid in node_ids if in_degree[nid] == 0]
    while queue:
        nid = queue.pop()
        for nxt in adjacency[nid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    return sorted(nid for nid in node_ids if in_degree[nid] > 0)
