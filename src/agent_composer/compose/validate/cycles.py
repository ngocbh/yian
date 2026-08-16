"""Graph-stage: reject a directed cycle in the inferred (real-node) graph.

Cross-node assembly pass. After all data + ordering edges are inferred, a flow graph
must be a DAG — the scheduler resolves nodes by settling their producers, so a loop
(`a -> b -> a`) can never make progress. This pass runs Kahn's algorithm over the
non-sentinel edges and raises a located `LoadError` naming the stuck nodes when a cycle
remains. The boundary sentinels (`__start__`/`__end__`) are acyclic by construction, so
passing the fully-assembled edge set is fine.

Located parse metadata: the caller passes `node_ranges` (node id -> `SourceRange`). The
cycle diagnostic carries one frame per located stuck node in authored source order. Its
notes form a "why" legend: one line per dependency edge inside the loop, so the author sees
which `${...}` reference closes the cycle.

Imports flow down/peer only: `compile.model` (Edge + boundary ids), `compile.validation`
(the shared Kahn check), `compose.helpers.errors`, `events` (SourceRange). Only the loader (and
the compose package facade) import this back.
"""

from __future__ import annotations

from agent_composer.compile.model import END_ID, START_ID, Edge
from agent_composer.compile.validation import FlowValidationError, _reject_cycles
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.events import SourceRange


def reject_cycles(
    edges: list[Edge],
    node_ids: "set[str]",
    node_lines: "dict[str, int] | None" = None,
    node_ranges: "dict[str, SourceRange] | None" = None,
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Raise a `LoadError` if the inferred (real-node) graph has a directed cycle.

    Delegates to the shared `_reject_cycles` (Kahn's algorithm over the non-sentinel
    edges); a cyclic flow (`a -> b -> a`) is loud and names the stuck nodes. Sentinel
    `__start__`/`__end__` edges can't close a cycle, so passing the fully-assembled edge
    set is fine.

    `node_ranges` (node id -> `SourceRange`) gives every located node in the loop its own
    frame, ordered by authored line and column. `node_lines` remains only for transitional
    caller compatibility and is not location authority.

    The error also carries a "why" legend (`Diagnostic.notes`): one line per dependency edge
    *inside* the loop — "<consumer> depends on <producer> (<consumer>.input.<group>)" — so the
    author sees which reference closes the cycle, not just which nodes are in it.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    ranges = node_ranges or {}
    try:
        _reject_cycles(edges, node_ids)
    except FlowValidationError as exc:
        stuck = _stuck_nodes(edges, node_ids)
        authored = sorted(
            (
                source_range.start_line,
                source_range.start_col,
                node_id,
                source_range,
            )
            for node_id in stuck
            if (source_range := ranges.get(node_id)) is not None
        )
        frames = tuple(
            frame
            for _, _, _, source_range in authored
            if (frame := diagnostics.frame(source_range)) is not None
        )
        notes = _cycle_notes(edges, set(stuck))
        raise diagnostics.error(
            str(exc),
            frames=frames,
            notes=notes,
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

    Mirrors the stuck-set `_reject_cycles` reports so each located member can receive a
    frame. Sentinel `__start__`/`__end__` edges are ignored.
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
