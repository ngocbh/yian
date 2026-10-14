"""Graph-stage: reject a directed cycle in the inferred (real-node) graph.

Cross-node assembly pass. After all data + ordering edges are inferred, a flow graph
must be a DAG — the scheduler resolves nodes by settling their producers, so a loop
(`a -> b -> a`) can never make progress. This pass uses Kahn's algorithm to detect a
cycle, then isolates the cyclic strongly connected components so downstream blocked
nodes are not blamed. The boundary sentinels (`__start__`/`__end__`) are acyclic by
construction, so passing the fully-assembled edge set is fine.

Located parse metadata: the caller passes `node_ranges` (node id -> `SourceRange`). The
cycle diagnostic carries one frame per located cycle member in authored source order.
Its notes form a "why" legend: one line per dependency edge inside a cyclic component,
so the author sees which `${...}` reference closes the cycle.

Imports flow down/peer only: `compile.model` (Edge + boundary ids), `compile.validation`
(the shared Kahn check), `compose.helpers.errors`, and `diagnostics` (SourceRange). Only the
loader (and the compose package facade) imports this back.
"""

from __future__ import annotations

from yian.compile.model import END_ID, START_ID, Edge
from yian.compile.validation import FlowValidationError, _reject_cycles
from yian.compose.helpers.errors import LoadDiagnostics
from yian.diagnostics import SourceRange


def reject_cycles(
    edges: list[Edge],
    node_ids: "set[str]",
    *,
    node_ranges: "dict[str, SourceRange] | None" = None,
    diagnostics: LoadDiagnostics | None = None,
    gate_ids: frozenset[str] = frozenset(),
) -> None:
    """Raise a `LoadError` if the inferred (real-node) graph has a directed cycle.

    Delegates detection to the shared `_reject_cycles` (Kahn's algorithm over the
    non-sentinel edges), then identifies the actual cyclic strongly connected
    components. A node blocked downstream of a cycle is not a cycle member. Sentinel
    `__start__`/`__end__` edges cannot close a cycle.

    `node_ranges` (node id -> `SourceRange`) gives every located node in the loop its own
    frame, ordered by authored line and column.

    The error also carries a "why" legend (`Diagnostic.notes`): one line per dependency edge
    *inside* the loop — "<consumer> depends on <producer> (<consumer>.input.<group>)" — so the
    author sees which reference closes the cycle, not just which nodes are in it.

    `gate_ids` names the data gates a lifted node got from its host (group
    `<node>.<group>`); their notes cite the host's input, "(gate: <node>.input.<group>)".
    """
    diagnostics = diagnostics or LoadDiagnostics()
    ranges = node_ranges or {}
    try:
        _reject_cycles(edges, node_ids)
    except FlowValidationError as exc:
        components = _cyclic_components(edges, node_ids)
        cycle_nodes = set().union(*components)
        authored = sorted(
            (
                source_range.start_line,
                source_range.start_col,
                node_id,
                source_range,
            )
            for node_id in cycle_nodes
            if (source_range := ranges.get(node_id)) is not None
        )
        frames = tuple(
            frame
            for _, _, _, source_range in authored
            if (frame := diagnostics.frame(source_range)) is not None
        )
        notes = _cycle_notes(edges, components, gate_ids=gate_ids)
        raise diagnostics.error(
            f"flow has a cycle involving {sorted(cycle_nodes)}; flows must be acyclic",
            frames=frames,
            notes=notes,
        ) from exc


def _cycle_notes(
    edges: list[Edge],
    components: tuple[frozenset[str], ...],
    *,
    gate_ids: frozenset[str] = frozenset(),
) -> list[str]:
    """Return one note per dependency edge within a cyclic component.

    Each loop-internal edge `from_ -> to` reads as "`to` consumes `from_`", i.e. the
    consumer depends on the producer. An edge connecting two separate cycles is not
    itself cyclic and is omitted. A data edge cites `to.input.<group>`; a pure ordering
    edge (`depends_on`/`runs_after`, no data) is labelled as such. An edge whose id is in
    `gate_ids` is a lifted node's data gate, whose group is `<node>.<group>`: it cites
    `gate: <node>.input.<group>`, the host input it mirrors. Notes are sorted and
    de-duplicated for deterministic presentation.
    """
    component_by_node = {
        node_id: index
        for index, component in enumerate(components)
        for node_id in component
    }
    seen: set[tuple[str, str, str]] = set()
    notes: list[str] = []
    for e in edges:
        component = component_by_node.get(e.from_)
        if component is not None and component_by_node.get(e.to) == component:
            if e.id in gate_ids and e.input_group:
                # Only gate ids are split: an authored input key may itself contain a `.`.
                node, _, group = e.input_group.partition(".")
                via = f"gate: {node}.input.{group}"
            elif e.input_group:
                via = f"{e.to}.input.{e.input_group}"
            else:
                via = "ordering"
            key = (e.to, e.from_, via)
            if key not in seen:
                seen.add(key)
                notes.append(f"{e.to} depends on {e.from_} ({via})")
    return sorted(notes)


def _cyclic_components(
    edges: list[Edge],
    node_ids: set[str],
) -> tuple[frozenset[str], ...]:
    """Return the graph's cyclic strongly connected components deterministically.

    Two iterative depth-first passes implement Kosaraju's algorithm without a Python
    recursion-depth limit. Components with multiple nodes are cyclic; a singleton is
    cyclic only when it has a self-edge. Sentinel edges are ignored.
    """
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    reverse: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        if edge.from_ in (START_ID, END_ID) or edge.to in (START_ID, END_ID):
            continue
        adjacency[edge.from_].add(edge.to)
        reverse[edge.to].add(edge.from_)

    visited: set[str] = set()
    finish_order: list[str] = []
    for start in sorted(node_ids):
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            node_id, expanded = stack.pop()
            if expanded:
                finish_order.append(node_id)
                continue
            if node_id in visited:
                continue
            visited.add(node_id)
            stack.append((node_id, True))
            stack.extend(
                (target, False)
                for target in sorted(adjacency[node_id], reverse=True)
                if target not in visited
            )

    assigned: set[str] = set()
    components: list[frozenset[str]] = []
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component: set[str] = set()
        stack = [start]
        assigned.add(start)
        while stack:
            node_id = stack.pop()
            component.add(node_id)
            for source in sorted(reverse[node_id], reverse=True):
                if source not in assigned:
                    assigned.add(source)
                    stack.append(source)
        if len(component) > 1 or start in adjacency[start]:
            components.append(frozenset(component))

    return tuple(sorted(components, key=lambda item: tuple(sorted(item))))
