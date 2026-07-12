"""DAG boundary synthesis: `START_ID`/`END_ID` nodes + their edges over ALL edges.

The `build_leaf_node` funnel is DISSOLVED (#70, Step 6): every node kind now builds through
its own `descriptor.build(ctx) -> BuildResult` (see `compose/nodes/<kind>.py`), and the loader's
conductor loop drives them kind-blind. What remains here is the boundary synthesis — `synthesize_roots`
and `synthesize_boundary_graph` — which is representation-neutral graph plumbing, not per-kind build.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.model import END_ID, START_ID, Edge, FlowOutput
from agent_composer.expr import binding_co_skips
from agent_composer.nodes.base import Node
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.compose.edges.producers import _binding_producers


# --------------------------------------------------------------------------- #
# DAG assembly: root/`__start__` + terminal/`__end__` over ALL edges
#
# Runs AFTER both the data edges and the `case` control edges are
# reconciled — so a `case` branch target carrying only a `gate -> target` control
# edge is correctly demoted from root. A root is a node with NO incoming edge of ANY
# kind (data OR control); the engine seeds every root unconditionally, so a wrongly-
# marked root makes BOTH branches of a `case` run. The flow `outputs:` bindings give
# the terminals: each `${P.output[...]}` producer gets a `P -> __end__` edge (a
# coalesce flow-output -> one terminal edge per producer; only the taken branch's
# value survives, resolved by `terminal_output`).
# --------------------------------------------------------------------------- #


def synthesize_roots(node_ids: "set[str]", edges: list[Edge]) -> list[Edge]:
    """`START_ID -> root` edges for body nodes with no incoming edge of ANY kind.

    `edges` must be the FULL reconciled edge set (data + control + END_ID producer); a node
    demoted by a control edge (a `case` branch target) or an END_ID producer is therefore
    excluded from the roots. The START_ID/END_ID boundary nodes are never roots
    here (END_ID gets its own bare-root edge in `synthesize_boundary` when 0-output).
    """
    has_incoming = {e.to for e in edges}
    roots = sorted(nid for nid in node_ids if nid not in has_incoming)
    return [Edge(id=f"{START_ID}->{nid}", from_=START_ID, to=nid) for nid in roots]


def synthesize_boundary_graph(
    input_decls: list,
    outputs: list[FlowOutput],
    node_ids: "set[str]",
    body_edges: list[Edge],
    start_field_ranges: "Optional[dict]" = None,
) -> tuple[dict[str, Node], list[Edge], dict[str, dict[str, Any]]]:
    """Synthesize the `START_ID`/`END_ID` boundary nodes + their edges + wiring.

    Every flow becomes `START_ID -> body -> END_ID` with ORDINARY edges (the `__start__`/`__end__`
    sentinels are retired — the strings persist only as the reserved node ids). Returns:

    - `boundary_nodes`: `{START_ID: StartNode, END_ID: EndNode(record)}`.
    - `edges`: the END_ID producer edges (one per producer per declared output, `input_group`
      = the output name, `optional = not binding_co_skips` so a coalesce/`:-`/`:?` output
      never co-skips) + the START_ID root edges (a bare
      `START_ID->nid` for every body node with no incoming of any kind, over the FULL set
      incl. the END_ID producer edges) + a bare `START_ID->END_ID` root edge for a 0-output END_ID
      (so the 0-output END_ID runs and commits `None`).
    - `boundary_wiring`: parity-clean wiring for both — `START_ID` keyed by input name
      (provisional `${input.X}` source so `check_wiring_parity` passes; the engine seeds
      START_ID's record directly at runtime), `END_ID` keyed by output name -> the output's `from_`.

    The START_ID-side `${input.X}`-producer DATA edges + the demote-input-reader-from-root
    rule live here too (every input read resolves through the START_ID namespace).

    `start_field_ranges` (optional) is the flow-boundary source map stamped onto the StartNode
    (`input name -> SourceRange`, flow-`asserts:` `expr -> SourceRange`) — render-only, so a
    boundary input-type failure / false boundary assert resolves a precise caret. `None` for a
    spanless flow (compact / in-memory / def).
    """
    start = StartNode(START_ID, input_decls=list(input_decls),
                      field_ranges=start_field_ranges)
    end = EndNode.record(END_ID, output_names=[o.name for o in outputs])
    boundary_nodes: dict[str, Node] = {START_ID: start, END_ID: end}

    # END_ID producer edges — one per producer per output, keyed by the output NAME, with the
    # coalesce/optional stance (reuses the infer_data_edges emit_for convention).
    end_edges: list[Edge] = []
    counts: dict[tuple[str, str], int] = {}
    for o in outputs:
        optional = not binding_co_skips(o.from_)
        for producer in _binding_producers(o.from_):
            i = counts.get((producer, END_ID), 0)
            counts[(producer, END_ID)] = i + 1
            end_edges.append(Edge(id=f"{producer}->{END_ID}#{i}", from_=producer, to=END_ID,
                                  input_group=o.name, optional=optional))

    # START_ID root edges over the FULL set (body + END_ID producer edges): a body node with no
    # incoming of any kind is a bare root. A 0-output END_ID (no producer edge) is itself a root.
    full = body_edges + end_edges
    has_incoming = {e.to for e in full}
    root_edges = [Edge(id=f"{START_ID}->{nid}", from_=START_ID, to=nid)
                  for nid in sorted(node_ids) if nid not in has_incoming]
    if END_ID not in has_incoming:
        root_edges.append(Edge(id=f"{START_ID}->{END_ID}", from_=START_ID, to=END_ID))

    boundary_wiring: dict[str, dict[str, Any]] = {
        # START_ID wiring: input names -> `${input.X}` sources. Provisional in the wiring
        # table (parity check) — at runtime the engine seeds store[START_ID] directly
        # from the run-args record (START_ID is never scheduled as a regular node).
        START_ID: {d.name: f"${{input.{d.name}}}" for d in input_decls},
        END_ID: {o.name: o.from_ for o in outputs},
    }
    return boundary_nodes, end_edges + root_edges, boundary_wiring
