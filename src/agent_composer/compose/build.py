"""Build leaf runtime `Node`s from node descriptors.

The analogue of `compiler._build_node` + `_declared_inputs` + `_io_binding`,
working from the parsed `NodeDescriptor`s (flat keyed-map bodies) instead of the
legacy `FlowNode`/`IOField`. For each LEAF kind (agent/code/model/tool) it:

- instantiates the matching runtime `Node` per-kind (mirroring `_build_node`'s
  ctor args), with `node_name` -> the Node `title`;
- stamps `output_type = read_type(descriptor.outputs, registry)` (the source
  reader), or `None` when the descriptor declares no `outputs:`;
- stamps `params` (the node-side signature, names only) from the descriptor's `inputs:`
  map (a TOOL's `args:`) and returns the flow-owned `wiring` (`param -> source`) — the
  split: the node holds no source, the flow owns it (the type travels with it).

CASE desugar, REF/MAP build happen elsewhere — only leaf kinds here.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.model import END_ID, START_ID, Edge, FlowOutput
from agent_composer.expr import binding_co_skips
from agent_composer.nodes.base import Node
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.typesys.types import TypeRegistry
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.edges.producers import _binding_producers
from agent_composer.compose.parse.file import (
    CallDescriptor,
    NodeDescriptor,
    LoopDescriptor,
)


def build_leaf_node(
    desc: NodeDescriptor,
    registry: TypeRegistry,
    *,
    source_file: str = "<inline-code>",
    code_line: Optional[int] = None,
    code_field_line: Optional[int] = None,
) -> tuple[Node, dict[str, Any]]:
    """Build one leaf runtime `Node` (agent/code/model/tool/human_input/wait) + its WIRING.

    Returns `(node, wiring)` where `wiring` is the flow-owned `{param -> source}` map (the
    split: the node carries only `params`, the flow owns the sources in
    `CompiledFlow.wiring`). A timed WAIT's reserved `until` source rides the wiring under the
    `"until"` key; an event WAIT's wiring is `{}`. `registry` is the flow's
    `read_typedefs(...)` map. A MODEL node builds with no serving seam (removed as dead).

    `source_file` (the flow's file label) frames an inline CODE node's tracebacks at the
    absolute YAML line. `code_line` (the `code:` block's content start line) and
    `code_field_line` (the `code:` field-key line, anchoring an inline reject / no-`return`
    error) are optional caller overrides; when omitted (the range-native loader path) the
    CODE build derives both from the descriptor's own recorded `code:` `SourceRange`.
    """
    # Two-axis dispatch (#70): every leaf kind (agent/code/model/tool/human_input/wait) builds
    # through its own descriptor's `build` method. Leaf builds ignore `resolver` (the shared
    # signature carries it only for call/map/loop). `source_file`/`code_line`/`code_field_line`
    # are inline-CODE traceback anchors — only the CODE build reads them; every other build
    # absorbs them via `**_loader_hints`. call/map/loop are not leaves — they need a resolver and
    # route through their own `build` from the loader, so they are rejected up front here. case is
    # desugared; ref nodes expand at runtime.
    if isinstance(desc, (CallDescriptor, LoopDescriptor)):
        raise LoadError(
            f"node {desc.id!r}: kind {type(desc).__name__} is not a leaf node "
            f"(call/map/loop build through their own descriptor with a resolver)"
        )
    return desc.build(
        registry=registry, resolver=None,
        source_file=source_file, code_line=code_line, code_field_line=code_field_line,
    )


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
