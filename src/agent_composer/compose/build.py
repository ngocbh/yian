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
from agent_composer.expr import (
    ExpressionError,
    binding_co_skips,
    condition_refs,
    expr_refs_of,
)
from agent_composer.nodes.base import Node
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.typesys.types import TypeRegistry
from agent_composer.compose.errors import LoadError
from agent_composer.compose.parser import (
    CallDescriptor,
    CaseDescriptor,
    NodeDescriptor,
    LoopDescriptor,
    WaitDescriptor,
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
# Data-edge inference
#
# The analogue of scope/dependency analysis: a sink binding that reads a free
# variable `${<id>.output[.…]}` depends on the producing node `<id>`, so we emit a
# data `Edge(from_=<id>, to=<consumer>, input_group=<sink key>)`. `expr_refs_of`
# already flattens a coalesce `${a | b}` to BOTH ref paths, so the two alternatives
# of one sink share that sink's `input_group` (per-input readiness). Refs whose
# head is not `outputs` (`inputs`/`system`/`item`) are NOT data edges.
#
# Root/terminal `__start__`/`__end__` synthesis is a later pass (it must run after the
# `case` control edges): this pass emits ONLY real-node data edges.
# --------------------------------------------------------------------------- #


def _outputs_producer(path: str) -> Optional[str]:
    """The producing node id of a `${<id>.output[.…]}` whole-string ref, else None
    (other head). Only the singular node-first spelling is recognized; legacy
    `${outputs.<id>}` is rejected at parse time."""
    parts = path.split(".")
    if len(parts) >= 2 and parts[1] == "output":
        return parts[0]
    return None


def _ref_producer(path: str) -> Optional[str]:
    """The producer node id of a `${...}` reference head (singular):
    - `<id>.output[.…]`: produce from node `<id>`.
    - `input.<key>`: produce from the synthesized START_ID, so an `${input.X}`
      reader gets a `START_ID->reader` data edge.
    - `system.<key>` is run-global and produces NO edge; `item` is MAP-body-local.
    Legacy plural heads (`outputs`/`inputs`) are rejected at parse time."""
    parts = path.split(".")
    if len(parts) >= 2 and parts[0] == "input":
        return START_ID
    if len(parts) >= 2 and parts[1] == "output":
        return parts[0]
    return None


def _binding_producers(value: Any) -> list[str]:
    """The producer node ids a binding/expression value reads (in source order).

    `value` may be a `${...}` binding string, a `when:`/`on:` expression template,
    or a non-string literal (no refs). Coalesce alternatives flatten to several
    producers — the caller assigns them ALL the one sink `input_group`. An `inputs.X` ref
    produces from START_ID (the input-reader data edge); `system.X` produces nothing.
    """
    if not isinstance(value, str):
        return []
    try:
        refs = expr_refs_of(value)
    except ExpressionError:
        return []  # malformed refs surface in the ref-wiring pass, located
    producers: list[str] = []
    for ref in refs:
        producer = _ref_producer(ref)
        if producer is not None:
            producers.append(producer)
    return producers


def _condition_producers(value: Any) -> list[str]:
    """The producer node ids a CONDITION expression reads (in source order).

    A `case` `when:`/`on:` is a whole boolean/value expression, not a template — it is
    written in any of the three condition spellings (bare `a.output > 5`, mixed
    `${a.output} > 5`, whole-span `${a.output > 5}`), so refs are extracted via
    `condition_refs` (which parses the whole expression) rather than `expr_refs_of`
    (which only sees `${...}` spans and would miss the bare spelling). `system.X`
    produces nothing; an `input.X` ref produces from START_ID.
    """
    if not isinstance(value, str):
        return []
    try:
        refs = condition_refs(value)
    except ExpressionError:
        return []  # malformed refs surface in the ref-wiring pass, located
    producers: list[str] = []
    for ref in refs:
        producer = _ref_producer(ref)
        if producer is not None:
            producers.append(producer)
    return producers



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
