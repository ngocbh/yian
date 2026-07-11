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

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from agent_composer.compile.model import END_ID, START_ID, Edge, FlowOutput
from agent_composer.compile.validation import types_compatible
from agent_composer.expr import (
    ExpressionError,
    binding_co_skips,
    condition_refs,
    expr_refs_of,
)
from agent_composer.expr.template import Span, scan_template, template_refs
from agent_composer.nodes.base import Node
from agent_composer.nodes.binding import ParamDecl
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.map import MapNode
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.llm_clients import LLMConfig
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.typesys.types import TypeRegistry
from agent_composer.compose.errors import LoadError
from agent_composer.compose.parser import (
    CallDescriptor,
    CaseDescriptor,
    NodeDescriptor,
    LoopDescriptor,
    WaitDescriptor,
)
from agent_composer.compose.types import InputDecl, read_type

# A list-type helper: the LIST_OBJECT/LIST_* seg for an element Type (mirrors
# `typesys.types._LIST_SEG_FOR_ELEMENT` + the list-of-record / list-of-variant rule in
# `resolve_type`, applied to a precomputed element Type rather than a Type).
_LIST_SEG_FOR_ELEMENT: dict[ValueKind, ValueKind] = {
    ValueKind.STRING: ValueKind.LIST_STRING,
    ValueKind.INTEGER: ValueKind.LIST_INTEGER,
    ValueKind.NUMBER: ValueKind.LIST_NUMBER,
    ValueKind.BOOLEAN: ValueKind.LIST_BOOLEAN,
    ValueKind.OBJECT: ValueKind.LIST_OBJECT,
}


def _output_type(outputs: Any, registry: TypeRegistry) -> Optional[Type]:
    """A descriptor's declared `outputs:` as one `Type`; `None` when absent."""
    if outputs is None:
        return None
    return read_type(outputs, registry)


def _sink_params(mapping: dict[str, Any]) -> list[ParamDecl]:
    """A descriptor's `inputs:`/`args:` map -> untyped `ParamDecl`s (the node-side signature).

    The split: `params` carries only the declared NAMES (no source); the flow owns the
    sources in `CompiledFlow.wiring`. Untyped (`type_str`/`type` None) — the source carries the
    type; derived from the SAME map as `_sink_wiring` so names and keys are in lockstep."""
    return [ParamDecl(name=k) for k in (mapping or {})]


def _sink_wiring(mapping: dict[str, Any]) -> dict[str, Any]:
    """A descriptor's `inputs:`/`args:` map -> the flow-owned wiring (`param -> source`).
    The source half of the split; the same dict `_sink_params` reads the names from."""
    return dict(mapping or {})


def _validate_llm_config(
    cfg: dict[str, Any] | None, node_id: str
) -> tuple[dict[str, Any], bool]:
    """Validate a raw llm_config dict at LOAD time and split off the reserved `inherit` key.

    `inherit` (whole-node cascade opt-out, default True) is NOT an `LLMConfig` field, so it
    must be popped before the `extra="forbid"` round-trip that fires on typo'd keys (e.g.
    `temparature`). Returns `(config_without_inherit, inherit)`. Raises `LoadError` (not
    pydantic ValidationError) so the surface speaks one error type.
    """
    if not cfg:
        return {}, True
    cfg = dict(cfg)
    inherit = bool(cfg.pop("inherit", True))
    try:
        LLMConfig(**cfg)
    except Exception as exc:
        raise LoadError(f"node {node_id!r}: llm_config: {exc}") from exc
    return cfg, inherit


def build_leaf_node(
    desc: NodeDescriptor,
    registry: TypeRegistry,
    *,
    source_file: str = "<inline-code>",
    code_line: int = 1,
    code_field_line: Optional[int] = None,
) -> tuple[Node, dict[str, Any]]:
    """Build one leaf runtime `Node` (agent/code/model/tool/human_input/wait) + its WIRING.

    Returns `(node, wiring)` where `wiring` is the flow-owned `{param -> source}` map (the
    split: the node carries only `params`, the flow owns the sources in
    `CompiledFlow.wiring`). A timed WAIT's reserved `until` source rides the wiring under the
    `"until"` key; an event WAIT's wiring is `{}`. `registry` is the flow's
    `read_typedefs(...)` map. A MODEL node builds with no serving seam (removed as dead).

    `source_file` (the flow's file label) and `code_line` (the `code:` block's content
    start line) are used only by an inline CODE node, to frame its tracebacks at the
    absolute YAML line. `code_field_line` (the `code:` field-key line) anchors an inline
    load error that has no interior line — a reject, or a no-`return` body — at the `code:`
    field rather than the node header.
    """
    # Two-axis dispatch (#70): every leaf kind (agent/code/model/tool/human_input/wait) builds
    # through its own `KindSpec.build`. Leaf descriptors are 1:1 with their kind, so we match a
    # spec by descriptor class; leaf builds ignore `resolver` (the shared signature carries it
    # only for call/map/loop). `source_file`/`code_line`/`code_field_line` are inline-CODE
    # traceback anchors — only the CODE build reads them; every other build absorbs them via
    # `**_loader_hints`. Imported lazily: kind modules import build helpers from here, so a
    # top-level import would cycle. Anything past this loop is not a leaf (case is desugared;
    # ref/map nodes expand at runtime) — a loud LoadError.
    from agent_composer.compose.kinds import KINDS

    for spec in KINDS.values():
        if spec.descriptor_cls is type(desc):
            return spec.build(
                desc, registry=registry, resolver=None,
                source_file=source_file, code_line=code_line, code_field_line=code_field_line,
            )
    raise LoadError(
        f"node {desc.id!r}: kind {type(desc).__name__} is not a leaf node "
        f"(case nodes are desugared; ref/map nodes expand at runtime)"
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
    """
    start = StartNode(START_ID, input_decls=list(input_decls))
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


# --------------------------------------------------------------------------- #
# `call` build + child resolve-and-bake.
#
# A `call` node names a CALLABLE (a defs: entry or an external flow). The child resolver
# is the load-time seam: `(flow_id, version) -> LoadedFlow` (the loader composes the
# defs-first resolution). From the child the loader derives its `ChildSignature` —
# declared input types + single codomain `Type` — to stamp the node's `output_type` (a
# plain call re-exports the codomain; a mapped call wraps it in `list[U]`) and to
# name/arity- + type-check the bindings, AND bakes the child's compiled flow + input
# decls onto the node so the built `CallNode` `run` drives the embedded child
# (resolve-and-bake — the call graph, incl. nested flows, is fixed at compile).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChildSignature:
    """A child flow's compile-time SIGNATURE: declared input decls + the single
    codomain `Type` (the value the child returns; `None` when it declares no outputs).

    `inputs` are the child's `read_flow_inputs(...)` `InputDecl`s (the name/arity check
    reads their names + `.required`/`.default`). `output` is the child's single value:
    one declared output -> that output's Type; >=2 -> a closed-record Type keyed by
    name (the same arity rule `terminal_output` assembles: one value out)."""

    inputs: list[InputDecl]
    output: Optional[Type]


# resolver(flow_id, flow_version) -> LoadedFlow (the load-time child seam; duck-typed
# on `.compiled`/`.inputs` so `build` need not import `loader`). `flow_version` is an
# opaque str tag (a `uses:` ref's `@<version>`); the in-file/legacy path passes None.
ChildResolver = Callable[[str, Optional[str]], Any]


def _output_value_type(
    from_: Any, producers: dict[str, Type], flow_input_types: dict[str, Type]
) -> Optional[Type]:
    """Resolve ONE flow-output binding `from_` to the Type of the value it carries.

    The cases:
      - a **single lone `${…}` span** — `${<id>.output[.…]}` for a node value or
        `${input.<name>}` for a flow input — is a typed passthrough (eval_template
        returns the referenced value unchanged; a multi-ref span stays opaque -> None).
      - a **template that concatenates** — one or more spans mixed with literal text, or
        two or more spans — is joined by eval_template, so it always carries a `str`
        (see `template.py`).
      - a **pure literal** (no spans) stays None — lenient, so a literal call-arg can
        still be coerced to the declared type at the boundary.
    An opaque producer / `${system.X}` in the single-span case -> None (codomain stays
    opaque). Legacy plural heads were retired."""
    if not isinstance(from_, str):
        return None
    try:
        segments = scan_template(from_)
    except ExpressionError:
        return None
    spans = [s for s in segments if isinstance(s, Span)]
    if not spans:
        return None  # pure literal -> lenient (coerced at the boundary)
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return Type.scalar(ValueKind.STRING)  # concatenating template -> str
    refs = template_refs(segments)
    if len(refs) != 1:
        return None  # single span, multi-ref expr -> not a single typed value
    parts = refs[0].split(".")
    head = parts[0]
    if head == "input" and len(parts) == 2:
        return flow_input_types.get(parts[1])
    # Node-first `<id>.output[.…]`.
    if len(parts) >= 2 and parts[1] == "output":
        producer_id = parts[0]
        fields = parts[2:]
    else:
        return None  # ${system.X}/literal/other -> not a typed codomain value
    typ = producers.get(producer_id)
    for field in fields:
        if typ is None or typ.fields is None:
            return None  # opaque producer / not a checked record -> lenient
        typ = typ.fields.get(field)
    return typ


def child_signature(loaded: Any) -> ChildSignature:
    """A loaded child `LoadedFlow` -> its compile-time `ChildSignature`.

    Duck-typed on `loaded.input` (the `InputDecl`s) + `loaded.compiled` (`.nodes`/`.outputs`)
    so `build` need not import `loader` (which imports `build`). The codomain `Type`
    follows `terminal_output`'s arity rule over the flow outputs: one output -> its value
    Type; >=2 -> a closed record keyed by name (every output is a field; an output whose
    value Type can't be resolved is present but opaque)."""
    producers: dict[str, Type] = {
        nid: node.output_type
        for nid, node in loaded.compiled.nodes.items()
        if node.output_type is not None
    }
    flow_input_types = {d.name: d.type for d in loaded.input}
    outs = list(loaded.compiled.outputs)
    if not outs:
        output: Optional[Type] = None
    elif len(outs) == 1:
        output = _output_value_type(outs[0].from_, producers, flow_input_types)
    else:
        fields = {
            o.name: _output_value_type(o.from_, producers, flow_input_types)
            for o in outs
        }
        # Every declared output is a record field (a closed record keyed by name); an
        # output whose value Type didn't resolve is kept but opaque (its `fields=None`
        # leaves a deeper dotted walk lenient) and is NOT in `required`.
        output = Type(
            kind=ValueKind.OBJECT,
            fields={
                k: (v if v is not None else Type.scalar(ValueKind.NONE))
                for k, v in fields.items()
            },
            required=frozenset(k for k, v in fields.items() if v is not None),
        )
    return ChildSignature(inputs=list(loaded.input), output=output)


def _list_of(element: Type) -> Type:
    """`element` -> a `list[element]` Type (the MAP codomain). Mirrors `resolve_type`'s
    list-seg rule: a record/variant element -> LIST_OBJECT/LIST_STRING; else by scalar."""
    if element.tags is not None:
        list_seg = ValueKind.LIST_STRING
    elif element.fields is not None:
        list_seg = ValueKind.LIST_OBJECT
    else:
        list_seg = _LIST_SEG_FOR_ELEMENT.get(element.kind, ValueKind.LIST_ANY)
    return Type(kind=list_seg, element=element)


def _check_child_bindings(
    node_id: str,
    flow_id: str,
    params: list[ParamDecl],
    sig: ChildSignature,
    errors: list[str],
) -> None:
    """Name/arity check a `call` node's PARAMS against the child SIGNATURE (the analogue of
    `compile.validation._check_ref_bindings`'s name + required-input checks, read off the
    signature). Reads the node-side `params` (the split); the deep cross-flow TYPE
    check is `check_ref_map_types`, a loader post-build pass over the flow wiring."""
    declared = {d.name for d in sig.inputs}
    bound = set()
    for p in params:
        bound.add(p.name)
        if p.name not in declared:
            errors.append(
                f"call node {node_id!r}: binding {p.name!r} is not a declared "
                f"input of callable {flow_id!r}"
            )
    for decl in sig.inputs:
        if decl.required and decl.name not in bound and decl.default is None:
            errors.append(
                f"call node {node_id!r}: required callable input {decl.name!r} is unbound"
            )


def build_call_node(desc: CallDescriptor, resolver: ChildResolver) -> tuple[Node, dict[str, Any]]:
    """Build a `call`/`map` node — REF's `CallNode` (single application) or MAP's `MapNode`
    (`kind: map` + `over:`, mapped iteration) — resolve-and-bake.

    Returns `(node, wiring)` (the split): the node carries `params` + the transitional
    `inputs`; the flow owns the sources in `wiring`. A MAP's reserved `over:` iteration source
    rides the wiring under the `"over"` key (over-then-inputs order); a REF's wiring is just
    its `inputs:`. Resolves the callable via `resolver`, derives its `ChildSignature`, stamps
    `output_type` (REF re-exports the codomain; MAP wraps it in `list[U]`), name/arity-checks the
    params, and BAKES the child's compiled flow onto the node. The REF/MAP discriminator is
    `desc.kind` (`"map"` -> `MapNode`, else `CallNode`)."""
    # The construction boundary: this reads the RAW authored `desc.kind` STRING (not the
    # compiled kind enum) to pick the runtime class. Kind identity is BORN here — downstream
    # compile passes then dispatch on the node's traits, never on `node.kind`.
    is_map = desc.kind == "map"
    if is_map and "over" in (desc.inputs or {}):
        # Reserved-name guard: a MAP's `over` is the iteration source, carried under
        # the reserved wiring key — an `inputs:` param named `over` would collide.
        raise LoadError(
            f"map node {desc.id!r}: input name 'over' is reserved (the iteration "
            f"source); rename the `inputs:` entry"
        )
    child = _resolve_child(desc.id, desc.call, resolver)
    sig = child_signature(child)
    # Stamp the child's AssertSet onto its compiled flow: the Grow subgraph body is `child.compiled`,
    # and `expand.clone_child` reads `child_asserts` off the cloned child to carry the
    # boundary asserts (eval'd eagerly in the CALL grow residual) + re-home the post asserts onto __out.
    child.compiled.child_asserts = child.asserts
    errors: list[str] = []
    if is_map:
        node: Node = MapNode(
            desc.id,
            flow_id=desc.call,
            flow_version=None,
            child=child.compiled,
            child_inputs=child.input,
            child_asserts=child.asserts,
            child_source=child.source,        # render-only; the CLI nested-error traceback
            parallel=desc.parallel,           # inert; carried for the over case only
            title=desc.node_name,
        )
        node.params = _sink_params(desc.inputs)
        wiring = {"over": desc.over, **_sink_wiring(desc.inputs)}   # over FIRST (edge order)
        if sig.output is None:
            errors.append(
                f"map node {desc.id!r}: callable {desc.call!r} must declare exactly one "
                f"output value — a mapped call (over:) returns list[U] of that single output"
            )
        else:
            node.output_type = _list_of(sig.output)
    else:
        node = CallNode(
            desc.id,
            flow_id=desc.call,
            flow_version=None,
            child=child.compiled,
            child_inputs=child.input,
            child_asserts=child.asserts,
            child_source=child.source,        # render-only; the CLI nested-error traceback
            title=desc.node_name,
        )
        node.params = _sink_params(desc.inputs)
        wiring = _sink_wiring(desc.inputs)
        node.output_type = sig.output         # re-export the child's single codomain value
    _check_child_bindings(desc.id, desc.call, node.params, sig, errors)
    if errors:
        raise LoadError("\n  ".join(errors))
    return node, wiring


def _type_field_names(typ: Optional[Type]) -> Optional[set[str]]:
    """The per-field NAMES of a record `Type`, or `None` when `typ` is not a
    closed record (no `fields:` — a scalar/list/opaque codomain). Used for the
    body-output field-NAME-set half of the loop's `'a -> 'a` contract."""
    if typ is None or typ.fields is None:
        return None
    return set(typ.fields.keys())


def build_loop_node(desc: LoopDescriptor, resolver: ChildResolver) -> tuple[Node, dict[str, Any]]:
    """Build a `kind: loop` node — a `('a -> 'a) -> 'a -> 'a` driver — resolve-and-bake.

    `loop` is a higher-order driver: the body subflow (`call:`) reads a subset of the
    carried record (`inputs:`) and returns the WHOLE next carried record. Returns
    `(node, wiring)` (the split): the node carries `params`; the flow owns the carried-record
    seed sources in `wiring`. Mirrors `build_call_node` — resolves the callable, derives its
    `ChildSignature`, bakes the compiled child + input decls, stamps `output_type` (the body
    codomain, which IS the carried record type), and enforces the FIELD-NAME half of the
    contract + slice legality (`while:` present, `max:` present). The deep TYPE-level half
    (body output Type == carried record Type) is the loader pass `check_loop_type_contract`,
    which needs the assembled parent producers/flow-input types.
    """
    child = _resolve_child(desc.id, desc.call, resolver)
    sig = child_signature(child)
    # Stamp the child's AssertSet onto its compiled flow (the Grow subgraph body), exactly as
    # build_call_node — clone_child reads child_asserts off the cloned body child.
    child.compiled.child_asserts = child.asserts
    node = LoopNode(
        desc.id,
        flow_id=desc.call,
        flow_version=None,
        child=child.compiled,
        child_inputs=child.input,
        child_asserts=child.asserts,
        child_source=child.source,        # render-only; the CLI nested-error traceback
        title=desc.node_name,
    )
    node.params = _sink_params(desc.inputs)
    wiring = _sink_wiring(desc.inputs)
    node.output_type = sig.output          # the body codomain == the carried record type

    errors: list[str] = []
    # Predicate slice legality: exactly one of `while:`/`until:`/`times:` selects the loop's
    # driver. `while`/`until` are PREDICATE loops (record-scoped boolean, `max:` runaway guard);
    # `times: N` is a fixed COUNT (no predicate, `max:` redundant since the count bounds it). Bake
    # `predicate_kind`/`predicate`/`times`/`max_iters` per the present kind below.
    present = [k for k, v in (("while", desc.while_), ("until", desc.until_), ("times", desc.times)) if v is not None]
    if len(present) != 1:
        errors.append(
            f"loop node {desc.id!r}: exactly one of while:/until:/times: is required "
            f"(got {sorted(present)})"
        )
    elif present[0] in ("while", "until"):
        kind = present[0]
        node.predicate_kind = kind
        node.predicate = desc.while_ if kind == "while" else desc.until_
        node.max_iters = desc.max
        if desc.max is None:
            errors.append(f"loop node {desc.id!r}: `max:` is required (runaway guard)")
        elif not isinstance(desc.max, int) or isinstance(desc.max, bool):
            # `max:` bounds the iteration COUNT, so it must be a plain integer. The descriptor is a
            # bare dataclass (its `Optional[int]` annotation isn't enforced), so a YAML string/float
            # passes through and would blow up the `< 1` compare or the runtime guard, and a YAML
            # bool (an int subclass in Python) would silently read as 0/1. Reject any non-int type.
            errors.append(f"loop node {desc.id!r}: `max:` must be an integer (got {desc.max!r})")
        elif desc.max < 1:
            # A runaway guard below 1 is nonsensical: the guard bounds the iteration COUNT, so it
            # must permit at least one body run. Reject at the boundary rather than letting `max: 0`
            # run one iteration and then fail with a confusing "exceeded max (0)".
            errors.append(f"loop node {desc.id!r}: `max:` must be >= 1 (got {desc.max})")
    elif present[0] == "times":
        # `times: N` is a fixed count: exactly N runs, no predicate. `max_iters` doubles as the
        # count N so the runtime guard needs no `times`-special case. Same int/bool/`>= 1` guard as
        # `max:` — the count must be a plain integer permitting at least one run.
        node.predicate_kind = "times"
        node.predicate = None
        node.times = desc.times
        node.max_iters = desc.times
        if not isinstance(desc.times, int) or isinstance(desc.times, bool):
            errors.append(f"loop node {desc.id!r}: `times:` must be an integer (got {desc.times!r})")
        elif desc.times < 1:
            errors.append(f"loop node {desc.id!r}: `times:` must be >= 1 (got {desc.times})")
        if desc.max is not None:
            errors.append(
                f"loop node {desc.id!r}: max: is redundant with times: (times bounds the count)"
            )

    # The `'a -> 'a` FIELD-NAME contract (types are the loader pass). The carried record's
    # keys are the seed `inputs:` names. The body OUTPUT field NAMES must equal them (the
    # whole next state threads back); the body INPUT NAMES must be a subset (it reads a
    # slice of the carried record). A non-record body output (no fields) can't match a
    # multi-field carried record — flag it too.
    carried = set(desc.inputs or {})
    out_names = _type_field_names(sig.output)
    if out_names is None:
        errors.append(
            f"loop node {desc.id!r}: body callable {desc.call!r} must declare a record "
            f"output whose field names equal the carried record {sorted(carried)}"
        )
    elif out_names != carried:
        errors.append(
            f"loop node {desc.id!r}: body output fields {sorted(out_names)} must equal the "
            f"carried record fields {sorted(carried)} (the whole next state threads back)"
        )
    body_inputs = {d.name for d in sig.inputs}
    extra = body_inputs - carried
    if extra:
        errors.append(
            f"loop node {desc.id!r}: body input(s) {sorted(extra)} are not carried record "
            f"fields {sorted(carried)} (the body reads a subset of the carried record)"
        )
    # `while:`/`until:` predicate scope: the predicate is evaluated against the carried record
    # (record-scoped `evaluate_when_record`), where the ONLY resolvable names are the carried
    # keys — an undeclared name silently resolves falsy and spins the loop to `max`. Reject a
    # predicate whose ref head isn't a carried field at load, so a typo is loud not a runaway.
    # `times:` has no predicate, so skip it.
    predicate_src = desc.while_ or desc.until_
    if predicate_src:
        pred_kw = "while" if desc.while_ else "until"
        # The unified condition ref-walk: works for any spelling (bare `score >= 0.8`,
        # mixed `${score} >= 0.8`, whole `${score >= 0.8}`), unlike a flat `${...}` regex
        # which read a whole-span predicate as one bogus head and a bare one as no heads.
        try:
            heads = {r.split(".", 1)[0] for r in condition_refs(predicate_src)}
        except ExpressionError as exc:
            errors.append(
                f"loop node {desc.id!r}: `{pred_kw}:` is not a valid boolean expression: {exc}"
            )
            heads = set()
        unknown = sorted(heads - carried)
        if unknown:
            errors.append(
                f"loop node {desc.id!r}: `{pred_kw}:` references unknown name(s) {unknown} — "
                f"the predicate reads the carried record {sorted(carried)}"
            )
    if errors:
        raise LoadError("\n  ".join(errors))
    return node, wiring


def check_loop_type_contract(
    nodes: dict,
    producers: dict[str, Type],
    flow_input_types: dict[str, Type],
    flow_wiring: dict[str, dict[str, Any]],
    node_lines: Optional[dict] = None,
) -> None:
    """The TYPE-level half of a `loop`'s `'a -> 'a` contract (a loader post-build pass).

    `build_loop_node` checked only the field-NAME sets (it lacks the resolved carried-record
    Types — the seed sources' types depend on the parent producers/flow inputs assembled
    LATER). This pass, run after `check_ref_map_types`, resolves each carried field's Type
    from its bound seed source (`${<id>.output}`/`${input.X}` over the parent producers +
    flow-input types, exactly as `check_ref_map_types` via `_output_value_type`) and checks
    it is structurally compatible (`types_compatible`, C-EQUIV) with the body OUTPUT's
    same-named field Type (the body `output_type`). A literal/coalesce/opaque seed source,
    or an opaque body output field, stays lenient (skipped). Loud + located at the loop
    node's `.yaml` line on a same-names/different-TYPES mismatch."""
    for nid, node in nodes.items():
        if not node.is_loop:
            continue
        body_out = node.output_type
        # A non-record / absent body output was already caught by the field-name half in
        # build_loop_node; nothing typed to compare against here.
        if body_out is None or body_out.fields is None:
            continue
        for field_name, src in flow_wiring.get(nid, {}).items():
            body_field = body_out.fields.get(field_name)
            if body_field is None:
                continue  # name mismatch caught in build; nothing to type-check
            source = _output_value_type(src, producers, flow_input_types)
            if source is None:
                continue  # literal/coalesce/opaque seed source -> lenient
            if not types_compatible(source, body_field):
                raise LoadError(
                    f"loop node {nid!r}: carried field {field_name!r} — body output is "
                    f"{body_field.kind.value!r}, carried source is "
                    f"{source.kind.value!r} (the body must return the same 'a it reads)",
                    line=(node_lines or {}).get(nid),
                )


def check_wiring_parity(
    nodes: dict, flow_wiring: dict[str, dict[str, Any]], node_lines: Optional[dict] = None
) -> None:
    """Load-time key parity: every node's `flow.wiring` keys must equal its declared
    `params` names PLUS the kind's reserved keys (`until` for a timed WAIT, `over` for a mapped
    `call`).

    Closes the duplicate-param hole by construction (a nested-dict wiring cannot hold a duplicate
    param) and turns an orphan source (a wiring key with no param), a missing source (a param with no
    wiring), or a node absent from `flow.wiring` entirely (silent {} bind -> lost input) into a
    located `LoadError`. Engine bind-time reserved keys (the mapped-call `over` / WAIT `until` the
    eval_node seam pre-resolves) are never author wiring and do not appear here."""
    lines = node_lines or {}
    for nid, node in nodes.items():
        expected = {p.name for p in (node.params or [])} | node.reserved_wiring_keys()
        actual = set(flow_wiring.get(nid, {}))
        if actual != expected:
            raise LoadError(
                f"node {nid!r}: flow.wiring keys {sorted(actual)} != declared params + reserved "
                f"{sorted(expected)} (wiring/params key parity)",
                line=lines.get(nid),
            )


def _resolve_child(node_id: str, flow_id: str, resolver: ChildResolver) -> Any:
    """Resolve a callable to its `LoadedFlow` via the resolver, loud on failure."""
    try:
        return resolver(flow_id, None)
    except LoadError:
        raise
    except Exception as exc:  # any resolver failure -> the callable is unknown
        raise LoadError(
            f"call node {node_id!r}: could not resolve callable {flow_id!r}: {exc}"
        ) from exc
