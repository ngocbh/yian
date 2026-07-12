"""kind=call|map — function application of a callable; descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules. `call` and `map` SHARE one frozen
`CallDescriptor` (the REF/MAP discriminator is `desc.kind`), so this module owns the
descriptor, the shared located parse, and the resolve-and-bake `build` (which branches on
`desc.kind`: `"map"` -> `MapNode`, else `CallNode`); `kinds/map.py` reuses all three with the
map field surface. Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch
routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.nodes.base import Node
from agent_composer.nodes.binding import ParamDecl
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.map import MapNode
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.compose.kinds.helpers import (
    _sink_params,
    _sink_wiring,
    check_node_keys,
    read_common_fields,
)
from agent_composer.compose.kinds.resolver import (
    ChildSignature,
    _resolve_child,
    child_signature,
)
from agent_composer.events import SourceRange

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

# Call's own field surface (was the `call` row of parser._KIND_SPECS): `call` (the callable) is
# the one required flat field. `map` reuses CallDescriptor with a wider surface (see kinds/map.py).
CALL_REQUIRED: frozenset[str] = frozenset({"call"})
CALL_ALLOWED: frozenset[str] = frozenset({"call", "inputs", "asserts"})
# Descriptor fields carrying `${...}` binding strings (the kind-blind walk in bindings.py):
# `inputs` (the call-arg bindings) and `over` (the map iteration source — always None for a
# plain `call`, so the walk skips its non-string value). Mirrors the CallDescriptor branch of
# the legacy `calls.py:map_binding_strings_in_descriptor` ladder.
CALL_BINDING_FIELDS: tuple[str, ...] = ("inputs", "over")


@dataclass(frozen=True, kw_only=True)
class CallDescriptor(NodeDescriptor):
    """kind=call|map — function application of a callable (`call:` + call-arg bindings).

    `call` names the callable, resolved **defs-first, else an external flow id** (the
    loader composes that resolution). `kind: call` is a single application (built as a plain
    `CallNode`); `kind: map` + `over: ${list}` is `List.map` — iteration with `${item}` in
    element scope (built as a `MapNode`), and `parallel:` overlaps the element runs. `inputs:`
    are the (per-element) call-arg bindings.

    `kind` is the REF/MAP discriminator: `"call"` (REF, the default) or `"map"`. The parser sets
    it from the YAML kind; the two SYNTH paths (inline-call + case desugar) keep the REF default
    (`over=None`). `build` branches on it: a `map` builds a `MapNode`, a `call` a `CallNode`.

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`; only the call/map-specific
    fields are declared here.
    """

    call: str
    inputs: dict[str, Any] = field(default_factory=dict)
    over: Optional[str] = None
    parallel: bool = False
    kind: str = "call"
    asserts: list[str] = field(default_factory=list)


def parse_call(
    node_id: str,
    reader: NodeReader,
    *,
    kind: str,
    required: frozenset[str],
    allowed: frozenset[str],
) -> CallDescriptor:
    """Shared located parse for `call` and `map` (both build a `CallDescriptor`).

    `kind` (`"call"`/`"map"`) is stamped onto the descriptor (the REF/MAP build discriminator);
    `required`/`allowed` are the kind's flat-field surface. Records the node body's `SourceRange`
    (`desc.range`) and each present field's `SourceRange` (`desc.field_ranges[field]`). Field
    legality mirrors the legacy `parser._parse_node` call/map path.
    """
    check_node_keys(node_id, kind, reader, allowed, required)
    body = reader.value

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id, "kind": kind}
    for key in allowed:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        kwargs[key] = body[key]

    asserts = kwargs.get("asserts")
    if asserts is not None and not (
        isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
    ):
        raise LoadError(
            f"node {node_id!r} (kind={kind}): `asserts:` must be a list of strings",
            range=ranges.get("asserts"),
        )

    read_common_fields(node_id, kind, reader, kwargs, ranges)

    return CallDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def parse(node_id: str, reader: NodeReader) -> CallDescriptor:
    """Read one `kind: call` node body into a `CallDescriptor` (single application)."""
    return parse_call(node_id, reader, kind="call", required=CALL_REQUIRED, allowed=CALL_ALLOWED)


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


def build(desc: CallDescriptor, *, registry=None, resolver, **_loader_hints):
    """Construct the runtime `CallNode`/`MapNode` (+ wiring) for one `CallDescriptor` —
    REF's `CallNode` (single application) or MAP's `MapNode` (`kind: map` + `over:`, mapped
    iteration) — resolve-and-bake, then stamp `.range`/`.field_ranges` from the descriptor.

    Returns `(node, wiring)` (the split): the node carries `params`; the flow owns the sources
    in `wiring`. A MAP's reserved `over:` iteration source rides the wiring under the `"over"`
    key (over-then-inputs order); a REF's wiring is just its `inputs:`. Resolves the callable
    via `resolver`, derives its `ChildSignature`, stamps `output_type` (REF re-exports the
    codomain; MAP wraps it in `list[U]`), name/arity-checks the params, and BAKES the child's
    compiled flow onto the node. The REF/MAP discriminator is `desc.kind` (`"map"` -> `MapNode`,
    else `CallNode`). `registry` is unused (call/map resolve their output type from the child
    signature, not the flow typedefs); `_loader_hints` absorbs the inline-CODE traceback
    anchors that only the CODE build reads — both ride the shared `KindSpec.build` signature.
    """
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
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring
