"""kind=call|map — function application of a callable; descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules. `call` and `map` SHARE one frozen
`CallDescriptor` (the REF/MAP discriminator is the instance `kind` field), so this module owns
the descriptor, the shared located `parse_call`, and the resolve-and-bake `build` (which branches
on `self.kind`: `"map"` -> `MapNode`, else `CallNode`); `kinds/map.py` reuses all three with the
map field surface. Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch
routes here.

`call` inherits the template `parse` from `NodeDescriptor` (its class attrs `allowed`/`required`/
`binding_fields` describe its own surface). `map` still calls the shared `parse_call` this step —
it becomes a `MapDescriptor(CallDescriptor)` on the template in a later step; until then the
instance `kind` field carries the REF/MAP distinction rather than a class attr.
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


@dataclass(frozen=True, kw_only=True)
class CallDescriptor(NodeDescriptor):
    """kind=call|map — function application of a callable (`call:` + call-arg bindings).

    `call` names the callable, resolved **defs-first, else an external flow id** (the
    loader composes that resolution). `kind: call` is a single application (built as a plain
    `CallNode`); `kind: map` + `over: ${list}` is `List.map` — iteration with `${item}` in
    element scope (built as a `MapNode`), and `parallel:` overlaps the element runs. `inputs:`
    are the (per-element) call-arg bindings.

    The instance `kind` field is the REF/MAP discriminator: `"call"` (REF, the default) or
    `"map"`. `call` parses via the inherited template (which leaves `kind` at its `"call"`
    default); `map` parses via the shared `parse_call` (which stamps `kind="map"`); the two SYNTH
    paths (inline-call + case desugar) keep the REF default (`over=None`). `build` branches on it:
    a `map` builds a `MapNode`, a `call` a `CallNode`. (`kind` stays an INSTANCE field — not a
    class attr — precisely so one `CallDescriptor` class can carry both discriminator values
    until `map` splits into its own subclass.)

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) and `inputs` from `NodeDescriptor`; only the
    call/map-specific fields are declared here.

    Class attrs (call's own surface): `allowed` adds `call`/`inputs`/`asserts` to the commons;
    `required` is `{"call"}`; `binding_fields=("inputs","over")` — the call-arg bindings plus the
    map iteration source (a plain `call` leaves `over=None`, so the kind-blind walk skips it).
    """

    allowed = NodeDescriptor.allow("call", "inputs", "asserts")
    required = NodeDescriptor.require("call")
    binding_fields = ("inputs", "over")

    call: str
    over: Optional[str] = None
    parallel: bool = False
    kind: str = "call"
    asserts: list[str] = field(default_factory=list)

    def build(self, *, registry=None, resolver=None, **_loader_hints):
        """Construct the runtime `CallNode`/`MapNode` (+ wiring) for this descriptor —
        REF's `CallNode` (single application) or MAP's `MapNode` (`kind: map` + `over:`, mapped
        iteration) — resolve-and-bake, then stamp `.range`/`.field_ranges` via `_finish`.

        Returns `(node, wiring)` (the split): the node carries `params`; the flow owns the sources
        in `wiring`. A MAP's reserved `over:` iteration source rides the wiring under the `"over"`
        key (over-then-inputs order); a REF's wiring is just its `inputs:`. Resolves the callable
        via `resolver`, derives its `ChildSignature`, stamps `output_type` (REF re-exports the
        codomain; MAP wraps it in `list[U]`), name/arity-checks the params, and BAKES the child's
        compiled flow onto the node. The REF/MAP discriminator is the instance `kind`
        (`"map"` -> `MapNode`, else `CallNode`). `registry` is unused (call/map resolve their
        output type from the child signature, not the flow typedefs); `_loader_hints` absorbs the
        inline-CODE traceback anchors that only the CODE build reads — both ride the shared build
        signature.
        """
        # The construction boundary: this reads the RAW authored `kind` STRING (not the
        # compiled kind enum) to pick the runtime class. Kind identity is BORN here — downstream
        # compile passes then dispatch on the node's traits, never on `node.kind`.
        is_map = self.kind == "map"
        if is_map and "over" in (self.inputs or {}):
            # Reserved-name guard: a MAP's `over` is the iteration source, carried under
            # the reserved wiring key — an `inputs:` param named `over` would collide.
            raise LoadError(
                f"map node {self.id!r}: input name 'over' is reserved (the iteration "
                f"source); rename the `inputs:` entry"
            )
        child = _resolve_child(self.id, self.call, resolver)
        sig = child_signature(child)
        # Stamp the child's AssertSet onto its compiled flow: the Grow subgraph body is `child.compiled`,
        # and `expand.clone_child` reads `child_asserts` off the cloned child to carry the
        # boundary asserts (eval'd eagerly in the CALL grow residual) + re-home the post asserts onto __out.
        child.compiled.child_asserts = child.asserts
        errors: list[str] = []
        if is_map:
            node: Node = MapNode(
                self.id,
                flow_id=self.call,
                flow_version=None,
                child=child.compiled,
                child_inputs=child.input,
                child_asserts=child.asserts,
                child_source=child.source,        # render-only; the CLI nested-error traceback
                parallel=self.parallel,           # inert; carried for the over case only
                title=self.node_name,
            )
            node.params = self.sink_params()
            wiring = {"over": self.over, **self.sink_wiring()}   # over FIRST (edge order)
            if sig.output is None:
                errors.append(
                    f"map node {self.id!r}: callable {self.call!r} must declare exactly one "
                    f"output value — a mapped call (over:) returns list[U] of that single output"
                )
            else:
                node.output_type = _list_of(sig.output)
        else:
            node = CallNode(
                self.id,
                flow_id=self.call,
                flow_version=None,
                child=child.compiled,
                child_inputs=child.input,
                child_asserts=child.asserts,
                child_source=child.source,        # render-only; the CLI nested-error traceback
                title=self.node_name,
            )
            node.params = self.sink_params()
            wiring = self.sink_wiring()
            node.output_type = sig.output         # re-export the child's single codomain value
        _check_child_bindings(self.id, self.call, node.params, sig, errors)
        if errors:
            raise LoadError("\n  ".join(errors))
        return self._finish(node, wiring)


def parse_call(
    node_id: str,
    reader: NodeReader,
    *,
    kind: str,
    required: frozenset[str],
    allowed: frozenset[str],
) -> CallDescriptor:
    """Shared located parse for `map` (and, until it inherits the template, any REF caller).

    `kind` (`"call"`/`"map"`) is stamped onto the descriptor (the REF/MAP build discriminator);
    `required`/`allowed` are the kind's flat-field surface. Records the node body's `SourceRange`
    (`desc.range`) and each present field's `SourceRange` (`desc.field_ranges[field]`). `call`
    parses via the inherited `NodeDescriptor.parse` template instead; `map` still routes here.
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
