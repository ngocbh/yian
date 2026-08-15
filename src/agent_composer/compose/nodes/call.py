"""kind=call|map — function application of a callable; descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules. `call` and `map` share the resolve-and-bake
`build` (which branches on `isinstance(self, MapDescriptor)`: a `map` -> `MapNode`, else `CallNode`);
`map.py`'s `MapDescriptor` subclasses `CallDescriptor`, adding only the map field surface
(`over`/`parallel`) and the `kind="map"` class attr. Both inherit the template `parse` from
`NodeDescriptor`. Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import Parameter
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.map import MapNode
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.compose.nodes.resolver import (
    ChildSignature,
    _callable_display,
    _resolve_child,
    child_signature,
)

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
    element scope (built as a `MapNode`). `parallel:` is retained but currently inert; run-level
    workers control concurrency. `inputs:` are the (per-element) call-arg bindings.

    The REF/MAP discriminator is the descriptor CLASS: a plain `CallDescriptor` is a REF (built
    as a `CallNode`); a `MapDescriptor` (its subclass, in `map.py`) is a MAP (built as a
    `MapNode`). `build` branches on `isinstance(self, MapDescriptor)`. The `kind` CLASS ATTR
    (`"call"` here, `"map"` on the subclass) is the surface string for dispatch/diagnostics; it is
    no longer an instance field. The two SYNTH paths (inline-call + case desugar) mint plain
    `CallDescriptor`s (REF, `over=None`).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) and `inputs` from `NodeDescriptor`; only the
    call/map-specific fields are declared here. `over`/`parallel` are fields on the shared base
    class (a plain `call` leaves them at `None`/`False`); only `map` lists them in its `allowed`.
    `child_label` is compiler-owned diagnostic metadata for an anonymous inline child. It is
    deliberately absent from `allowed`, never authorable, and excluded from descriptor equality.

    Class attrs (call's own surface): `kind="call"`; `allowed` adds `call`/`inputs`/`asserts` to
    the commons; `required` is `{"call"}`; `binding_fields=("inputs","over")` — the call-arg
    bindings plus the map iteration source (a plain `call` leaves `over=None`, so the kind-blind
    walk skips it).
    """

    kind = "call"
    allowed = NodeDescriptor.allow("call", "inputs", "asserts")
    required = NodeDescriptor.require("call")
    binding_fields = ("inputs", "over")

    call: str
    child_label: str | None = field(default=None, compare=False)
    over: Optional[str] = None
    parallel: bool = False
    asserts: list[str] = field(default_factory=list)

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `CallNode`/`MapNode` (+ wiring) for this descriptor —
        REF's `CallNode` (single application) or MAP's `MapNode` (`kind: map` + `over:`, mapped
        iteration) — resolve-and-bake, then stamp `.range`/`.field_ranges` via `_finish`.

        Returns a `BuildResult` (the split): the node carries `params`; the flow owns the sources
        in `wiring`. A MAP's reserved `over:` iteration source rides the wiring under the `"over"`
        key (over-then-inputs order); a REF's wiring is just its `inputs:`. Resolves the callable
        via `ctx.resolver`, derives its `ChildSignature`, stamps `output_type` (REF re-exports the
        codomain; MAP wraps it in `list[U]`), name/arity-checks the params, and BAKES the child's
        compiled flow onto the node. The REF/MAP discriminator is the descriptor CLASS
        (`MapDescriptor` -> `MapNode`, else `CallNode`). `ctx.registry` is unused (call/map resolve
        their output type from the child signature, not the flow typedefs); the inline-CODE anchors
        on `ctx` are read only by the CODE build.
        """
        resolver = ctx.resolver
        # The construction boundary: this reads the descriptor CLASS (not a compiled kind enum)
        # to pick the runtime class. `MapDescriptor` is imported locally to avoid a module-top
        # cycle (map.py imports CallDescriptor from here). Kind identity is BORN here — downstream
        # compile passes then dispatch on the node's traits, never on `node.kind`.
        from agent_composer.compose.nodes.map import MapDescriptor

        is_map = isinstance(self, MapDescriptor)
        if is_map and "over" in (self.inputs or {}):
            # Reserved-name guard: a MAP's `over` is the iteration source, carried under
            # the reserved wiring key — an `inputs:` param named `over` would collide.
            raise LoadError(
                f"map node {self.id!r}: input name 'over' is reserved (the iteration "
                f"source); rename the `inputs:` entry"
            )
        child = _resolve_child(
            self.id,
            self.call,
            resolver,
            self.child_label,
        )
        sig = child_signature(child)
        child_display = _callable_display(self.call, self.child_label)
        # The child's asserts ride its compiled boundary nodes (lowered at the child's own
        # build): PRE/boundary asserts on its START, POST-terminal on its END. `clone_child`
        # deep-copies the whole child subgraph, so those lowered asserts survive the splice.
        errors: list[str] = []
        if is_map:
            node: Node = MapNode(
                self.id,
                flow_id=self.call,
                flow_version=None,
                child=child.compiled,
                child_inputs=sig.inputs,
                child_source=child.source,        # render-only; the CLI nested-error traceback
                parallel=self.parallel,           # inert; carried for the over case only
                title=self.node_name,
            )
            node.params = self.sink_params()
            wiring = {"over": self.over, **self.sink_wiring()}   # over FIRST (edge order)
            if sig.output is None:
                errors.append(
                    f"map node {self.id!r}: {child_display} must declare exactly one "
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
                child_inputs=sig.inputs,
                child_source=child.source,        # render-only; the CLI nested-error traceback
                title=self.node_name,
            )
            node.params = self.sink_params()
            wiring = self.sink_wiring()
            node.output_type = sig.output         # re-export the child's single codomain value
        _check_child_bindings(
            self.id,
            self.call,
            node.params,
            sig,
            errors,
            self.child_label,
        )
        if errors:
            raise LoadError("\n  ".join(errors))
        return self._finish(node, wiring)


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
    params: list[Parameter],
    sig: ChildSignature,
    errors: list[str],
    child_label: str | None = None,
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
                f"input of {_callable_display(flow_id, child_label)}"
            )
    for decl in sig.inputs:
        if decl.required and decl.name not in bound:
            errors.append(
                f"call node {node_id!r}: required callable input {decl.name!r} is unbound"
            )
