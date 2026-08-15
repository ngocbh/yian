"""Child resolve-and-signature — the load-time callable seam shared by `call`/`map`/`loop`.

A `call`/`map`/`loop` node names a CALLABLE (a `defs:` entry or an external flow). The child
resolver is the load-time seam: `(flow_id, version) -> ComposedFlow` (the loader composes the
defs-first resolution). From the resolved child the kind builder derives its `ChildSignature` —
canonical parameters + single codomain `Type` — to stamp the node's `output_type` and to
name/arity- + type-check the bindings, AND bakes the child's compiled flow onto the node
(resolve-and-bake — the call graph, incl. nested flows, is fixed at compile). The bake itself
lives in each kind's own `build` (`call.py`/`loop.py`); this module supplies the shared
resolution + signature derivation they both call, plus `_output_value_type` (the ref-to-Type
resolver the loader validate passes `validate/call_types.py`/`validate/loop_types.py` also read).

Like `base.py` this is a leaf of the nodes package: it imports no `nodes/__init__`
symbol and no other kind module, so `call`/`loop`, the validate passes, and `compose.__init__`/
`loader` (the `ChildResolver` seam type) import from it at module top level without cycling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent_composer.expr import ExpressionError
from agent_composer.expr.template import Span, scan_template, template_refs
from agent_composer.nodes.contracts import Parameter, binding_source
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.helpers.errors import LoadError


@dataclass(frozen=True)
class ChildSignature:
    """A child flow's compile-time signature: canonical parameters + the single
    codomain `Type` (the value the child returns; `None` when it declares no outputs).

    `inputs` are the child's canonical `Parameter`s (the name/arity check reads
    their names + required/default policy). `output` is the child's single value:
    one declared output -> that output's Type; >=2 -> a closed-record Type keyed by
    name (the same arity rule `terminal_output` assembles: one value out)."""

    inputs: list[Parameter]
    output: Optional[Type]


# resolver(flow_id, flow_version) -> ComposedFlow (the load-time child seam; duck-typed
# on `.compiled`/`.input` so `build` need not import `loader`). `flow_version` is an
# opaque str tag (a `uses:` ref's `@<version>`); an in-file def passes None.
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
    from_ = binding_source(from_)
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
    for seg in fields:
        if typ is None or typ.fields is None:
            return None  # opaque producer / not a checked record -> lenient
        typ = typ.fields.get(seg)
    return typ


def child_signature(loaded: Any) -> ChildSignature:
    """A loaded child `ComposedFlow` -> its compile-time `ChildSignature`.

    Duck-typed on `loaded.input` (the canonical parameters) + `loaded.compiled`
    so `build` need not import `loader` (which imports `build`). The codomain `Type`
    follows `terminal_output`'s arity rule over the flow outputs: one output -> its value
    Type; >=2 -> a closed record keyed by name (every output is a field; an output whose
    value Type can't be resolved is present but opaque)."""
    producers: dict[str, Type] = {
        nid: node.output_type
        for nid, node in loaded.compiled.nodes.items()
        if node.output_type is not None
    }
    flow_input_types = {
        parameter.name: parameter.contract.type
        for parameter in loaded.input
        if parameter.contract is not None
    }
    outs = list(loaded.compiled.outputs)
    if not outs:
        output: Optional[Type] = None
    elif len(outs) == 1:
        output = _output_value_type(outs[0].binding, producers, flow_input_types)
    else:
        fields = {
            o.name: _output_value_type(o.binding, producers, flow_input_types)
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


def _callable_display(flow_id: str, child_label: str | None = None) -> str:
    """Return an author-facing callable description without changing named errors."""
    return child_label or f"callable {flow_id!r}"


def _resolve_child(
    node_id: str,
    flow_id: str,
    resolver: ChildResolver,
    child_label: str | None = None,
) -> Any:
    """Resolve a callable to its `ComposedFlow` via the resolver, loud on failure."""
    try:
        return resolver(flow_id, None)
    except LoadError:
        raise
    except Exception as exc:  # any resolver failure -> the callable is unknown
        raise LoadError(
            f"call node {node_id!r}: could not resolve "
            f"{_callable_display(flow_id, child_label)}: {exc}"
        ) from exc
