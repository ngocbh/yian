"""Child resolve-and-signature — the load-time callable seam shared by `call`/`map`/`loop`.

A `call`/`map`/`loop` node names a CALLABLE (a `defs:` entry or an external flow). The child
resolver is the load-time seam: `(flow_id, version) -> LoadedFlow` (the loader composes the
defs-first resolution). From the resolved child the kind builder derives its `ChildSignature` —
declared input types + single codomain `Type` — to stamp the node's `output_type` and to
name/arity- + type-check the bindings, AND bakes the child's compiled flow onto the node
(resolve-and-bake — the call graph, incl. nested flows, is fixed at compile). The bake itself
lives in each kind's own `build` (`call.py`/`loop.py`); this module supplies the shared
resolution + signature derivation they both call, plus `_output_value_type` (the ref-to-Type
resolver the loader graph-stages `ref_map_types`/`loop_contract` also read).

Like `base.py` this is a leaf of the kinds package: it imports no `kinds/__init__`
symbol and no other kind module, so `call`/`loop`, the graph-stages, and `compose.__init__`/
`loader` (the `ChildResolver` seam type) import from it at module top level without cycling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent_composer.expr import ExpressionError
from agent_composer.expr.template import Span, scan_template, template_refs
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.errors import LoadError
from agent_composer.compose.types import InputDecl


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
    for seg in fields:
        if typ is None or typ.fields is None:
            return None  # opaque producer / not a checked record -> lenient
        typ = typ.fields.get(seg)
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
