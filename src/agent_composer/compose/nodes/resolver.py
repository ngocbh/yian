"""Child resolve-and-signature — the load-time callable seam shared by `call`/`map`/`loop`.

A `call`/`map`/`loop` node names a CALLABLE (a `defs:` entry or an external flow). The child
resolver is the load-time seam: `(flow_id, version) -> ComposedFlow` (the loader composes the
defs-first resolution). From the resolved child the kind builder derives its `ChildSignature` —
canonical parameters + single codomain `ValueContract` — to stamp the node's
`output_contract` and to
name/arity- + type-check the bindings, AND bakes the child's compiled flow onto the node
(resolve-and-bake — the call graph, incl. nested flows, is fixed at compile). The bake itself
lives in each kind's own `build` (`call.py`/`loop.py`); this module supplies the shared
resolution + signature derivation they both call, plus the output-binding resolvers the loader
validation passes also read.

Like `base.py` this is a leaf of the nodes package: it imports no `nodes/__init__`
symbol and no other kind module, so `call`/`loop`, the validate passes, and `compose.__init__`/
`loader` (the `ChildResolver` seam type) import from it at module top level without cycling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent_composer.expr import ExpressionError
from agent_composer.expr.template import Span, scan_template, template_refs
from agent_composer.nodes.contracts import (
    ContractOrigins,
    Parameter,
    ValueContract,
    binding_source,
)
from agent_composer.typesys.values import Type, ValueKind
from agent_composer.compose.helpers.errors import LoadError


@dataclass(frozen=True)
class ChildSignature:
    """A child flow's compile-time signature: canonical parameters + the single
    codomain `ValueContract` (the value the child returns; `None` when it declares
    no outputs).

    `inputs` are the child's canonical `Parameter`s (the name/arity check reads
    their names + required/default policy). `output` is the child's single value:
    one declared output -> that output's contract; >=2 -> a closed-record contract
    keyed by name (the same arity rule `terminal_output` assembles: one value out)."""

    inputs: list[Parameter]
    output: Optional[ValueContract]


# resolver(flow_id, flow_version) -> ComposedFlow (the load-time child seam; duck-typed
# on `.compiled`/`.input` so `build` need not import `loader`). `flow_version` is an
# opaque str tag (a `uses:` ref's `@<version>`); an in-file def passes None.
ChildResolver = Callable[[str, Optional[str]], Any]


def _project_output_contract(
    contract: ValueContract,
    fields: list[str],
) -> ValueContract | None:
    """Project a nested record contract and rebase its origin paths to the field."""
    if not fields:
        return contract
    typ = contract.type
    for field in fields:
        if typ.fields is None:
            return None
        typ = typ.fields.get(field)
        if typ is None:
            return None
    prefix = tuple(fields)
    entries = tuple(
        (path[len(prefix):], origin)
        for path, origin in contract.origins.entries
        if len(path) > len(prefix) and path[:len(prefix)] == prefix
    )
    return ValueContract(
        type=typ,
        origins=ContractOrigins(
            default=contract.origins.at(prefix),
            entries=entries,
        ),
    )


def _output_value_contract(
    from_: Any,
    producers: dict[str, ValueContract],
    flow_inputs: dict[str, ValueContract],
) -> ValueContract | None:
    """Resolve one output binding to its canonical contract, preserving provenance."""
    from_ = binding_source(from_)
    if not isinstance(from_, str):
        return None
    try:
        segments = scan_template(from_)
    except ExpressionError:
        return None
    spans = [segment for segment in segments if isinstance(segment, Span)]
    if not spans:
        return None
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return ValueContract(Type.scalar(ValueKind.STRING))
    refs = template_refs(segments)
    if len(refs) != 1:
        return None
    tree = segments[0].tree
    is_direct_reference = (
        getattr(tree, "data", None) == "refcall"
        and not any(
            getattr(child, "data", None) == "call_suffix"
            for child in getattr(tree, "children", ())
        )
    )
    parts = refs[0].split(".")
    head = parts[0]
    if head == "input" and len(parts) == 2:
        contract = flow_inputs.get(parts[1])
        if contract is not None and type(contract) is not ValueContract:
            raise TypeError(
                "flow input contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {parts[1]!r}"
            )
    elif len(parts) >= 2 and parts[1] == "output":
        producer_id = parts[0]
        fields = parts[2:]
        contract = producers.get(producer_id)
        if contract is not None and type(contract) is not ValueContract:
            raise TypeError(
                "output producer contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {producer_id!r}"
            )
        if contract is not None:
            contract = _project_output_contract(contract, fields)
    else:
        return None
    if contract is None:
        return None
    return contract if is_direct_reference else ValueContract(contract.type)


def _origin_path_sort_key(
    entry: tuple[tuple[str | int, ...], Any],
) -> tuple[tuple[int, str | int], ...]:
    """Return the same total path ordering required by `ContractOrigins`."""
    return tuple(
        (0, segment) if type(segment) is str else (1, segment)
        for segment in entry[0]
    )


def _record_output_contract(
    fields: dict[str, ValueContract | None],
) -> ValueContract:
    """Combine named output contracts into one closed-record contract."""
    origins: list[tuple[tuple[str | int, ...], Any]] = []
    for name, contract in fields.items():
        if contract is None:
            continue
        if contract.origins.default is not None:
            origins.append(((name,), contract.origins.default))
        origins.extend(
            ((name, *path), origin)
            for path, origin in contract.origins.entries
        )
    origins.sort(key=_origin_path_sort_key)
    return ValueContract(
        type=Type(
            kind=ValueKind.OBJECT,
            fields={
                name: (
                    contract.type
                    if contract is not None
                    else Type.scalar(ValueKind.NONE)
                )
                for name, contract in fields.items()
            },
            required=frozenset(
                name for name, contract in fields.items() if contract is not None
            ),
        ),
        origins=ContractOrigins(entries=tuple(origins)),
    )


def child_signature(loaded: Any) -> ChildSignature:
    """A loaded child `ComposedFlow` -> its compile-time `ChildSignature`.

    Duck-typed on `loaded.input` (the canonical parameters) + `loaded.compiled`
    so `build` need not import `loader` (which imports `build`). The codomain contract
    follows `terminal_output`'s arity rule over the flow outputs: one output -> its exact
    value contract; >=2 -> a closed-record contract keyed by name (every output is a
    field; an output whose value contract cannot be resolved is present but opaque)."""
    producers: dict[str, ValueContract] = {
        nid: node.output_contract
        for nid, node in loaded.compiled.nodes.items()
        if node.output_contract is not None
    }
    flow_inputs = {
        parameter.name: parameter.contract
        for parameter in loaded.input
        if parameter.contract is not None
    }
    outs = list(loaded.compiled.outputs)
    if not outs:
        output: Optional[ValueContract] = None
    elif len(outs) == 1:
        output = _output_value_contract(outs[0].binding, producers, flow_inputs)
    else:
        fields = {
            o.name: _output_value_contract(o.binding, producers, flow_inputs)
            for o in outs
        }
        output = _record_output_contract(fields)
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
