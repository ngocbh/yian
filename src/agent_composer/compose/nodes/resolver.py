"""Child resolve-and-signature — the load-time callable seam shared by `call`/`map`/`loop`.

A `call`/`map`/`loop` node names a CALLABLE (a `defs:` entry or an external flow). The child
resolver is the load-time seam: `(flow_id, version) -> ComposedFlow` (the loader composes the
defs-first resolution). From the resolved child the kind builder derives its `ChildSignature` —
canonical parameters + single codomain `ValueContract` — to stamp the node's
`output_contract` and to
name/arity- + type-check the bindings, AND bakes the child's compiled flow onto the node
(resolve-and-bake — the call graph, incl. nested flows, is fixed at compile). The bake itself
lives in each kind's own `build` (`call.py`/`loop.py`); this module supplies the shared
resolution + signature derivation they both call. Compose output-binding contract inference
lives one layer lower in `helpers.output_contracts` and is reused by validation directly.

Like `base.py` this is a leaf of the nodes package: it imports no `nodes/__init__`
symbol and no other kind module, so `call`/`loop`, `compose.__init__`, and `loader`
(the `ChildResolver` seam type) import from it at module top level without cycling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.compose.helpers.output_contracts import (
    combine_output_contracts,
    resolve_output_binding_contract,
)
from agent_composer.diagnostics import SourceRange
from agent_composer.nodes.contracts import Parameter, ValueContract


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
        output = resolve_output_binding_contract(
            outs[0].binding,
            producers,
            flow_inputs,
        )
    else:
        fields = {
            o.name: resolve_output_binding_contract(
                o.binding,
                producers,
                flow_inputs,
            )
            for o in outs
        }
        output = combine_output_contracts(fields)
    return ChildSignature(inputs=list(loaded.input), output=output)


def _callable_display(flow_id: str, child_label: str | None = None) -> str:
    """Return an author-facing callable description without changing named errors."""
    return child_label or f"callable {flow_id!r}"


def _resolve_child(
    node_id: str,
    flow_id: str,
    resolver: ChildResolver,
    child_label: str | None = None,
    *,
    diagnostics: LoadDiagnostics | None = None,
    range: SourceRange | None = None,
) -> Any:
    """Resolve a callable to its `ComposedFlow` via the resolver, loud on failure."""
    diagnostics = diagnostics or LoadDiagnostics()
    try:
        return resolver(flow_id, None)
    except LoadError:
        raise
    except Exception as exc:  # any resolver failure -> the callable is unknown
        raise diagnostics.error(
            f"call node {node_id!r}: could not resolve "
            f"{_callable_display(flow_id, child_label)}: {exc}",
            range=range,
        ) from exc
