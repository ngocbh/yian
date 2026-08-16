"""Child resolve-and-signature — the load-time callable seam shared by `call`/`map`/`loop`.

A `call`/`map`/`loop` node names a CALLABLE (a `defs:` entry or an external flow). The child
resolver is the load-time seam: `(flow_id, version) -> ComposedFlow` (the loader composes the
defs-first resolution). From the resolved child the kind builder derives its `ChildSignature` —
canonical parameters + the exact-or-partial codomain contract — to stamp a CALL/LOOP's
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
from agent_composer.nodes.contracts import (
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
)


@dataclass(frozen=True)
class ChildSignature:
    """A child flow's compile-time signature: canonical parameters + the single
    codomain `OutputContract` and its declared terminal outputs.

    `inputs` are the child's canonical `Parameter`s (the name/arity check reads
    their names + required/default policy). `declared_outputs` retains one
    `(name, optional contract)` pair per terminal declaration, so absent, opaque, and
    proven results remain distinct without duplicating a names list. `output` is the
    child's single codomain: one declared output re-exports that binding's exact or
    partial contract; >=2 produces an exact or partial closed record keyed by name;
    `None` means no output or no statically known shape. This follows the same arity
    rule `terminal_output` assembles."""

    inputs: list[Parameter]
    output: OutputContract | None
    declared_outputs: tuple[tuple[str, OutputContract | None], ...] = ()

    def __post_init__(self) -> None:
        """Reject contradictory declared-output evidence when callers provide it."""
        if self.output is not None and type(self.output) not in (
            ValueContract,
            PartialRecordContract,
        ):
            raise TypeError(
                "child output must be ValueContract, PartialRecordContract, or None, got "
                f"{type(self.output).__name__}"
            )
        if type(self.declared_outputs) is not tuple:
            raise TypeError("declared child outputs must be a tuple")
        if not self.declared_outputs:
            return  # preserves the legacy two-argument source-free constructor
        names: set[str] = set()
        for pair in self.declared_outputs:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError(
                    "declared child outputs must be (name, optional contract) tuples"
                )
            name, contract = pair
            if type(name) is not str or not name:
                raise TypeError("declared child output names must be non-empty strings")
            if name in names:
                raise ValueError(f"duplicate declared child output {name!r}")
            names.add(name)
            if contract is not None and type(contract) not in (
                ValueContract,
                PartialRecordContract,
            ):
                raise TypeError(
                    "declared child output contracts must be ValueContract, "
                    "PartialRecordContract, or None, got "
                    f"{type(contract).__name__} for {name!r}"
                )
        expected = (
            self.declared_outputs[0][1]
            if len(self.declared_outputs) == 1
            else combine_output_contracts(dict(self.declared_outputs))
        )
        if self.output != expected:
            raise ValueError(
                "child output contract contradicts its declared output evidence"
            )

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return declared terminal names derived from the canonical output pairs."""
        return tuple(name for name, _contract in self.declared_outputs)


# resolver(flow_id, flow_version) -> ComposedFlow (the load-time child seam; duck-typed
# on `.compiled`/`.input` so `build` need not import `loader`). `flow_version` is an
# opaque str tag (a `uses:` ref's `@<version>`); an in-file def passes None.
ChildResolver = Callable[[str, Optional[str]], Any]


def child_signature(loaded: Any) -> ChildSignature:
    """A loaded child `ComposedFlow` -> its compile-time `ChildSignature`.

    Duck-typed on `loaded.input` (the canonical parameters) + `loaded.compiled`
    so `build` need not import `loader` (which imports `build`). The codomain contract
    follows `terminal_output`'s arity rule over the flow outputs: one output re-exports
    its exact or partial contract; >=2 produces an exact or partial closed record keyed
    by name."""
    producers: dict[str, OutputContract] = {}
    for nid, node in loaded.compiled.nodes.items():
        contract = node.output_contract
        if contract is None:
            continue
        if type(contract) not in (ValueContract, PartialRecordContract):
            raise TypeError(
                "child output producer contracts must be ValueContract, "
                "PartialRecordContract, or None values, "
                f"got {type(contract).__name__} for {nid!r}"
            )
        producers[nid] = contract
    flow_inputs = {
        parameter.name: parameter.contract
        for parameter in loaded.input
        if parameter.contract is not None
    }
    outs = list(loaded.compiled.outputs)
    declared_outputs = tuple(
        (
            declared.name,
            resolve_output_binding_contract(
                declared.binding,
                producers,
                flow_inputs,
            ),
        )
        for declared in outs
    )
    if not declared_outputs:
        output: OutputContract | None = None
    elif len(declared_outputs) == 1:
        output = declared_outputs[0][1]
    else:
        output = combine_output_contracts(dict(declared_outputs))
    return ChildSignature(
        inputs=list(loaded.input),
        output=output,
        declared_outputs=declared_outputs,
    )


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
