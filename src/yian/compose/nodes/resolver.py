"""Child resolve-and-signature — the load-time callable seam shared by `call`/`map`/`loop`.

A `call`/`map`/`loop` node names a CALLABLE (a `defs:` entry or an external flow). The child
resolver is the load-time seam: `(flow_id, version) -> ComposedFlow` (the loader composes the
defs-first resolution). From the resolved child the kind builder derives its `ChildSignature` —
canonical parameters + the exact-or-partial codomain contract — to stamp a CALL's
`output_contract` (a LOOP's, or the returned fields of its merged carried contract) and to
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

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional

from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.output_contracts import (
    combine_output_contracts,
    resolve_output_binding_contract,
)
from yian.defaults import current
from yian.diagnostics import SourceRange
from yian.nodes.contracts import (
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
)
from yian.nodes.end import EndNode
from yian.typesys.types import BUILTIN_TYPE_NAMES


@dataclass(frozen=True)
class ChildSignature:
    """A child flow's compile-time signature: canonical parameters + the single
    codomain `OutputContract` and its declared terminal outputs.

    `inputs` are the child's canonical `Parameter`s (the name/arity check reads
    their names + required/default policy). `declared_outputs` retains one
    `(name, optional contract)` pair per terminal declaration, so absent, opaque, and
    proven results remain distinct without duplicating a names list. `returns_record`
    says whether the child's END is in record mode (`output: {...}`). `output` is the
    child's single codomain: a record child's is an exact or partial closed record keyed
    by every declared name (even one or none); a value child's re-exports its one
    declared output's exact or partial contract; `None` means no output or no
    statically known shape."""

    inputs: list[Parameter]
    output: OutputContract | None
    declared_outputs: tuple[tuple[str, OutputContract | None], ...] = ()
    returns_record: bool = False

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
        if not self.declared_outputs and not self.returns_record:
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
        if self.returns_record:
            expected = combine_output_contracts(dict(self.declared_outputs))
        elif len(self.declared_outputs) > 1:
            raise ValueError(
                "a child that does not return a record declares at most one output"
            )
        else:
            expected = self.declared_outputs[0][1]
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
    follows the child's END: a record END gives an exact or partial closed record keyed by
    every declared name; a value END re-exports its one output's exact or partial contract,
    or `None` without one. A `T(...)` constructor in an output is typed through END's type
    table, which holds the child's own typedefs the outputs use.

    Args:
        loaded (`ComposedFlow`):
            The resolved child flow.

    Returns:
        `ChildSignature`: The child's parameters, codomain contract, and declared outputs.

    Raises:
        TypeError: If the child's END is not an `EndNode`, or a producer contract has an
            unexpected type.
    """
    end = loaded.compiled.nodes.get(EndNode.ID)
    if not isinstance(end, EndNode):
        raise TypeError(f"child END must be an EndNode, got {type(end).__name__}")
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
    # `isinstance` may also put a builtin scalar name in the table; no constructor uses one.
    constructor_types = {
        name: typ for name, typ in end.types.items() if name not in BUILTIN_TYPE_NAMES
    }
    declared_outputs = tuple(
        (
            declared.name,
            resolve_output_binding_contract(
                declared.binding,
                producers,
                flow_inputs,
                type_fn=constructor_types.get,
            ),
        )
        for declared in outs
    )
    output: OutputContract | None
    if end.returns_record:
        output = combine_output_contracts(dict(declared_outputs))
    elif declared_outputs:
        output = declared_outputs[0][1]
    else:
        output = None
    return ChildSignature(
        inputs=list(loaded.input),
        output=output,
        declared_outputs=declared_outputs,
        returns_record=end.returns_record,
    )


def _callable_display(flow_id: str, child_label: str | None = None) -> str:
    """Return an author-facing callable description without changing named errors."""
    return child_label or f"callable {flow_id!r}"


# Calls being resolved around the current one in the file being loaded.
_ACTIVE_CALLS: ContextVar[int] = ContextVar("_ACTIVE_CALLS", default=0)


@contextmanager
def call_nesting_scope() -> Iterator[None]:
    """Count active calls from zero while one flow file loads.

    Each file counts its own calls; a call into another file is checked against the
    built callee's depth instead.

    Yields:
        `None`: Control to the file's load.
    """
    token = _ACTIVE_CALLS.set(0)
    try:
        yield
    finally:
        _ACTIVE_CALLS.reset(token)


def call_nesting(flow: Any) -> int:
    """Return the longest chain of static child flows under a compiled flow.

    Kind-blind: child flows come from `Node.origin_children()`. The walk is iterative and
    visits each flow object once, so a deep or shared child neither overflows the stack
    nor costs one visit per path.

    Args:
        flow (`CompiledFlow`):
            The flow to measure; duck-typed on `.nodes` and each node's
            `origin_children()`.

    Returns:
        `int`:
            The number of nested bodies on the longest chain; 0 for a flow with no child
            flow.
    """
    # Keyed by id() and local to this walk, which holds every visited flow alive.
    depths: dict[int, int] = {}
    children: dict[int, list[Any]] = {}
    stack: list[Any] = [flow]
    while stack:
        current = stack[-1]
        key = id(current)
        if key in depths:
            stack.pop()
            continue
        if key not in children:
            children[key] = [
                child
                for node in current.nodes.values()
                for child in node.origin_children()
            ]
            stack.extend(c for c in children[key] if id(c) not in depths)
            continue
        stack.pop()
        depths[key] = max((1 + depths[id(c)] for c in children[key]), default=0)
    return depths[id(flow)]


def _resolve_child(
    node_id: str,
    flow_id: str,
    resolver: ChildResolver,
    child_label: str | None = None,
    *,
    diagnostics: LoadDiagnostics | None = None,
    range: SourceRange | None = None,
) -> Any:
    """Resolve a callable to its `ComposedFlow` via the resolver, loud on failure.

    Args:
        node_id (`str`):
            The calling node's id, named in messages.
        flow_id (`str`):
            The callable the node names.
        resolver (`ChildResolver`):
            The load-time child resolver.
        child_label (`str | None`, *optional*, defaults to `None`):
            Author-facing name of the callable; `None` names it by `flow_id`.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            The context errors are raised in; `None` uses a fresh one.
        range (`SourceRange | None`, *optional*, defaults to `None`):
            Location of the call in the owning document.

    Returns:
        `ComposedFlow`:
            The resolved child flow.

    Raises:
        `LoadError`:
            Raised when the callable can't be resolved, or the call nests bodies deeper
            than `current().max_call_nesting` levels.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    display = _callable_display(flow_id, child_label)
    limit = current().max_call_nesting

    def too_deep(levels: str) -> LoadError:
        return diagnostics.error(
            f"call node {node_id!r}: calling {display} nests calls {levels} deep, past the "
            f"call nesting limit of {limit} (each call, map, or loop body is a "
            "level); flatten the chain",
            range=range,
        )

    depth = _ACTIVE_CALLS.get()
    # Checked before resolving: building the callee would recurse another level.
    if depth >= limit:
        raise too_deep(f"more than {limit} levels")
    token = _ACTIVE_CALLS.set(depth + 1)
    try:
        child = resolver(flow_id, None)
    except LoadError:
        raise
    except Exception as exc:  # any resolver failure -> the callable is unknown
        raise diagnostics.error(
            f"call node {node_id!r}: could not resolve {display}: {exc}",
            range=range,
        ) from exc
    finally:
        _ACTIVE_CALLS.reset(token)
    # A callee built earlier (a cached def, an imported file) never ran on this count.
    levels = depth + 1 + call_nesting(child.compiled)
    if levels > limit:
        raise too_deep(f"{levels} levels")
    return child
