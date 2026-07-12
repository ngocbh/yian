"""The `case` desugar: a `case` node -> a strict `CaseNode`.

Pattern-match compilation, the OCaml analogue of lowering `match e with | p -> e1
| _ -> e2` to nested conditionals. A `case` node carries no inputs and no built leaf
Node; this pass lowers it to the existing strict `CaseNode` plus the control + data
edges — with NO new node kind. Two surface forms:

- **searched** (`cases: [{when: "<bool>", then: <t>}]`, `else: <e>`): scan each `when:`
  for its distinct `${...}` refs -> allocate one `CaseNode` input `__rN` per distinct
  ref (`source` = the original `${ref}`) -> rewrite the `when:` to the bare local
  `${__rN}`. Each case routes to its `then:` (the edge handle); `else:` routes to the
  shipped internal `default` handle (the surface key is `else:`).

- **`on:`** (`on: ${ref}`, `cases: [{when: <value>, then: <t>}]`): one input
  `__on = ${ref}`; each `when: <value>` becomes `${__on} == "<value>"`.

Because the desugared `when:`s interpolate ONLY the node-local `__rN`/`__on` names, the
strict `CaseNode` evaluates them via the EXISTING `evaluate_when_record` against its
bound input record — no new evaluator, and the strict declared-input check
accepts them (the `__rN`/`__on` ARE the declared inputs).

The data edges produced here RECONCILE the earlier provisional `<case>:<n>` groups to the
allocated `__rN`/`__on` input names (so `score -> gate` carries `input_group="__r0"`);
the caller replaces the case node's provisional edges with these.

**Exhaustiveness:** when `on:` names an ENUM producer — the dotted `on:` ref
resolves through the producing node's `output_type` (walking dotted fields into a record
to reach a `Type` with `.tags`) — every tag must be covered by a case `when:` value OR a
present `else:`. A missing tag with no `else:` is a `LoadError`.

Imports flow down/peer only: `expr` (ref scan), `nodes` (Case/CaseNode/ParamDecl),
`compile.model` (Edge) + `compile.validation._walk_record_fields` (the dotted-field walk,
a representation-neutral leaf checker), `typesys` (Type), and the compose peers
`compose.errors`, `compose.parse.file`. Nothing in the engine imports this back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from agent_composer.compile.model import Edge
from agent_composer.expr import (
    ExpressionError,
    binding_co_skips,
    condition_refs,
    expr_refs_of,
    rewrite_condition_refs,
)
from agent_composer.nodes.binding import ParamDecl
from agent_composer.nodes.case import DEFAULT_HANDLE, Case, CaseNode
from agent_composer.typesys.values import Type
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.parse.file import CaseDescriptor


@dataclass(frozen=True)
class CaseDesugar:
    """The lowering of one `case` node: the gate `CaseNode` + its edges.

    `node` is the desugared `CaseNode` (its `.params` = the `__rN`/`__on` names). `wiring` is the
    flow-owned `{__rN|__on -> source}` map (the gate's sources live on `CompiledFlow.wiring`).
    `data_edges` are the reconciled input edges (proper `__rN`/`__on` `input_group`);
    `control_edges` are the `gate -> <then|else target>` edges.
    """

    node: CaseNode
    wiring: dict[str, Any]
    data_edges: list[Edge]
    control_edges: list[Edge]


def _outputs_producer(ref: str) -> Optional[str]:
    """The producing node id of a ref path. Singular only: accepts
    `<id>.output[.…]`. Legacy `outputs.<id>` is rejected at parse time."""
    parts = ref.split(".")
    if len(parts) >= 2 and parts[1] == "output":
        return parts[0]
    return None


def _refs_in(expr: str) -> list[str]:
    """The distinct reference paths a `when:` expression reads, in first-seen order.

    Routes through `condition_refs` (parse the WHOLE `when:` value, all three spellings —
    bare / mixed / whole-span) so it agrees with `_rewrite_when`, which rewrites the same
    parsed leaves. Raises `ExpressionError` on a malformed `when:` (the caller frames it
    as a located `LoadError`)."""
    out: list[str] = []
    for path in condition_refs(expr):
        if path not in out:
            out.append(path)
    return out


def _quote(value: str) -> str:
    """A `when:` match value quoted as a string literal for `${__on} == "<v>"`.

    The `when:`/`asserts:` grammar's STRING terminal is `/"[^"]*"/ | /'[^']*'/` with
    NO escape support (the evaluator strips the outer quotes verbatim), so we cannot
    escape — we pick the quote style the value allows: double-quote when it has no `"`,
    single-quote when it has no `'`. A value containing BOTH quote characters is
    unrepresentable in this grammar → a loud, located load error rather than a
    route-time crash. (Backslashes pass through `[^"]*` fine — no escaping needed.)
    """
    s = str(value)
    if '"' not in s:
        return f'"{s}"'
    if "'" not in s:
        return f"'{s}'"
    raise LoadError(
        f"case `on:` match value {s!r} contains both ' and \" — unrepresentable in a "
        f"`when:` string literal (no escape support); rename the matched value"
    )


def _data_edges(node_id: str, wiring: dict[str, Any]) -> list[Edge]:
    """Reconciled data edges: one per `${<id>.output}` ref a `__rN`/`__on` source reads.

    Reads the flow-owned `wiring` (`{__rN|__on -> source}`); the `input_group` is the
    param name (`__rN`/`__on`) — replacing the earlier provisional `<case>:<n>` groups so
    per-input readiness keys on the desugared input.
    """
    edges: list[Edge] = []
    counts: dict[str, int] = {}
    for name, source in wiring.items():
        if not isinstance(source, str):
            continue
        try:
            refs = expr_refs_of(source)
        except ExpressionError:
            continue  # malformed refs surface, located, in the ref-wiring pass
        for ref in refs:
            producer = _outputs_producer(ref)
            if producer is None:
                continue
            i = counts.get(producer, 0)
            counts[producer] = i + 1
            edges.append(
                Edge(
                    id=f"{producer}->{node_id}#{i}",
                    from_=producer,
                    to=node_id,
                    input_group=name,
                    # a `when:`/`on:` ref can carry an escape (`:-`/`:?`); a
                    # co-skipping condition would otherwise wrongly skip-flood the gate.
                    optional=not binding_co_skips(source),
                )
            )
    return edges


def _control_edges(node_id: str, handle_targets: list[tuple[str, str]]) -> list[Edge]:
    """`gate -> target` edges, each carrying its `source_handle` (case id or default)."""
    edges: list[Edge] = []
    counts: dict[str, int] = {}
    for handle, target in handle_targets:
        i = counts.get(target, 0)
        counts[target] = i + 1
        edges.append(
            Edge(
                id=f"{node_id}->{target}#{i}",
                from_=node_id,
                to=target,
                source_handle=handle,
            )
        )
    return edges


def _resolve_on_type(on_ref: str, producers: dict[str, Type]) -> Optional[Type]:
    """The `Type` an `on: ${<id>.output[.<field>…]}` ref resolves to, else None.

    Walks the producing node's `output_type` dotted-field by dotted-field
    (`_walk_record_fields`) into a record to reach the named field. A
    non-node-first head / opaque producer / non-record step -> None (lenient: no
    exhaustiveness check). Imported here to avoid coupling the walk logic twice.
    """
    from agent_composer.compile.validation import _walk_record_fields

    try:
        refs = expr_refs_of(on_ref)
    except ExpressionError:
        return None
    if len(refs) != 1:
        return None
    parts = refs[0].split(".")
    # Singular only: `<id>.output[.<field>…]`.
    if len(parts) >= 2 and parts[1] == "output":
        producer_id = parts[0]
        fields = parts[2:]
    else:
        return None
    typ = producers.get(producer_id)
    # Reuse the dotted-field walk only to FAIL on a bad dotted field; then re-walk to the Type.
    if _walk_record_fields(typ, fields, refs[0]) is not None:
        return None  # bad field -> the reference-wiring pass reports it; skip exhaustiveness
    for f in fields:
        if typ is None or typ.fields is None:
            return None
        typ = typ.fields.get(f)
    return typ


def _check_exhaustive(
    desc: CaseDescriptor, covered: set[str], producers: dict[str, Type]
) -> None:
    """Enum-exhaustiveness for the `on:` form.

    Resolves the `on:` ref to a `Type`; if it carries `.tags` (an enum), every tag must
    be a `when:` match value OR be covered by a present `else:`. A missing tag with no
    `else:` is a `LoadError`. A non-enum / unresolved producer is lenient (no check).
    """
    if desc.on is None or desc.else_ is not None:
        return  # searched form, or an else: that satisfies coverage
    typ = _resolve_on_type(desc.on, producers)
    if typ is None or typ.tags is None:
        return  # not a checked enum producer -> lenient
    missing = sorted(typ.tags - covered)
    if missing:
        raise LoadError(
            f"case node {desc.id!r} on {desc.on}: non-exhaustive — enum tag(s) "
            f"{missing} not covered by a case `when:` or an `else:`"
        )


def desugar_case(desc: CaseDescriptor, producers: dict[str, Type]) -> CaseDesugar:
    """Lower one `case` descriptor to a strict `CaseNode` + control + data edges.

    `producers` maps producer-node id -> its `output_type` (for `on:` enum
    exhaustiveness). Raises `LoadError` on a malformed case or a non-exhaustive enum.
    """
    if desc.on is not None:
        result = _desugar_on(desc)
    else:
        result = _desugar_searched(desc)
    # Exhaustiveness uses the case `when:` VALUES (the on:-form match labels).
    _check_exhaustive(desc, _on_covered(desc), producers)
    return result


def _on_covered(desc: CaseDescriptor) -> set[str]:
    """The set of enum tags an on:-form case explicitly matches (its `when:` values)."""
    if desc.on is None:
        return set()
    return {str(c.get("when")) for c in desc.cases if c.get("when") is not None}


def _desugar_on(desc: CaseDescriptor) -> CaseDesugar:
    """`on: ${ref}` form -> one `__on` param + a `${__on} == "<value>"` per case."""
    cases: list[Case] = []
    handle_targets: list[tuple[str, str]] = []
    for c in desc.cases:
        value = c.get("when")
        then = c.get("then")
        if value is None or then is None:
            raise LoadError(
                f"case node {desc.id!r}: each on:-form case needs a `when:` value and a `then:`"
            )
        cases.append(Case(handle=then, when=f'${{__on}} == {_quote(value)}'))
        handle_targets.append((then, then))
    if desc.else_ is not None:
        handle_targets.append((DEFAULT_HANDLE, desc.else_))
    node = CaseNode(desc.id, cases, title=desc.node_name)
    node.params = [ParamDecl(name="__on")]
    wiring = {"__on": desc.on}
    return CaseDesugar(
        node=node,
        wiring=wiring,
        data_edges=_data_edges(desc.id, wiring),
        control_edges=_control_edges(desc.id, handle_targets),
    )


def _desugar_searched(desc: CaseDescriptor) -> CaseDesugar:
    """Searched form -> one `__rN` input per distinct `${...}` ref; `when:` rewritten."""
    # Allocate one local per distinct ref across ALL when:s (first-seen order).
    ref_to_local: dict[str, str] = {}
    for c in desc.cases:
        when = c.get("when")
        if not isinstance(when, str):
            raise LoadError(
                f"case node {desc.id!r}: each searched case needs a string `when:` expression"
            )
        try:
            refs = _refs_in(when)
        except ExpressionError as exc:
            raise LoadError(
                f"case node {desc.id!r}: `when:` {when!r} is not a valid boolean expression: {exc}"
            ) from exc
        for ref in refs:
            if ref not in ref_to_local:
                ref_to_local[ref] = f"__r{len(ref_to_local)}"

    wiring = {local: f"${{{ref}}}" for ref, local in ref_to_local.items()}

    cases: list[Case] = []
    handle_targets: list[tuple[str, str]] = []
    for c in desc.cases:
        when = c["when"]
        then = c.get("then")
        if then is None:
            raise LoadError(f"case node {desc.id!r}: each searched case needs a `then:`")
        rewritten = _rewrite_when(when, ref_to_local)
        cases.append(Case(handle=then, when=rewritten))
        handle_targets.append((then, then))
    if desc.else_ is not None:
        handle_targets.append((DEFAULT_HANDLE, desc.else_))

    node = CaseNode(desc.id, cases, title=desc.node_name)
    node.params = [ParamDecl(name=local) for local in ref_to_local.values()]
    return CaseDesugar(
        node=node,
        wiring=wiring,
        data_edges=_data_edges(desc.id, wiring),
        control_edges=_control_edges(desc.id, handle_targets),
    )


def _rewrite_when(when: str, ref_to_local: dict[str, str]) -> str:
    """Rewrite each reference in `when` to its bare local `__rN`, honouring all three
    condition spellings. A `${<ref>}` (mixed) becomes `${__rN}`, a bare `<ref>`
    (whole-span / bare) becomes `__rN`; the surrounding expression is preserved. A ref
    with no allocated local is left untouched. Shares `rewrite_condition_refs` with
    `_refs_in`, so extraction and rewrite parse the SAME leaves."""
    return rewrite_condition_refs(when, lambda path: ref_to_local.get(path))


def reconcile_case_edges(
    inferred_edges: list[Edge], desugars: dict[str, CaseDesugar]
) -> list[Edge]:
    """Merge the inferred data edges with the case desugars (the reconciliation pass).

    The data-edge pass emits PROVISIONAL data edges into each `case` node with `<case>:<n>` groups
    (the edges exist, but keyed by a placeholder). For each desugared case, DROP those
    provisional incoming edges and substitute the desugar's reconciled `data_edges`
    (keyed by `__rN`/`__on`) plus its `control_edges`. Non-case edges pass through.
    `desugars` is keyed by case node id.
    """
    case_ids = set(desugars)
    kept = [e for e in inferred_edges if e.to not in case_ids]
    for d in desugars.values():
        kept.extend(d.data_edges)
        kept.extend(d.control_edges)
    return kept

