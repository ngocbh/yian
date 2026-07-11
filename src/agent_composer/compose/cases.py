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
`compose.calls` (the shared binding-walk, for `expand_case_outputs`), `compose.errors`,
`compose.parser`. The `compose.calls` edge is acyclic (calls.py never imports cases.py; the
loader imports both). Nothing in the engine imports this back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

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
from agent_composer.compose.bindings import map_binding_strings
from agent_composer.compose.calls import (
    _to_call_descriptor,
    desugar_call_directives,
    map_outputs_strings,
)
from agent_composer.compose.errors import LoadError
from agent_composer.compose.parser import CaseDescriptor


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


# --------------------------------------------------------------------------- #
# `then:/else:` — an inline-call branch target
#
# A `case` branch may route to a FRESH owned call instead of a placed node id:
# `then: call(take, stance="pro")` (and `else:` symmetric). It desugars to a synth
# `__call_<n>` node (the same machinery as an inline-binding call); the then:/else:
# target is rewritten to that synth id (which is then both the CaseNode handle and
# the control-edge target, exactly like a plain `then: id`). Accepted grammar: a
# whole-value `call(...)` directive — a route target resolves to exactly one node id, so
# a coalesce / embedded text / dotted call (or any `${...}` span) is a located LoadError.
# Sound because the case-route veto skip-floods the non-chosen synth branch even when its
# call args carry data edges.
#
# Runs in `_assemble` AFTER `desugar_inline_calls` (sharing its `next_id` minter — else
# the ids collide) and BEFORE `expand_case_outputs`, so the synth call's args (which may
# read `${<case>.output}`) are expanded by that pass and `${<case>.output}` over THIS
# case sees the rewritten (synth-id) targets.
# --------------------------------------------------------------------------- #


def desugar_case_call_targets(
    descriptors: dict, mint: Callable[[], str], *, node_lines: "dict[str, int] | None" = None
) -> dict:
    """Rewrite each `case` then:/else: that is an inline call — a whole-value
    `call(<flow-id>, kw=...)` directive — to a synth call node id (minting its
    `CallDescriptor`, `over=None`); return the new descriptor map (augmented with the synth
    nodes). A plain node id (no `${`, not a `call(...)` directive) passes through unchanged."""
    lines = node_lines or {}
    synth: dict = {}
    new_descriptors: dict = {}
    for nid, desc in descriptors.items():
        if not isinstance(desc, CaseDescriptor):
            new_descriptors[nid] = desc
            continue
        new_cases = [
            {**c, "then": _lift_case_target(
                c.get("then"), mint, synth, host=f"case node {nid!r} then:", line=lines.get(nid))}
            for c in desc.cases
        ]
        new_else = (
            _lift_case_target(desc.else_, mint, synth, host=f"case node {nid!r} else:", line=lines.get(nid))
            if desc.else_ is not None else None
        )
        new_descriptors[nid] = replace(desc, cases=new_cases, else_=new_else)
    new_descriptors.update(synth)
    return new_descriptors


def _lift_case_target(
    target: Any, mint: Callable[[], str], synth: dict, *, host: str, line: Optional[int]
) -> Any:
    """A then:/else: target rewritten to a single node id. Accepts TWO forms and
    rejects everything else with a located `LoadError`:

    - a whole-value `call(<flow-id>, kw=...)` directive (the compile-time form) -> lift
      its synth call node(s) into `synth`, return the OUTER call's synth id;
    - a plain node id (no `${`, not a `call(...)` directive) -> passed through unchanged.

    Anything else — a coalesce / embedded text / dotted call, or any `${...}` span — is a
    `LoadError`: a route target must resolve to exactly one node id."""
    if not isinstance(target, str):
        return target  # None (handled by the caller) or a native literal
    stripped = target.strip()
    # A whole-value `call(...)` directive: recognized BEFORE the plain-id passthrough,
    # since a directive carries no `${` and would otherwise look like a bare node id.
    new_value, calls = desugar_call_directives(target, mint)
    if calls:  # inner-first: the OUTER call is calls[-1] and names the branch target
        outer = calls[-1]
        if new_value.strip() == f"${{{outer.id}.output}}":
            for c in calls:
                synth[c.id] = _to_call_descriptor(c, host=host, line=line)
            return outer.id
    if "${" not in target:
        # Not a directive and no span: a plain node id, UNLESS it is a rejected `call(...)`
        # attempt (e.g. a coalesce of directives) that the recognizer declined — those must
        # not slip through as a node id.
        if stripped.startswith("call("):
            raise LoadError(
                f"{host}: a branch target must be a node id or a single call "
                f"(got {target!r}) — not a coalesce / embedded text / dotted call",
                line=line,
            )
        return target
    # A `${...}` branch target is the retired inline `${flow(args)}` form: a route target
    # is a node id or a whole-value call(...) directive (handled above) — nothing else.
    raise LoadError(
        f"{host}: a branch target must be a node id or a call(...) directive "
        f"(got {target!r}) — an inline ${{flow(args)}} target is no longer supported",
        line=line,
    )


# --------------------------------------------------------------------------- #
# `${<case>.output}` = the taken-branch value
#
# A `case` is read as a VALUE: `${<case>.output}` resolves to the value of whichever
# branch was taken. Pure load-time sugar — it rewrites the ref to a COALESCE over the
# case's branch targets (`${t1.output | t2.output | … | else.output}`). The existing
# skip-flood makes every non-taken branch null, so the coalesce yields the taken value
# (exactly what a hand-written coalesce join uses). Runtime IR unchanged.
#
# Only a CLEAN whole-span ref `${<case>.output[.dotted.path]}` is expanded (incl.
# embedded-in-text and dotted). A case-value ref in a non-clean position — inside a
# `|` coalesce / a `:-`/`:?` default / a `when:`/`on:` condition / an assert — is a
# located `LoadError` (deferred). The `then:/else: call(...)` inline-call branch form
# needs the case-route hard-gate veto; see docs/TODO.md.
# --------------------------------------------------------------------------- #

# A clean whole-span case-output ref INTERIOR. Singular only:
# - `<id>.output[.<seg>…]` (capture group 1 = id, group 2 = dotted rest)
# Path segments are reference-path style — letter/underscore then alnum/underscore, no `-`.
_CASE_OUTPUT_INTERIOR = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_#/]*)\.output((?:\.[A-Za-z_][A-Za-z0-9_#/]*)*)$"
)


def expand_case_outputs(
    descriptors: dict,
    outputs_section: Any,
    asserts_section: list,
    *,
    node_lines: "dict[str, int] | None" = None,
    outputs_line: "int | None" = None,
    asserts_line: "int | None" = None,
) -> tuple:
    """Expand every `${<case>.output}` binding ref to a coalesce over the case's branch
    targets, returning `(new_descriptors, new_outputs)`.

    Walks every binding field (node `inputs:` / TOOL `args:` / mapped-call `over:`) and
    the flow `outputs:` bindings; rewrites a clean whole-span case-output ref to the
    branch coalesce. Rejects (located `LoadError`) a case-value ref in a non-clean
    binding position, or in a `case` condition (`on:`/`when:`) or an `assert` — those are
    not bindings. A flow with no `case` node returns its inputs unchanged."""
    case_ids = {nid for nid, d in descriptors.items() if isinstance(d, CaseDescriptor)}
    if not case_ids:
        return descriptors, outputs_section  # fast path
    lines = node_lines or {}
    targets = _case_leaf_targets(descriptors, case_ids)

    new_descriptors = {
        nid: map_binding_strings(
            d, _expander(case_ids, targets, f"node {nid!r}", lines.get(nid))
        )
        for nid, d in descriptors.items()
    }
    new_outputs = map_outputs_strings(
        outputs_section, _expander(case_ids, targets, "flow outputs", outputs_line)
    )

    # A case value in a condition / assert (not a binding) is unsupported — loud.
    for nid, d in descriptors.items():
        if isinstance(d, CaseDescriptor):
            conds = ([d.on] if d.on is not None else []) + [c.get("when") for c in d.cases]
            _reject_case_refs(conds, case_ids, f"case node {nid!r} condition", lines.get(nid))
    _reject_case_refs(asserts_section, case_ids, "assert", asserts_line)

    return new_descriptors, new_outputs


def _expander(case_ids: set, targets: dict, host: str, line: Optional[int]):
    """A `value -> value` binding transform: expand a clean WHOLE-SPAN case-output ref to
    the branch coalesce; reject a case ref in any non-clean position. Scans TOP-LEVEL
    `${…}` spans only — a case ref NESTED inside an enclosing span (e.g. a `:-` default
    value `${x:-${gate.output}}`) is caught as a non-clean ref of that span, and a `:?`
    message literal (`${x:?${gate.output}}` — not a ref) is left untouched (no
    rewrite/corruption)."""

    def expand(value: str) -> str:
        # Singular only: detect `.output` marker.
        if ".output" not in value:
            return value
        out: list = []
        i, n = 0, len(value)
        while i < n:
            if value[i] == "$" and value[i + 1 : i + 2] == "$":
                out.append("$$")  # a literal `$` — never a span start
                i += 2
            elif value[i] == "$" and value[i + 1 : i + 2] == "{":
                j = _span_end(value, i + 2)
                if j is None:  # unbalanced ${ — leave the rest for the binding parser
                    out.append(value[i:])
                    break
                out.append(_expand_span(value[i + 2 : j], case_ids, targets, host, line))
                i = j + 1
            else:
                out.append(value[i])
                i += 1
        return "".join(out)

    return expand


def _expand_span(
    interior: str, case_ids: set, targets: dict, host: str, line: Optional[int]
) -> str:
    """One TOP-LEVEL `${…}` span (its interior) -> the replacement `${…}` text.

    A clean case-output interior `outputs.<case>[.path]` -> the branch coalesce over the case's
    LEAF (non-case) targets (the path suffix re-applied per branch; a case-of-case is flattened
    by `_case_leaf_targets`). Any OTHER interior that reads a case value (via a nested
    ref / coalesce / default operand) -> a located `LoadError`. Everything else (a non-case ref,
    a `:?` message literal) is returned unchanged."""
    m = _CASE_OUTPUT_INTERIOR.match(interior.strip())
    if m is not None:
        # Singular only: group 1 = <case>, group 2 = dotted suffix.
        cid = m.group(1)
        suffix = m.group(2) or ""
        if cid in case_ids:
            leaves = targets[cid]
            if not leaves:  # no then:/else:, or a case-target cycle (the flatten dropped the arm)
                raise LoadError(
                    f"{host}: ${{{cid}.output}} — case {cid!r} has no resolvable branch value "
                    f"(a case needs a `then:` or `else:`, and its targets must not form a cycle)",
                    line=line,
                )
            # Emit the NEW node-first branch coalesce.
            return "${" + " | ".join(f"{t}.output{suffix}" for t in leaves) + "}"
    _reject_case_refs(["${" + interior + "}"], case_ids, host, line)  # non-clean read -> loud
    return "${" + interior + "}"


def _span_end(s: str, start: int) -> Optional[int]:
    """Index of the `}` closing a `${…}` span whose interior begins at `start`
    (brace-depth + quote aware), or None if unbalanced. Mirrors `expr.template`."""
    depth, quote, i, n = 1, None, start, len(s)
    while i < n:
        c = s[i]
        if quote is not None:
            if c == quote:
                quote = None
        elif c in ("'", '"'):
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _reject_case_refs(strings, case_ids: set, where: str, line: Optional[int]) -> None:
    """Raise a located `LoadError` if any string reads a `${<case>.output}` value
    (`<case>` a case node id) — the floor for the non-clean / condition / assert
    positions where the value-case is not supported."""
    for s in strings:
        if not isinstance(s, str):
            continue
        try:
            refs = expr_refs_of(s)
        except ExpressionError:
            continue  # malformed -> surfaced (located) by the relevant downstream pass
        for ref in refs:
            parts = ref.split(".")
            # Singular only: a case-id in head position of `<id>.output`.
            case_id = None
            if len(parts) >= 2 and parts[1] == "output" and parts[0] in case_ids:
                case_id = parts[0]
            if case_id is not None:
                raise LoadError(
                    f"{where}: ${{{case_id}.output}} (a case value) is only supported as a "
                    f"whole single reference in a binding — not in a coalesce / default / "
                    f"condition / assert (only the `then:/else: call(...)` value form is supported)",
                    line=line,
                )


def _case_leaf_targets(descriptors: dict, case_ids: set) -> dict:
    """case id -> its branch targets, flattened to LEAF (non-case) node ids.

    Direct targets are each `then:` (case order) then the `else:`; a target that is itself a
    `case` is flattened to ITS leaf targets (recursive, cycle-guarded — a case-target cycle
    drops that arm and surfaces as an empty-leaves `LoadError` here, or a graph cycle at DAG
    validation). Duplicates removed preserving order. Sound via the case-route veto: the nested gate
    skip-floods when the outer branch loses, so only the taken leaf is non-null in the coalesce."""
    raw: dict = {}
    for nid, d in descriptors.items():
        if isinstance(d, CaseDescriptor):
            t = [c.get("then") for c in d.cases if c.get("then") is not None]
            if d.else_ is not None:
                t.append(d.else_)
            raw[nid] = t

    def leaves(cid: str, seen: set) -> list:
        out: list = []
        for t in raw.get(cid, []):
            if t in case_ids:
                if t not in seen:
                    out.extend(leaves(t, seen | {t}))
            else:
                out.append(t)
        return out

    return {cid: _dedup(leaves(cid, {cid})) for cid in case_ids}


def _dedup(items: list) -> list:
    seen: set = set()
    out: list = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
