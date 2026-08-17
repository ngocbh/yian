"""Desugar pass: `${<case>.output}` binding refs -> branch coalesce over case leaves.

`expand_case_outputs` (below) walks every binding field (any node's `input:` /
mapped-call `over:`) and the flow `output:` bindings, and rewrites a clean WHOLE-SPAN
case-output ref `${<case>.output[.path]}` to a coalesce `${t1.output[.path] | t2.output… }`
over the case's LEAF (non-case) branch targets. A case-value ref in any NON-clean binding
position (coalesce / default operand / embedded text), or in a `case` condition
(`on:`/`when:`) or an `assert`, is a located `LoadError` — those are not bindings.

Sound via the case-route veto: the losing branch skip-floods (its target output is null),
so only the taken leaf is non-null in the emitted coalesce.

Runs AFTER `desugar_case_call_targets` (so a `then:/else: call(...)` synth id is a leaf
target here) and, like the other desugars, BEFORE the build loop.

Imports flow DOWN/peer only: `expr` (the ref-walk), `compose.nodes.bindings` (the
kind-blind binding-field walk), `compose.desugar.inline_calls` (the `output:` walk —
within-phase peer), `compose.helpers.errors`, `compose.parse.file` (CaseDescriptor).
Only the loader (and the compose package root re-export) import this back.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from agent_composer.diagnostics import SourceRange
from agent_composer.expr import ExpressionError, expr_refs_of
from agent_composer.compose.nodes.bindings import map_binding_strings
from agent_composer.compose.desugar.inline_calls import map_outputs_strings
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.parse.file import CaseDescriptor
from agent_composer.nodes.contracts import Assertion


# A clean whole-span case-output ref INTERIOR. Singular only:
# - `<id>.output[.<seg>…]` (capture group 1 = id, group 2 = dotted rest)
# Path segments are reference-path style — letter/underscore then alnum/underscore, no `-`.
_CASE_OUTPUT_INTERIOR = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_#/]*)\.output((?:\.[A-Za-z_][A-Za-z0-9_#/]*)*)$"
)


def expand_case_outputs(
    descriptors: dict,
    outputs_section: Any,
    asserts_section: list[Assertion],
    *,
    outputs_range: Optional[SourceRange] = None,
    assertion_range_at: Optional[Callable[[int], Optional[SourceRange]]] = None,
    diagnostics: LoadDiagnostics | None = None,
) -> tuple:
    """Expand every `${<case>.output}` binding ref to a coalesce over the case's branch
    targets, returning `(new_descriptors, new_outputs)`.

    Walks every binding field (any node's `input:` / a mapped call's `over:`) and
    the flow `output:` bindings; rewrites a clean whole-span case-output ref to the
    branch coalesce. Rejects (located `LoadError`) a case-value ref in a non-clean
    binding position, or in a `case` condition (`on:`/`when:`) or an `assert` — those are
    not bindings. Descriptor field/item ranges locate node-owned sites;
    ``outputs_range`` and ``assertion_range_at`` locate section-owned sites. A flow with
    no `case` node returns its inputs unchanged."""
    diagnostics = diagnostics or LoadDiagnostics()
    case_ids = {nid for nid, d in descriptors.items() if isinstance(d, CaseDescriptor)}
    if not case_ids:
        return descriptors, outputs_section  # fast path
    targets = _case_leaf_targets(descriptors, case_ids)

    new_descriptors = {}
    for nid, descriptor in descriptors.items():

        def expand_binding(
            field: str,
            key: str,
            value: str,
            *,
            _descriptor=descriptor,
            _nid=nid,
        ) -> str:
            binding_range = _binding_range(_descriptor, field, key)
            return _expander(
                case_ids,
                targets,
                f"node {_nid!r}",
                range=binding_range,
                diagnostics=diagnostics,
            )(value)

        new_descriptors[nid] = map_binding_strings(descriptor, expand_binding)

    new_outputs = map_outputs_strings(
        outputs_section,
        _expander(
            case_ids,
            targets,
            "flow outputs",
            range=outputs_range,
            diagnostics=diagnostics,
        ),
    )

    # A case value in a condition / assert (not a binding) is unsupported — loud.
    for nid, d in descriptors.items():
        if isinstance(d, CaseDescriptor):
            if d.on is not None:
                condition_range = _field_range(d, "on")
                _reject_case_refs(
                    [d.on],
                    case_ids,
                    f"case node {nid!r} condition",
                    range=condition_range,
                    diagnostics=diagnostics,
                )
            for index, case in enumerate(d.cases):
                condition_range = _when_range(d, index)
                _reject_case_refs(
                    [case.get("when")],
                    case_ids,
                    f"case node {nid!r} condition",
                    range=condition_range,
                    diagnostics=diagnostics,
                )
    for index, assertion in enumerate(asserts_section or []):
        assertion_range = (
            assertion_range_at(index) if assertion_range_at is not None else None
        )
        _reject_case_refs(
            [assertion],
            case_ids,
            "assert",
            range=assertion_range,
            diagnostics=diagnostics,
        )

    return new_descriptors, new_outputs


def _binding_range(
    descriptor: Any,
    field: str,
    key: str,
) -> Optional[SourceRange]:
    """Return the narrowest available range for a descriptor binding."""
    field_ranges = getattr(descriptor, "field_ranges", None) or {}
    if field == "inputs":
        return (
            field_ranges.get(("input", key))
            or field_ranges.get(("field", "inputs"))
            or getattr(descriptor, "range", None)
        )
    return field_ranges.get(("field", field)) or getattr(descriptor, "range", None)


def _field_range(descriptor: Any, name: str) -> Optional[SourceRange]:
    """Return a scalar field's range, falling back to its descriptor range."""
    field_ranges = getattr(descriptor, "field_ranges", None) or {}
    return field_ranges.get(("field", name)) or getattr(descriptor, "range", None)


def _when_range(descriptor: Any, index: int) -> Optional[SourceRange]:
    """Return a case item's positional `when:` range with progressively coarser fallbacks."""
    when_ranges = getattr(descriptor, "when_ranges", None) or []
    if index < len(when_ranges) and when_ranges[index] is not None:
        return when_ranges[index]
    return _field_range(descriptor, "cases")


def _expander(
    case_ids: set,
    targets: dict,
    host: str,
    *,
    range: Optional[SourceRange] = None,
    diagnostics: LoadDiagnostics,
):
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
                out.append(
                    _expand_span(
                        value[i + 2 : j],
                        case_ids,
                        targets,
                        host,
                        range=range,
                        diagnostics=diagnostics,
                    )
                )
                i = j + 1
            else:
                out.append(value[i])
                i += 1
        return "".join(out)

    return expand


def _expand_span(
    interior: str,
    case_ids: set,
    targets: dict,
    host: str,
    *,
    range: Optional[SourceRange] = None,
    diagnostics: LoadDiagnostics,
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
            if (
                not leaves
            ):  # no then:/else:, or a case-target cycle (the flatten dropped the arm)
                raise diagnostics.error(
                    f"{host}: ${{{cid}.output}} — case {cid!r} has no resolvable branch value "
                    f"(a case needs a `then:` or `else:`, and its targets must not form a cycle)",
                    range=range,
                )
            # Emit the NEW node-first branch coalesce.
            return "${" + " | ".join(f"{t}.output{suffix}" for t in leaves) + "}"
    _reject_case_refs(
        ["${" + interior + "}"],
        case_ids,
        host,
        range=range,
        diagnostics=diagnostics,
    )  # non-clean read -> loud
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


def _reject_case_refs(
    strings,
    case_ids: set,
    where: str,
    *,
    range: Optional[SourceRange] = None,
    diagnostics: LoadDiagnostics,
) -> None:
    """Raise a located `LoadError` if any string reads a `${<case>.output}` value
    (`<case>` a case node id) — the floor for the non-clean / condition / assert
    positions where the value-case is not supported."""
    for value in strings:
        s = value.expression if isinstance(value, Assertion) else value
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
                raise diagnostics.error(
                    f"{where}: ${{{case_id}.output}} (a case value) is only supported as a "
                    f"whole single reference in a binding — not in a coalesce / default / "
                    f"condition / assert (only the `then:/else: call(...)` value form is supported)",
                    range=range,
                )


def _case_leaf_targets(descriptors: dict, case_ids: set) -> dict:
    """case id -> its branch targets, flattened to LEAF (non-case) node ids.

    Direct targets are each `then:` (case order) then the `else:`; a target that is itself a
    `case` is flattened to ITS leaf targets (recursive, cycle-guarded — a case-target cycle
    drops that arm and surfaces as an empty-leaves `LoadError` here, or a graph cycle at DAG
    validation). Duplicates removed preserving order. Sound via the case-route veto: the nested gate
    skip-floods when the outer branch loses, so only the taken leaf is non-null in the coalesce.
    """
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
