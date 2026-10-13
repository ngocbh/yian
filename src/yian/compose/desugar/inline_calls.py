"""Graph-stage: whole-value `call(<flow-id>, kw=...)` directive desugar.

`desugar_call_directives` (below) is the pure string→data half: a field whose WHOLE
trimmed value is exactly `call( … )` rewrites to `${<synth>.output}` and emits a
`CallDescriptor`. This module is the assembly pass that runs it over a flow's parsed
descriptors + `output:` section and installs each synth `call` node (`over=None`).
After it runs, the rest of `_assemble` (build → edge-infer → validate → DAG) sees the
synth nodes as ordinary `call` nodes — inline calls are pure load-time sugar, no new
runtime kind, the synth callee resolves through the SAME composite resolver
(defs-first) as a named call.

An unquoted value whose trimmed text starts with `call(` must be exactly one well-formed
directive; anything else (unbalanced parens, trailing text, a missing comma, a positional
or repeated keyword) is a located `LoadError` that names the problem. A quoted value
(YAML `'...'`, `"..."`, or a block scalar) is text and never a directive, so quoting is
the escape for literal `call(` text. An assert is an expression, not text, so it is held
to the directive rule however it is quoted.

The retired inline `${ f(args) }` form — a FLOW call embedded inside a `${...}` span —
is a hard-break `LoadError` pointing the author at the `call(...)` directive; a pure
builtin (`upper`, `join`, …) inside `${...}` stays legal.

Binding positions + the flow `asserts:` section: every descriptor-declared binding field
(ordinary node `input:` mappings plus a mapped call's `over:`), the flow `output:`
bindings, AND each flow
`asserts:` expression (a whole-value `call(...)` lifts to a synth node so the assert reads
its output). NOT `when:`/`on:` (the case condition grammar) and NOT a node-local
`asserts:` (those stay node-local — no inline call). A `case` descriptor carries no walked
bindings, so it is untouched.

This is also the flow-assert carrier boundary. Raw entries become one ordered
`Assertion` per source position; assembly injects an origin lookup by authored index.
Desugaring replaces only the executable expression, preserving display and any existing
or injected origin for later START/END lowering.

Three guards (loud): an inline call cannot target the compiler-private
`__inline_flow_` namespace or capture `${item}` (the synth node is a top-level node
with no map-element scope — use a named `call` node with `over:`), and a user node id
may not use the reserved synth prefix `__call_`.

Located parse metadata: each parsed host descriptor carries its own `range`/`field_ranges`
(read from its source cursor). A synth `call` node locates at the SITE it desugared from — the host
binding's own line (looked up in the host `field_ranges` by the binding key), not the host
node header — and owns an empty `field_ranges`, so a bad ref inside the directive carets the
directive rather than a host input that shares its param name. The flow `output:`/`asserts:`
sections have no descriptor, so their inline calls use a supplied section/item range.

Imports flow DOWN/peer only: `compose.parse.file` (descriptors), `compose.nodes.bindings` (the
kind-blind binding-field walk) + `compose.helpers` (errors, YAML quoting; peers), `expr` (the ref-walk + the
span flow-call detector), `diagnostics` (SourceRange). Only the loader imports this back.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import replace
from typing import Any, Callable, Optional

from yian.diagnostics import OriginRef, SourceRange
from yian.nodes.contracts import Assertion
from yian.expr import (
    ExpressionError,
    expr_refs_of,
)
from yian.expr.template import (
    InlineCall,
    _arg_source,
    _find_paren_end,
    _split_calls_aware,
    flow_call_callees_in_spans,
)
from yian.compose.nodes.bindings import map_binding_strings
from yian.compose.desugar.inline_bodies import INLINE_FLOW_PREFIX
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.typed_bindings import CALL_HEAD, QUOTED_STYLES, yaml_quoted
from yian.compose.parse.file import (
    CallDescriptor,
    NodeDescriptor,
)

# The reserved id prefix for synthesized inline-call nodes. A monotonic counter per
# flow/def keeps them unique; the collision guard reserves the namespace from users.
_SYNTH_PREFIX = "__call_"
_FLOW_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")


def _synth_id_gen() -> Callable[[], str]:
    """A fresh `__call_0`, `__call_1`, … minter (one per flow/def `_assemble` run)."""
    counter = itertools.count()
    return lambda: f"{_SYNTH_PREFIX}{next(counter)}"


def desugar_inline_calls(
    descriptors: dict[str, NodeDescriptor],
    outputs_section: Any,
    *,
    asserts_section: Optional[list] = None,
    outputs_range: Optional[SourceRange] = None,
    assertion_range_at: Optional[Callable[[int], Optional[SourceRange]]] = None,
    assertion_origin_at: Optional[Callable[[int], OriginRef | None]] = None,
    binding_style_at: Optional[Callable[[str, str, str], Optional[str]]] = None,
    output_style_at: Optional[Callable[[Any], Optional[str]]] = None,
    next_id: Optional[Callable[[], str]] = None,
    diagnostics: LoadDiagnostics | None = None,
) -> tuple:
    """Desugar every inline call in a flow's binding sites into synth `call` nodes.

    Returns `(new_descriptors, new_outputs, new_asserts)`: the descriptor map augmented with
    one synth `CallDescriptor` per inline call (the host binding rewritten to
    `${<synth>.output}`), the flow `output:` section, and ordered semantic `Assertion`
    carriers. ``assertion_origin_at`` supplies an optional origin by authored list index
    exactly when raw text crosses into its carrier. Existing origins take precedence. An
    assertion's inline calls rewrite only its executable expression; display/origin stay
    stable. A span-wrapped inline `${ f(...) }` in an assert expression
    (`${ f(...) } == x`) lifts to a synth node so the assert reads its output.

    Each synth node from a node host locates at the range of the binding it was found in (its
    own `call(...)` directive line). A synth from flow `output:` uses ``outputs_range``; one
    from `asserts:` uses the authored item returned by ``assertion_range_at``.

    ``binding_style_at(node_id, field, key)`` and ``output_style_at(key)`` (`None` for a
    single output) return a binding's YAML scalar style; a quoted value is text, so it is
    never a directive. A missing lookup or a `None` style reads as plain. An assert is an
    expression, so its quoting never makes it text.

    Raises `LoadError` on a malformed inline call (any unquoted value starting with
    `call(` that is not exactly one directive), a quoted `call(...)` holding a span, a
    `${item}` capture, or a reserved-prefix collision — located at the host node /
    `output:` / `asserts:`."""
    diagnostics = diagnostics or LoadDiagnostics()
    mint = next_id or _synth_id_gen()

    # Reserve the synth namespace: a user node id must not use the prefix (else it
    # could collide with / shadow a generated id). Checked unconditionally.
    for nid, desc in descriptors.items():
        if nid.startswith(_SYNTH_PREFIX):
            raise diagnostics.error(
                f"node id {nid!r} uses the reserved inline-call prefix {_SYNTH_PREFIX!r}",
                range=getattr(desc, "range", None),
            )

    synth: dict[str, CallDescriptor] = {}
    new_descriptors: dict[str, NodeDescriptor] = {}
    for nid, desc in descriptors.items():
        rng = getattr(desc, "range", None)
        field_ranges = getattr(desc, "field_ranges", None)
        new_desc, calls, origins = _desugar_descriptor(
            desc,
            mint,
            host=nid,
            range=rng,
            field_ranges=field_ranges,
            style_at=(
                (lambda field, key, nid=nid: binding_style_at(nid, field, key))
                if binding_style_at is not None
                else None
            ),
            diagnostics=diagnostics,
        )
        new_descriptors[nid] = new_desc
        for call in calls:
            # The synth node locates at the binding the `call(...)` came from (its own
            # `input:` line), not the host node header — so a bad ref inside the directive
            # carets the directive, not a (valid) host input that shares the param name.
            synth[call.id] = _to_call_descriptor(
                call,
                host=f"node {nid!r}",
                range=origins.get(call.id, rng),
                diagnostics=diagnostics,
            )

    new_outputs, out_calls = _desugar_outputs(
        outputs_section,
        mint,
        range=outputs_range,
        style_at=output_style_at,
        diagnostics=diagnostics,
    )
    for call in out_calls:
        synth[call.id] = _to_call_descriptor(
            call,
            host="flow outputs",
            range=outputs_range,
            diagnostics=diagnostics,
        )

    new_asserts, assert_calls, assert_ranges = _desugar_asserts(
        asserts_section,
        mint,
        assertion_range_at=assertion_range_at,
        assertion_origin_at=assertion_origin_at,
        diagnostics=diagnostics,
    )
    for call in assert_calls:
        assert_range = assert_ranges.get(call.id)
        synth[call.id] = _to_call_descriptor(
            call,
            host="assert",
            range=assert_range,
            diagnostics=diagnostics,
        )

    new_descriptors.update(synth)
    return new_descriptors, new_outputs, new_asserts


# --------------------------------------------------------------------------- #
# The flow `output:` binding walk (used by this pass AND the case-value
# expansion in `compose.desugar.case_outputs`). Descriptor binding fields go through the
# kind-blind `compose.nodes.bindings.map_binding_strings` instead.
# --------------------------------------------------------------------------- #


def map_outputs_strings(outputs: Any, fn: Callable[[str], Any]) -> Any:
    """Apply `fn` to each STRING binding in the flow `output:` section (a name→binding
    map, a bare binding string, or `None`)."""
    if isinstance(outputs, dict):
        return {k: (fn(v) if isinstance(v, str) else v) for k, v in outputs.items()}
    if isinstance(outputs, str):
        return fn(outputs)
    return outputs


def _binding_origin(
    field: str,
    key: str,
    field_ranges: Optional[dict],
    host_range: Optional[SourceRange],
) -> Optional[SourceRange]:
    """The source range a synth node desugared from binding `key` should locate at.

    Prefers the binding's OWN entry — `("input", key)` for the `inputs` mapping,
    `("field", field)` for a scalar binding field (a map's `over:`) — i.e. the line the
    `call(...)` directive was written on. Falls back to the host node's range when there is no
    stamped entry (a spanless flow, or a binding shape with no per-entry range)."""
    fr = field_ranges or {}
    if field == "inputs":
        return fr.get(("input", key)) or fr.get(("field", "inputs")) or host_range
    return fr.get(("field", field)) or host_range


def _desugar_descriptor(
    desc: NodeDescriptor,
    mint: Callable[[], str],
    *,
    host: str,
    range: Optional[SourceRange],
    field_ranges: Optional[dict],
    style_at: Optional[Callable[[str, str], Optional[str]]],
    diagnostics: LoadDiagnostics,
) -> tuple:
    """Desugar a descriptor's binding fields, returning `(new_desc, calls, origins)`.

    Walks the binding fields via `map_binding_strings`; each inline call is tagged in
    `origins` with the SOURCE RANGE of the binding it was found in (via `_binding_origin`),
    so its synth node locates at its own `call(...)` directive rather than the host header.
    Nested inner-first calls share their enclosing binding's origin. ``style_at(field,
    key)`` gives each binding's YAML scalar style (plain when absent)."""
    calls: list = []
    origins: dict[str, Optional[SourceRange]] = {}

    def mapper(field: str, key: str, value: str) -> str:
        origin = _binding_origin(field, key, field_ranges, range)
        before = len(calls)
        new_value = _desugar_value(
            value,
            mint,
            calls,
            host,
            range=origin,
            style=style_at(field, key) if style_at is not None else None,
            diagnostics=diagnostics,
        )
        for c in calls[before:]:
            origins[c.id] = origin
        return new_value

    new_desc = map_binding_strings(desc, mapper)
    return new_desc, calls, origins


def _desugar_value(
    value: str,
    mint: Callable[[], str],
    calls: list,
    host: str,
    *,
    range: Optional[SourceRange] = None,
    style: Optional[str] = None,
    quotable: bool = True,
    diagnostics: LoadDiagnostics | None = None,
) -> str:
    """Desugar one binding string. A WHOLE-VALUE `call(<flow-id>, kw=...)` directive lifts
    to a synth call node (accumulated in `calls`, the value rewritten to `${<synth>.output}`).
    A flow call embedded INSIDE a `${...}` span is the retired `${flow(args)}` form — a
    located `LoadError` pointing at the `call(...)` directive. Any other value (plain refs,
    pure builtins, literals) is returned unchanged.

    ``style`` is the value's YAML scalar style. A quoted value is text: it is never a
    directive, and a quoted `call(...)` holding a span is a load error (its author most
    likely meant the call). ``quotable`` is False for an assert, whose quoting does not
    make it text, so its errors offer no quoting fix. Maps `ExpressionError` (a malformed
    directive or an unparseable span) to a located `LoadError` at ``range``."""
    diagnostics = diagnostics or LoadDiagnostics()
    quoted = quotable and style in QUOTED_STYLES
    is_call = CALL_HEAD.match(value.strip()) is not None
    if quoted and is_call and "${" in value:
        raise diagnostics.error(
            f"{host}: a quoted `call(...)` is text, so it can't hold `${{...}}` here; "
            "drop the quotes to call the flow, or reword the text",
            range=range,
        )
    if is_call and not quoted:
        try:
            new_value, found = desugar_call_directives(value, mint)
        except ExpressionError as exc:
            hint = (
                f". If you meant the text, quote the whole value: "
                f"`{yaml_quoted(value.strip())}`"
                if quotable
                else ""
            )
            raise diagnostics.error(f"{host}: {exc}{hint}", range=range) from exc
        calls.extend(found)
        return new_value
    try:
        flow_callees = flow_call_callees_in_spans(value)
    except ExpressionError as exc:
        raise diagnostics.error(
            f"{host}: {exc}",
            range=range,
        ) from exc
    if flow_callees:
        callee = flow_callees[0]
        raise diagnostics.error(
            f"{host}: a flow call {callee!r} inside ${{...}} is not supported — write it "
            f"as the whole value with the call(...) directive: call({callee}, ...)",
            range=range,
        )
    return value


def desugar_call_directives(value: str, mint: Callable[[], str]) -> tuple:
    """Desugar a WHOLE-VALUE `call(<flow-id>, kw=...)` directive, else no-op.

    A value whose trimmed text starts with `call(` must be exactly one directive: a
    leading `call`, balanced parens, and nothing non-whitespace after the close paren.
    Its FIRST arg is the POSITIONAL flow id (a bare identifier, may be hyphenated — flow
    ids allow `-`); every remaining arg is a `name=value` keyword, each at most once.
    Each keyword value passes through `_arg_source` (so `window=30` -> int 30, a quoted
    scalar unwraps to text, a `${…}`-bearing value stays a binding string). An unquoted
    arg value starting with `call(` is ITSELF a directive, under the same rule —
    desugared inner-first.

    Returns `(new_value, calls)`: if `value` is a directive, `new_value` is the synth
    `${<id>.output}` ref and `calls` lists the `InlineCall`s (each nested call before
    its enclosing one, inner-first). If `value` does not start with `call(`, returns
    `(value, [])` unchanged (the caller then checks the value for a retired span
    flow-call). Raises `ExpressionError` naming the problem when a value starting with
    `call(` is not one directive: unbalanced parentheses, text after the close paren, a
    first arg that is not a bare flow id, a positional arg after the flow id (message
    mentions "keyword"), an unquoted `=` inside a keyword value (a missing comma), or a
    keyword passed more than once."""
    match = _match_directive(value)
    if match is None:
        return value, []
    flow_id, args_str = match
    calls: list = []
    args: dict = {}
    for pair in _split_calls_aware(args_str, ","):
        if not pair.strip():
            continue  # a trailing comma / an empty kwarg slot
        parts = _split_calls_aware(pair, "=")
        if len(parts) < 2:
            raise ExpressionError(
                f"call({flow_id}, ...) passes a positional argument {pair.strip()!r}; "
                "every argument after the flow id must be a keyword (name=value)"
            )
        name = parts[0].strip()
        if len(parts) > 2:
            raise ExpressionError(
                f"call({flow_id}, ...) keyword {name!r} has an unquoted `=` in its value "
                f"{'='.join(parts[1:]).strip()!r}; add the missing comma, or quote the value"
            )
        new_value, inner = desugar_call_directives(
            parts[1], mint
        )  # inner-first recursion
        calls.extend(inner)
        if name in args:
            raise ExpressionError(
                f"call({flow_id}, ...) passes keyword {name!r} more than once"
            )
        args[name] = _arg_source(new_value)
    cid = mint()
    calls.append(InlineCall(id=cid, callee=flow_id, args=args))
    return "${" + cid + ".output}", calls


def _match_directive(value: str) -> Optional[tuple]:
    """`value` -> `(flow_id, remaining_args_str)` if it is a whole-value `call(...)`
    directive, or None when its trimmed text does not start with `call(`.

    Returns `remaining_args_str` = the raw args substring AFTER the flow id's comma
    (empty when `call(f)` has no kwargs); `desugar_call_directives` checks those args.
    Raises `ExpressionError` when the value starts with `call(` but its parens are
    unbalanced, text follows the close paren, or the first arg is not a bare flow id."""
    s = value.strip()
    m = CALL_HEAD.match(s)
    if m is None:
        return None
    open_idx = m.end() - 1
    close_idx = _find_paren_end(s, open_idx + 1)
    if close_idx is None:
        raise ExpressionError(
            "`call(` is never closed (check its parentheses and quotes)"
        )
    trailing = s[close_idx + 1 :].strip()
    if trailing:
        raise ExpressionError(
            f"text follows the closing `)`: {trailing!r}; a call(...) must be the "
            "whole value"
        )
    args_str = s[open_idx + 1 : close_idx]
    first = _split_calls_aware(args_str, ",")[0].strip()
    if not _FLOW_ID.fullmatch(first):
        raise ExpressionError(
            "a call(...) must start with a flow id (a bare name) and a comma, "
            f"got {first!r}"
        )
    # everything after the first comma is the remaining (keyword) args; the loop over
    # this substring never re-sees the positional flow id, so a positional arg among
    # the REMAINING args (e.g. `call(f, 30)`) fails loudly there.
    comma = args_str.find(",")
    remaining = args_str[comma + 1 :] if comma != -1 else ""
    return first, remaining


def _desugar_outputs(
    outputs: Any,
    mint: Callable[[], str],
    *,
    range: Optional[SourceRange] = None,
    style_at: Optional[Callable[[Any], Optional[str]]] = None,
    diagnostics: LoadDiagnostics,
) -> tuple:
    """Desugar the flow `output:` section, returning `(new_outputs, inline_calls)`.
    ``range`` is the `output:` section source range; ``style_at(key)`` gives an output's
    YAML scalar style (`key` is None for a single output; plain when absent)."""
    calls: list = []

    def desugar(value: str, key: Any) -> str:
        return _desugar_value(
            value,
            mint,
            calls,
            "flow outputs",
            range=range,
            style=style_at(key) if style_at is not None else None,
            diagnostics=diagnostics,
        )

    if isinstance(outputs, dict):
        new_outputs = {
            k: (desugar(v, k) if isinstance(v, str) else v) for k, v in outputs.items()
        }
    elif isinstance(outputs, str):
        new_outputs = desugar(outputs, None)
    else:
        new_outputs = outputs
    return new_outputs, calls


def _desugar_asserts(
    asserts: Optional[list],
    mint: Callable[[], str],
    *,
    assertion_range_at: Optional[Callable[[int], Optional[SourceRange]]] = None,
    assertion_origin_at: Optional[Callable[[int], OriginRef | None]] = None,
    diagnostics: LoadDiagnostics,
) -> tuple:
    """Cross raw flow asserts into semantic carriers and desugar inline calls.

    Parsed YAML supplies strings. An existing carrier is accepted for source-neutral
    programmatic composition and retains its display/origin. ``assertion_origin_at``
    supplies an optional source-neutral origin by authored list index; an existing origin
    wins. ``assertion_range_at`` supplies a source range by authored list index, keeping
    duplicate assertion text distinct. Rewriting changes only the executable expression,
    so generated node ids never replace authored presentation. `None`/empty returns unchanged.
    """
    if not asserts:
        return asserts, [], {}
    calls: list = []
    call_ranges: dict[str, Optional[SourceRange]] = {}
    new_asserts: list[Assertion] = []
    for index, value in enumerate(asserts):
        assertion_range = (
            assertion_range_at(index) if assertion_range_at is not None else None
        )
        if isinstance(value, Assertion):
            assertion = value
            if assertion.origin is None and assertion_origin_at is not None:
                assertion = replace(
                    assertion,
                    origin=assertion_origin_at(index),
                )
        elif isinstance(value, str):
            assertion = Assertion(
                expression=value,
                display=value,
                origin=(
                    assertion_origin_at(index)
                    if assertion_origin_at is not None
                    else None
                ),
            )
        else:
            raise TypeError(
                "flow assertions must be strings or Assertion carriers, "
                f"got {type(value).__name__}"
            )
        before = len(calls)
        expression = _desugar_value(
            assertion.expression,
            mint,
            calls,
            "assert",
            range=assertion_range,
            quotable=False,
            diagnostics=diagnostics,
        )
        for call in calls[before:]:
            call_ranges[call.id] = assertion_range
        new_asserts.append(replace(assertion, expression=expression))
    return new_asserts, calls, call_ranges


def _to_call_descriptor(
    call: InlineCall,
    *,
    host: str,
    range: Optional[SourceRange] = None,
    diagnostics: LoadDiagnostics | None = None,
) -> CallDescriptor:
    """One `InlineCall` -> a synth `CallDescriptor` (`over=None`), after its guards.

    Authored calls into the compiler-private inline-flow namespace are rejected at this
    shared boundary, before any synth descriptor can reach child resolution. The synth node
    carries its OWN provenance: `range` is the site it desugared from (a node input's binding
    line, a case `then:`/`else:` line, or a section item). It owns no authored fields, so
    `field_ranges` is empty and a bad ref inside the directive carets `range`, never a host
    line."""
    diagnostics = diagnostics or LoadDiagnostics()
    if call.callee.startswith(INLINE_FLOW_PREFIX):
        raise diagnostics.error(
            f"{host}: call targets beginning with {INLINE_FLOW_PREFIX!r} are reserved "
            "for compiler-generated inline child flows",
            range=range,
        )
    for name, value in call.args.items():
        if _captures_item(value):
            raise diagnostics.error(
                f"{host}: inline call {call.callee!r} cannot capture ${{item}} "
                f"(arg {name!r}) — it desugars to a top-level call node with no "
                f"map-element scope; use a named `call` node (whose per-element "
                f"`input:` may read `${{item}}`)",
                range=range,
            )
    return CallDescriptor(
        id=call.id,
        call=call.callee,
        inputs=dict(call.args),
        over=None,
        range=range,
        field_ranges={},
    )


def _captures_item(value: Any) -> bool:
    """True if a binding-string arg value reads `${item}` (a ref with head `item`)."""
    if not isinstance(value, str):
        return False
    try:
        refs = expr_refs_of(value)
    except ExpressionError:
        return (
            False  # malformed -> surfaced (located) by the downstream ref-wiring pass
        )
    return any(ref.split(".")[0] == "item" for ref in refs)
