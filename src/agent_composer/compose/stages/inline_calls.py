"""Graph-stage: whole-value `call(<flow-id>, kw=...)` directive desugar.

`desugar_call_directives` (below) is the pure string→data half: a field whose WHOLE
trimmed value is exactly `call( … )` rewrites to `${<synth>.output}` and emits a
`CallDescriptor`. This module is the assembly pass that runs it over a flow's parsed
descriptors + `outputs:` section and installs each synth `call` node (`over=None`).
After it runs, the rest of `_assemble` (build → edge-infer → validate → DAG) sees the
synth nodes as ordinary `call` nodes — inline calls are pure load-time sugar, no new
runtime kind, the synth callee resolves through the SAME composite resolver
(defs-first) as a named call.

The retired inline `${ f(args) }` form — a FLOW call embedded inside a `${...}` span —
is a hard-break `LoadError` pointing the author at the `call(...)` directive; a pure
builtin (`upper`, `join`, …) inside `${...}` stays legal.

Binding positions + the flow `asserts:` section: a node's `inputs:` (agent/code/model/tool/call),
a mapped call's `over:`, the flow `outputs:` bindings, AND each flow
`asserts:` expression (a whole-value `call(...)` lifts to a synth node so the assert reads
its output). NOT `when:`/`on:` (the case condition grammar) and NOT a node-local
`asserts:` (those stay node-local — no inline call). A `case` descriptor carries no walked
bindings, so it is untouched.

Two guards (loud): an inline call cannot capture `${item}` (the synth node is a
top-level node with no map-element scope — use a named `call` node with `over:`), and
a user node id may not use the reserved synth prefix `__call_`.

Located parse metadata: each parsed host descriptor carries its own `range`/`field_ranges`
(from the span tree). A synth `call` node inherits the HOST descriptor's ranges, so its
downstream errors locate at the host binding site — no `node_lines` map. The flow
`outputs:`/`asserts:` sections have no descriptor, so their inline calls locate at the
section source line (`outputs_line`/`asserts_line`) until the span cutover retires those.

Imports flow DOWN/peer only: `compose.parse.file` (descriptors), `compose.nodes.bindings` (the
kind-blind binding-field walk) + `compose.errors` (peers), `expr` (the ref-walk + the
span flow-call detector), `events` (SourceRange). Only the loader imports this back.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Callable, Optional

from agent_composer.events import SourceRange
from agent_composer.expr import (
    ExpressionError,
    expr_refs_of,
)
from agent_composer.expr.template import (
    InlineCall,
    _arg_source,
    _find_paren_end,
    _split_calls_aware,
    _split_kv,
    flow_call_callees_in_spans,
)
from agent_composer.compose.nodes.bindings import map_binding_strings
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.parse.file import (
    CallDescriptor,
    NodeDescriptor,
)

# The reserved id prefix for synthesized inline-call nodes. A monotonic counter per
# flow/def keeps them unique; the collision guard reserves the namespace from users.
_SYNTH_PREFIX = "__call_"


def _synth_id_gen() -> Callable[[], str]:
    """A fresh `__call_0`, `__call_1`, … minter (one per flow/def `_assemble` run)."""
    counter = itertools.count()
    return lambda: f"{_SYNTH_PREFIX}{next(counter)}"


def desugar_inline_calls(
    descriptors: dict[str, NodeDescriptor],
    outputs_section: Any,
    *,
    asserts_section: Optional[list] = None,
    outputs_line: Optional[int] = None,
    asserts_line: Optional[int] = None,
    next_id: Optional[Callable[[], str]] = None,
) -> tuple:
    """Desugar every inline call in a flow's binding sites into synth `call` nodes.

    Returns `(new_descriptors, new_outputs, new_asserts)`: the descriptor map augmented with
    one synth `CallDescriptor` per inline call (the host binding rewritten to
    `${<synth>.output}`), the flow `outputs:` section, and the `asserts:` list — each with its
    inline calls rewritten the same way. A span-wrapped inline `${ f(...) }` in an assert
    expression (`${ f(...) } == x`) lifts to a synth node so the assert reads its output.

    Each synth node from a node host inherits that host descriptor's `range`/`field_ranges`;
    a synth node from the flow `outputs:`/`asserts:` section locates at `outputs_line`/
    `asserts_line`. Raises `LoadError` on a malformed inline call, a `${item}` capture, or a
    reserved-prefix collision — located at the host node / `outputs:` / `asserts:`."""
    mint = next_id or _synth_id_gen()

    # Reserve the synth namespace: a user node id must not use the prefix (else it
    # could collide with / shadow a generated id). Checked unconditionally.
    for nid, desc in descriptors.items():
        if nid.startswith(_SYNTH_PREFIX):
            raise LoadError(
                f"node id {nid!r} uses the reserved inline-call prefix {_SYNTH_PREFIX!r}",
                range=getattr(desc, "range", None),
            )

    synth: dict[str, CallDescriptor] = {}
    new_descriptors: dict[str, NodeDescriptor] = {}
    for nid, desc in descriptors.items():
        rng = getattr(desc, "range", None)
        field_ranges = getattr(desc, "field_ranges", None)
        new_desc, calls = _desugar_descriptor(desc, mint, host=nid, range=rng)
        new_descriptors[nid] = new_desc
        for call in calls:
            synth[call.id] = _to_call_descriptor(
                call, host=f"node {nid!r}", range=rng, field_ranges=field_ranges
            )

    new_outputs, out_calls = _desugar_outputs(outputs_section, mint, outputs_line)
    for call in out_calls:
        synth[call.id] = _to_call_descriptor(call, host="flow outputs", line=outputs_line)

    new_asserts, assert_calls = _desugar_asserts(asserts_section, mint, asserts_line)
    for call in assert_calls:
        synth[call.id] = _to_call_descriptor(call, host="assert", line=asserts_line)

    new_descriptors.update(synth)
    return new_descriptors, new_outputs, new_asserts


# --------------------------------------------------------------------------- #
# The flow `outputs:` binding walk (used by this pass AND the case-value
# expansion in `compose.cases`). Descriptor binding fields go through the
# kind-blind `compose.nodes.bindings.map_binding_strings` instead.
# --------------------------------------------------------------------------- #


def map_outputs_strings(outputs: Any, fn: Callable[[str], Any]) -> Any:
    """Apply `fn` to each STRING binding in the flow `outputs:` section (a name→binding
    map, a bare binding string, or `None`)."""
    if isinstance(outputs, dict):
        return {k: (fn(v) if isinstance(v, str) else v) for k, v in outputs.items()}
    if isinstance(outputs, str):
        return fn(outputs)
    return outputs


def _desugar_descriptor(
    desc: NodeDescriptor, mint: Callable[[], str], *, host: str, range: Optional[SourceRange]
) -> tuple:
    """Desugar a descriptor's binding fields, returning `(new_desc, inline_calls)`.

    Walks the binding fields via `map_binding_strings`; the closure `fn` desugars each
    string and accumulates the inline calls it finds."""
    calls: list = []
    new_desc = map_binding_strings(
        desc, lambda value: _desugar_value(value, mint, calls, host, range=range)
    )
    return new_desc, calls


def _desugar_value(
    value: str,
    mint: Callable[[], str],
    calls: list,
    host: str,
    *,
    range: Optional[SourceRange] = None,
    line: Optional[int] = None,
) -> str:
    """Desugar one binding string. A WHOLE-VALUE `call(<flow-id>, kw=...)` directive lifts
    to a synth call node (accumulated in `calls`, the value rewritten to `${<synth>.output}`).
    A flow call embedded INSIDE a `${...}` span is the retired `${flow(args)}` form — a
    located `LoadError` pointing at the `call(...)` directive. Any other value (plain refs,
    pure builtins, literals) is returned unchanged. Maps `ExpressionError` (a malformed
    directive or an unparseable span) to a located `LoadError` — at `range` (a node host) or
    `line` (a section host)."""
    try:
        new_value, found = desugar_call_directives(value, mint)
        if found:  # a whole-value `call(...)` directive
            calls.extend(found)
            return new_value
        flow_callees = flow_call_callees_in_spans(value)
    except ExpressionError as exc:
        raise LoadError(f"{host}: {exc}", range=range, line=line) from exc
    if flow_callees:
        callee = flow_callees[0]
        raise LoadError(
            f"{host}: a flow call {callee!r} inside ${{...}} is not supported — write it "
            f"as the whole value with the call(...) directive: call({callee}, ...)",
            range=range,
            line=line,
        )
    return value


def desugar_call_directives(value: str, mint: Callable[[], str]) -> tuple:
    """Recognize a WHOLE-VALUE `call(<flow-id>, kw=...)` directive, else no-op.

    A directive is recognized IFF the whole trimmed `value` is exactly `call( … )`:
    a leading `call`, balanced parens, and nothing non-whitespace after the close
    paren. Its FIRST arg is the POSITIONAL flow id (a bare identifier, may be
    hyphenated — flow ids allow `-`); every remaining arg is a `name=value` keyword.
    Each keyword value passes through `_arg_source` (so `window=30` -> int 30, a
    quoted scalar unwraps, a `${…}`-bearing value stays a binding string). An arg
    value may ITSELF be a whole-value `call(...)` — desugared inner-first.

    Returns `(new_value, calls)`: if `value` is a directive, `new_value` is the synth
    `${<id>.output}` ref and `calls` lists the `InlineCall`s (each nested call before
    its enclosing one, inner-first). If `value` is NOT a whole-value directive, returns
    `(value, [])` unchanged (the caller then checks the value for a retired span
    flow-call). Raises `ExpressionError` on a positional arg after the flow
    id (message mentions "keyword") or a malformed `name=value` keyword."""
    match = _match_directive(value)
    if match is None:
        return value, []
    flow_id, args_str = match
    calls: list = []
    args: dict = {}
    for pair in _split_calls_aware(args_str, ","):
        if not pair.strip():
            continue  # a trailing comma / an empty kwarg slot
        name, arg_value = _split_kv(pair)  # raises if no top-level `=` (positional -> loud)
        new_value, inner = desugar_call_directives(arg_value, mint)  # inner-first recursion
        calls.extend(inner)
        args[name] = _arg_source(new_value)
    cid = mint()
    calls.append(InlineCall(id=cid, callee=flow_id, args=args))
    return "${" + cid + ".output}", calls


def _match_directive(value: str) -> Optional[tuple]:
    """`value` -> `(flow_id, remaining_args_str)` if it is a whole-value `call(...)`
    directive, else None. Whole-value means: trimmed, a leading `call`, balanced
    parens (via `_find_paren_end`), and nothing after the close paren. The flow id is
    the first top-level arg (a bare identifier, may be hyphenated). A positional first
    arg is REQUIRED to be a bare id; a non-directive shape (mid-value, trailing
    content, unbalanced parens, or a non-`call` head) returns None.

    Returns `remaining_args_str` = the raw args substring AFTER the flow id's comma
    (empty when `call(f)` has no kwargs). A positional arg after the flow id is left
    for `_split_kv` to reject (so its "keyword" message surfaces)."""
    s = value.strip()
    m = re.match(r"call\s*\(", s)
    if m is None:
        return None
    open_idx = m.end() - 1
    close_idx = _find_paren_end(s, open_idx + 1)
    if close_idx is None:
        return None  # unbalanced parens — not a directive
    if s[close_idx + 1 :].strip():
        return None  # trailing content after `)` — not whole-value
    args_str = s[open_idx + 1 : close_idx]
    parts = _split_calls_aware(args_str, ",")
    first = parts[0].strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", first):
        return None  # the first arg is not a bare flow id — not a directive
    # everything after the first comma is the remaining (keyword) args; the loop over
    # this substring never re-sees the positional flow id, so a positional arg among
    # the REMAINING args (e.g. `call(f, 30)`) reaches `_split_kv` and fails loudly.
    comma = args_str.find(",")
    remaining = args_str[comma + 1 :] if comma != -1 else ""
    return first, remaining


def _desugar_outputs(outputs: Any, mint: Callable[[], str], line: Optional[int]) -> tuple:
    """Desugar the flow `outputs:` section, returning `(new_outputs, inline_calls)`.
    `line` is the `outputs:` section source line, so a malformed inline call there
    locates like a node host."""
    calls: list = []
    new_outputs = map_outputs_strings(
        outputs, lambda value: _desugar_value(value, mint, calls, "flow outputs", line=line)
    )
    return new_outputs, calls


def _desugar_asserts(asserts: Optional[list], mint: Callable[[], str], line: Optional[int]) -> tuple:
    """Desugar an inline `${ f(...) }` in each flow-assert string, returning
    `(new_asserts, inline_calls)`. A non-string assert (shouldn't occur — the schema is
    `list[str]`) passes through. `None`/empty returns unchanged."""
    if not asserts:
        return asserts, []
    calls: list = []
    new_asserts = [
        _desugar_value(a, mint, calls, "assert", line=line) if isinstance(a, str) else a
        for a in asserts
    ]
    return new_asserts, calls


def _to_call_descriptor(
    call: InlineCall,
    *,
    host: str,
    range: Optional[SourceRange] = None,
    field_ranges: Optional[dict] = None,
    line: Optional[int] = None,
) -> CallDescriptor:
    """One `InlineCall` -> a synth `CallDescriptor` (`over=None`), after the `${item}`
    guard. A node-host synth node inherits the host's `range`/`field_ranges` (so downstream
    errors locate at the host); a section host (`outputs:`/`asserts:`) locates at `line`."""
    for name, value in call.args.items():
        if _captures_item(value):
            raise LoadError(
                f"{host}: inline call {call.callee!r} cannot capture ${{item}} "
                f"(arg {name!r}) — it desugars to a top-level call node with no "
                f"map-element scope; use a named `call` node (whose per-element "
                f"`inputs:` may read `${{item}}`)",
                range=range,
                line=line,
            )
    return CallDescriptor(
        id=call.id,
        call=call.callee,
        inputs=dict(call.args),
        over=None,
        range=range,
        field_ranges=dict(field_ranges) if field_ranges else {},
    )


def _captures_item(value: Any) -> bool:
    """True if a binding-string arg value reads `${item}` (a ref with head `item`)."""
    if not isinstance(value, str):
        return False
    try:
        refs = expr_refs_of(value)
    except ExpressionError:
        return False  # malformed -> surfaced (located) by the downstream ref-wiring pass
    return any(ref.split(".")[0] == "item" for ref in refs)
