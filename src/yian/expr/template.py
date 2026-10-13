"""The `${...}` template language — binding values AND strict prompt rendering.

A binding value (a node `input:` entry or a flow `output:` entry) is a TEMPLATE:
plain text interspersed with `${...}` spans, with `$$` for a literal `$`.

- A value that is EXACTLY one `${...}` resolves to the **typed** value of that
  reference (a float stays a float, a list a list, an object a dict).
- A `${...}` **embedded** in surrounding text is **stringified** into it.
- A value with no `${...}` is a plain literal (after `$$` -> `$`).

The interior of a `${...}` uses the unified expression grammar: references such as
`node.output.field` and `input.name`, literals, arithmetic/boolean operators, pure
builtins, coalesce (`|`), default (`:-`), and required (`:?`).

Nesting is ONE level: `${x:-${y}}` is allowed; `${x:-${y:-${z}}}` is an error
(use `|` for multi-way chains: `${x | y | z}`).

Pure parse/eval (no pool): `eval_binding` takes a `resolve` callable so this stays
a leaf both `nodes` (runtime bind) and `compile` (the compile-time reference walk)
may import.

This module also owns the strict AGENT/HUMAN_INPUT prompt renderer
(`render_template_record`) and its compile-time companion (`prompt_refs`): a prompt is
NOT a binding (it reads already-bound declared inputs, mints no edge), but it is a
`${...}` template, and it reuses these scanners — so it lives here rather than in
`expressions` (which cannot import this module without a cycle).

Knows about: `expr.expressions` (peer — `_parse_literal`, `ExpressionError`) and
`expr.builtins` (the prompt `TEMPLATE_FNS` registry).
Never imports: `nodes`, `compile`, `runtime`, or `typesys` at runtime (pool-agnostic via
`resolve`; `Type` appears in annotations only).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from yian.expr import grammar
from yian.expr.builtins import TEMPLATE_FNS
from yian.expr.expressions import (
    ISINSTANCE,
    ExpressionError,
    ResolveMode,
    _parse_literal,
    eval_expr,
    expr_refs,
    rewrite_expr_refs,
)
from yian.expr.lineage import TracedValue, ValueRoute

if TYPE_CHECKING:  # annotation-only imports
    from lark import Token, Tree

    from yian.typesys.values import Type


# --------------------------------------------------------------------------- #
# Shared call-parsing helpers — reused by the whole-value `call(...)` directive
# (`compose.calls`) and the prompt-span parser further down.
#
# `InlineCall` is the plain-data record a desugared call produces (the directive
# path builds one per call); `_split_kv` / `_arg_source` / `_default_literal` /
# `_find_paren_end` / `_split_calls_aware` are the paren/quote/span-aware splitters +
# arg-literal coercion the directive recognizer reuses. Keyword args only; each arg
# VALUE is a full binding — a `${…}`-bearing value (or a quoted scalar) stays a string,
# else it is a literal (so `window=30` binds int 30). The literal grammar is a
# deliberate YAML-1.1 SUBSET — see `_arg_source`.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InlineCall:
    """One desugared inline call: the synth node `id`, the `callee` name, and the
    keyword `args` (name -> a YAML-scalar literal value or a `${…}` binding string)."""

    id: str
    callee: str
    args: dict  # name -> a literal value or a `${…}` binding string (see `_arg_source`)


def _default_literal(token: str) -> Any:
    """A bare arg token -> its typed literal — a word that isn't a number/bool/null
    spelling is a literal string (no quotes)."""
    try:
        return _parse_literal(token)
    except ExpressionError:
        return token


def _split_kv(pair: str) -> tuple:
    """Split one arg `name=value` on its FIRST top-level `=` (keyword args only)."""
    parts = _split_calls_aware(pair, "=")
    if len(parts) < 2:
        raise ExpressionError(
            f"inline call arg {pair.strip()!r} must be keyword (name=value)"
        )
    return parts[0].strip(), "=".join(parts[1:])


def _arg_source(value: str) -> Any:
    """One inline-call arg value -> its source, mirroring the named form's `input:`:

    - a fully-quoted scalar (`"…"`/`'…'`) -> its inner text (quotes stripped, like a
      YAML quoted scalar): a binding template if it interpolates (`"hi ${name}"` ->
      `hi ${name}`), else a plain string;
    - an unquoted `${…}`-bearing value -> the binding string (ref / coalesce / embedded);
    - else a bare token -> a literal via `_default_literal` (number / bool / null / a
      bare string), so `window=30` binds int 30.

    NB this literal grammar is a deliberate SUBSET of YAML 1.1: a bare `yes`/`no`/`on`/
    `off`/`None` stays a string (no boolean/null coercion) — avoiding the YAML bool
    footgun. Quote, number, true/false, and null spellings match the named form."""
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] in ("'", '"') and stripped[-1] == stripped[0]:
        return stripped[1:-1]  # a quoted scalar -> its inner text (template or literal)
    if "${" in stripped:
        return stripped
    return _default_literal(stripped)


def _find_span_end(s: str, start: int) -> Optional[int]:
    """Index of the `}` closing a `${…}` span whose interior begins at `start`
    (brace-depth + quote aware), or None if unbalanced."""
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


def _find_paren_end(s: str, start: int) -> Optional[int]:
    """Index of the `)` closing a call whose args begin at `start` — paren-depth
    aware, skipping `${…}` spans (their inner parens) and quoted text; None if
    unbalanced."""
    depth, quote, i, n = 1, None, start, len(s)
    while i < n:
        c = s[i]
        if quote is not None:
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
        elif c == "$" and s[i + 1 : i + 2] == "{":
            end = _find_span_end(s, i + 2)
            if end is None:
                return None
            i = end + 1
            continue
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _split_calls_aware(s: str, sep: str) -> list:
    """Split `s` on the single char `sep` at top level — ignoring `sep` inside quotes,
    `${…}` spans (brace depth), and parentheses (call args). Paren-aware so a call's
    top-level bare parens do not swallow the separator."""
    out: list = []
    buf: list = []
    quote: Optional[str] = None
    brace = 0
    paren = 0
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if quote is not None:
            buf.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            buf.append(c)
        elif c == "$" and s[i + 1 : i + 2] == "{":
            brace += 1
            buf.append("${")
            i += 2
            continue
        elif c == "{" and brace > 0:
            brace += 1
            buf.append(c)
        elif c == "}" and brace > 0:
            brace -= 1
            buf.append(c)
        elif brace > 0:
            buf.append(c)
        elif c == "(":
            paren += 1
            buf.append(c)
        elif c == ")":
            paren = max(0, paren - 1)  # clamp: a stray ')' must not disable top-level splits
            buf.append(c)
        elif c == sep and paren == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    out.append("".join(buf))
    return out


# --------------------------------------------------------------------------- #
# Evaluate (pool-agnostic: `resolve` is a path -> value callable; `item` is the
# MAP-body-local scope)
# --------------------------------------------------------------------------- #


def eval_binding(
    source: str,
    resolve: Callable[[str], Any],
    item: Any = None,
    *,
    trace: bool = False,
    types: dict[str, Type] | None = None,
) -> Any:
    """
    Evaluate a binding-value TEMPLATE string to a value (through the unified engine).

    A binding that is EXACTLY one `${...}` span resolves to the **typed** value of that
    span's expression (a float stays a float, a list a list, an object a dict). A span
    embedded in surrounding text — or a plain-text run — is stringified and concatenated;
    a source with no `${...}` is a plain literal (after `$$` -> `$`).

    This is the raw-string binding API: it scans `source` into template segments
    (`scan_template`, parsing only span interiors with the unified grammar) and evaluates
    them in `BINDING_NONE` mode (a missing reference resolves to `None`, so a coalesce /
    default may fire). The full `${...}` expression grammar is available inside a span —
    arithmetic (`${x + 1}`), string/list ops (`${xs + [item]}`), coalesce / default /
    required — not just the legacy coalesce-of-atoms.

    Args:
        source (`str`):
            The binding-value template: literal text interspersed with `${...}` spans.
            (A non-string source is a literal and never reaches here — callers short-
            circuit it — so this takes a `str`.)
        resolve (`Callable[[str], Any]`):
            Resolves a reference path to its value (pool-agnostic seam); a path it maps
            to `None` is a MISS (resolves to `None` in this binding mode).
        item (`Any`, *optional*, defaults to `None`):
            The MAP-body-local scope for `${item}` / `${item.path}`. `None` means no item
            scope (a `None` element and "no scope" coincide).
        trace (`bool`, *optional*, defaults to `False`):
            Internal retention switch. The resolver returns `ResolvedValue` and
            `item` may supply the same carrier; return a `TracedValue` containing
            the ordinary result and the route collected by this evaluation.
        types (`dict[str, Type] | None`, *optional*, defaults to `None`):
            The type table `isinstance(x, T)` looks `T` up in. `None` means no types.

    Returns:
        `Any`:
            The typed value for a whole single span; otherwise the rendered
            string. With tracing enabled, the same value inside `TracedValue`.

    Raises:
        `RequiredError`:
            If a `${ref:?message}` required atom is unbound.
        `ExpressionError`:
            On a malformed span (unbalanced `${...}`, unparseable interior) or any error
            the span's evaluation raises.
        `TypeError`:
            When a traced resolver does not return `ResolvedValue`.
    """
    return eval_template(
        scan_template(source), resolve, item=item, mode=ResolveMode.BINDING_NONE,
        trace=trace, types=types,
    )


def expr_refs_of(source: str) -> list[str]:
    """
    Collect every reference PATH a binding-value TEMPLATE string reads.

    The raw-string ref-walk: scan `source` into template segments and union
    (source order, NOT deduped) the [`expr_refs`][yian.expr.expressions.expr_refs]
    of each `${...}` span. This is the compile-time reference walk at every call site
    (edge inference, item-capture, on-shape resolution, output typing).

    Literal runs contribute nothing. `item`-headed refs ARE collected (the caller skips
    them for edge minting — the rule stays at the caller, per `expr_refs`).

    Args:
        source (`str`):
            The binding-value template: literal text interspersed with `${...}` spans.

    Returns:
        `list[str]`:
            The reference paths across all spans, in source order (NOT deduped).

    Raises:
        `ExpressionError`:
            On a malformed span (unbalanced `${...}`, unparseable interior) — callers
            that already frame parse errors keep their `except ExpressionError` handling.
    """
    return template_refs(scan_template(source))


def rewrite_template_refs(source: str, rename: Callable[[str], "str | None"]) -> str:
    """Rewrite the reference leaves across ALL `${...}` spans of a binding/template value.

    The rewrite analog of [`expr_refs_of`][yian.expr.template.expr_refs_of]: scan
    `source` for its `${...}` spans (brace/quote aware, `$$` escapes preserved verbatim) and
    rewrite each span's interior via
    [`rewrite_expr_refs`][yian.expr.expressions.rewrite_expr_refs] — so a whole-span
    `${x.output}`, an embedded `"pre ${a} post"`, and a nested `${x :- ${y}}` are all handled
    uniformly. `rename(path)` returns the replacement path (or `None` to keep). Literal text
    between spans is preserved exactly (unlike `scan_template`, which collapses `$$` -> `$`,
    this rewriter re-emits `$$` and every literal char verbatim, since it produces SOURCE).

    A value with no `${` (or a non-string) is returned unchanged. An unbalanced `${` leaves
    the trailing text as-is (the downstream binding parser reports it, located)."""
    if not isinstance(source, str) or "${" not in source:
        return source
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c == "$" and source[i + 1 : i + 2] == "$":
            out.append("$$")  # a literal `$` — never a span start; kept verbatim
            i += 2
        elif c == "$" and source[i + 1 : i + 2] == "{":
            end = _find_span_end(source, i + 2)
            if end is None:  # unbalanced ${ — leave the rest for the binding parser
                out.append(source[i:])
                break
            out.append("${" + rewrite_expr_refs(source[i + 2 : end], rename) + "}")
            i = end + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def binding_co_skips(source: Any) -> bool:
    """True if this binding co-skips when all its referenced producers are skipped.

    A hard data dependency: a whole-string `${...}` span whose expression is a pure group
    of references / ref-defaults with NO literal escape and NO `:?` required. A literal
    operand runs the node with the literal; a `:-literal` default supplies one; a `:?` runs
    it to fail loud; embedded text stringifies an absent ref to ''. None of those
    co-skip; non-strings never co-skip.

    Re-implemented over the RAW string via the unified engine: scan `source` into template
    segments; it co-skips iff there is exactly ONE segment and it is a `${...}` span whose
    parsed expression is a co-skipping operand (see `_operand_co_skips`). Any surrounding
    literal text (embedded span), a plain literal, a malformed span, or a computed
    expression (arithmetic / comparison / builtin call) -> `False`.
    """
    if not isinstance(source, str):
        return False
    try:
        segments = scan_template(source)
    except ExpressionError:
        return False
    if len(segments) != 1 or not isinstance(segments[0], Span):
        return False  # embedded text / plain literal -> stringifies, never co-skips
    return _operand_co_skips(segments[0].tree)


def _operand_co_skips(node: "Tree | Token") -> bool:
    """Does one parsed-expression operand co-skip (a pure ref / ref-default group)?

    True for: a bare reference (`refcall` with no `call_suffix`), a `${...}`-wrapped ref
    (`WRAPPED_REF` token), a `coalesce` whose EVERY operand co-skips, and a `default_expr`
    whose fallback is itself a co-skipping operand (`:-${y}` / `:-b.output`). False for a
    literal token, a builtin-call `refcall`, a `required_expr` (`:?`), and any computed node
    (arithmetic / comparison / boolean / list) — none of which are a pure ref dependency.
    """
    from lark import Token as _Token, Tree as _Tree

    if isinstance(node, _Token):
        return node.type == "WRAPPED_REF"  # a `${ref}` token co-skips; a literal token doesn't
    if not isinstance(node, _Tree):
        return False
    if node.data == "refcall":
        # a reference co-skips; a builtin call (has a `call_suffix`) is a computed value.
        return not any(isinstance(c, _Tree) and c.data == "call_suffix" for c in node.children)
    if node.data == "coalesce":
        return all(_operand_co_skips(c) for c in node.children)
    if node.data == "default_expr":  # `head :- fallback` — fallback must be a ref too
        return _operand_co_skips(node.children[1])
    return False  # required_expr / arithmetic / comparison / list / ... -> not a pure ref


# --------------------------------------------------------------------------- #
# Strict prompt rendering — an AGENT/HUMAN_INPUT prompt against its bound input
# record. A `${...}` span is EITHER a plain dotted reference (`${name.path}`) OR a
# builtin call (`${ fn(${ref}, lit).path }`). Unlike a binding, a prompt is not on the
# graph: it reads inputs already bound to this node, mints no edge, and (the call form)
# evaluates a read-only `TEMPLATE_FNS` formatter at render time — a bounded bend of the
# "all computation is a node" law documented in `docs/expressions.md`. The
# renderer (`render_template_record`) and the compile-time scope check
# (`prompt_refs` -> `compose.validate`) share ONE span parser so they cannot drift.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _PromptRef:
    """A plain `${name.path}` prompt span — a dotted record reference."""

    path: str


@dataclass(frozen=True)
class _PromptArg:
    """One builtin-call argument. `name` is the keyword (None -> positional). Exactly one
    of `ref` (a `${...}`-wrapped dotted record path) / `literal` is meaningful, per `is_ref`."""

    name: Optional[str]
    ref: Optional[str]
    literal: Any
    is_ref: bool


@dataclass(frozen=True)
class _PromptCall:
    """A `${ callee(args).trailing }` prompt span: a `TEMPLATE_FNS` builtin call with
    ordered `args` and optional dotted access `trailing` on the result."""

    callee: str
    args: tuple  # tuple[_PromptArg, ...]
    trailing: tuple  # tuple[str, ...] — dotted steps after the close paren


def _check_prompt_path(path: str) -> None:
    """A prompt reference path must split into non-empty dotted segments (the charset is
    left permissive — same lenient check the regex renderer used)."""
    parts = [p.strip() for p in path.split(".")]
    if not parts or any(not p for p in parts):
        raise ExpressionError(f"malformed reference ${{{path}}} in prompt")


def _parse_prompt_span(interior: str) -> Union[_PromptRef, _PromptCall]:
    """Parse one `${...}` interior into a plain ref or a builtin call. Raises on a
    malformed span (empty / unbalanced parens / bad dotted access / non-ref non-literal
    arg). Does NOT check the builtin exists or resolve refs — that is the caller's job."""
    s = interior.strip()
    if not s:
        raise ExpressionError("empty ${} span in prompt")
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
    if m is None:  # no call syntax -> a plain dotted reference
        _check_prompt_path(s)
        return _PromptRef(s)
    callee = m.group(1)
    open_idx = m.end() - 1
    close_idx = _find_paren_end(s, open_idx + 1)
    if close_idx is None:
        raise ExpressionError(f"unbalanced '(' in prompt call {s!r}")
    args = tuple(
        _parse_prompt_arg(raw)
        for raw in _split_calls_aware(s[open_idx + 1 : close_idx], ",")
        if raw.strip()  # skip `f()` / a trailing comma
    )
    trailing_str = s[close_idx + 1 :].strip()
    trailing: tuple = ()
    if trailing_str:
        if not trailing_str.startswith("."):
            raise ExpressionError(f"unexpected {trailing_str!r} after call in prompt {s!r}")
        steps = [p.strip() for p in trailing_str[1:].split(".")]
        if not steps or any(not p for p in steps):
            raise ExpressionError(f"malformed dotted access {trailing_str!r} in prompt {s!r}")
        trailing = tuple(steps)
    return _PromptCall(callee, args, trailing)


def _parse_prompt_arg(raw: str) -> _PromptArg:
    """One builtin-call arg `[(name=)]value` -> a `_PromptArg`. A top-level `=` with a
    bare-identifier LHS is a keyword; the value is either ONE whole `${ref}` (a record
    reference) or a literal (number/bool/null/quoted) — a bare unwrapped word is rejected
    (record refs MUST be `${...}`-wrapped)."""
    name: Optional[str] = None
    value = raw
    eq_parts = _split_calls_aware(raw, "=")
    if len(eq_parts) >= 2 and re.fullmatch(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*", eq_parts[0]):
        name = eq_parts[0].strip()
        value = "=".join(eq_parts[1:])
    v = value.strip()
    if v.startswith("${"):  # a record reference — must be exactly one whole span
        end = _find_span_end(v, 2)
        if end == len(v) - 1:
            ref = v[2:end].strip()
            _check_prompt_path(ref)
            return _PromptArg(name=name, ref=ref, literal=None, is_ref=True)
    try:
        lit = _parse_literal(v)
    except ExpressionError:
        raise ExpressionError(
            f"prompt call arg {v!r} must be a `${{ref}}` or a literal "
            f"(number/bool/null/quoted string)"
        ) from None
    return _PromptArg(name=name, ref=None, literal=lit, is_ref=False)


def _resolve_record_strict(path: str, record: dict) -> Any:
    """Resolve a dotted `path` against a node's bound input `record`, raising on an
    unknown head, a dotted-walk miss, or a None value (the strict prompt floor)."""
    parts = [p.strip() for p in path.split(".")]
    if not parts or any(not p for p in parts):
        raise ExpressionError(f"malformed reference ${{{path}}} in prompt")
    if parts[0] not in record:
        raise ExpressionError(f"unresolved input reference ${{{path}}} in prompt")
    value: Any = record[parts[0]]
    for step in parts[1:]:
        value = value.get(step) if isinstance(value, dict) else None
        if value is None:
            raise ExpressionError(f"unresolved input reference ${{{path}}} in prompt")
    if value is None:
        raise ExpressionError(f"unresolved input reference ${{{path}}} in prompt")
    return value


def _eval_prompt_span(interior: str, record: dict) -> Any:
    """Evaluate one `${...}` prompt span to its typed value (the caller stringifies)."""
    node = _parse_prompt_span(interior)
    if isinstance(node, _PromptRef):
        return _resolve_record_strict(node.path, record)
    fn = TEMPLATE_FNS.get(node.callee)
    if fn is None:
        raise ExpressionError(f"unknown prompt function {node.callee!r}")
    pos: list = []
    kw: dict = {}
    for a in node.args:
        val = _resolve_record_strict(a.ref, record) if a.is_ref else a.literal
        if a.name is None:
            pos.append(val)
        else:
            kw[a.name] = val
    try:
        value = fn(*pos, **kw)
    except ExpressionError:
        raise
    except Exception as exc:  # a builtin blew up (bad arity, wrong type) — surface loudly
        raise ExpressionError(f"prompt function {node.callee!r} failed: {exc}") from exc
    for step in node.trailing:
        value = value.get(step) if isinstance(value, dict) else None
        if value is None:
            raise ExpressionError(
                f"unresolved dotted access .{step} on {node.callee!r}() result in prompt"
            )
    return value


def _resolve_record_strict_or_none(path: str, record: dict) -> Any:
    """Resolve a dotted `path` against a node's bound input `record`, returning `None` on
    any miss (unknown head, a dotted-walk step off a non-dict / absent key, or a `None`
    value). This is the resolve seam handed to `eval_template` in `STRICT_RAISE` mode: the
    evaluator turns a `None` here into a raised `ExpressionError`, so a `None` return IS the
    strict prompt floor (no silent blank). A present-but-falsy value (`0`, `""`, `False`)
    is a hit and comes back unchanged."""
    parts = [p.strip() for p in path.split(".")]
    if not parts or any(not p for p in parts):
        raise ExpressionError(f"malformed reference ${{{path}}} in prompt")
    if parts[0] not in record:
        return None
    value: Any = record[parts[0]]
    for step in parts[1:]:
        value = value.get(step) if isinstance(value, dict) else None
        if value is None:
            return None
    return value


def render_template_record(text: str, record: dict) -> str:
    """
    Render a strict AGENT / HUMAN_INPUT prompt against its bound input `record`.

    Each `${...}` span is a unified expression (`expr.grammar`): a plain dotted reference
    (`${name}` / `${name.path}`), a builtin call (`${ render_as_json(${name}, 4) }`,
    optionally `.field` on the result), or arithmetic / string / list ops over them
    (`${a + 1}`). A reference resolves against `record` (a node's declared inputs); a call
    invokes the named `expr.builtins.TEMPLATE_FNS` formatter over its resolved args. Bare
    local-input refs only — the pool namespaces (`node`/`system`) are not in scope.

    Rendered in `STRICT_RAISE` mode: unlike `eval_binding` (missing -> None) and the strict
    CASE `when:` (missing -> falsy), this renderer raises on a missing reference or a span
    whose final value is `None`, before whole-value return or embedded stringification. A
    prompt is text, so the result is always a `str`: a whole-single-span value is
    stringified, embedded spans are stringified in place. `$$` renders a single `$` (the
    unified-scanner universal escape).

    Args:
        text (`str`):
            The prompt template: literal text interspersed with `${...}` spans.
        record (`dict`):
            The node's bound input record; reference heads must be declared keys.

    Returns:
        `str`:
            The fully rendered prompt with every span substituted by its string value.

    Raises:
        `ExpressionError`:
            On an unbalanced span, an unresolved reference (unknown input or dict-path
            miss), a final `None` span value, an unknown builtin, or a builtin that fails.
    """
    segments = scan_template(text)
    result = eval_template(
        segments,
        lambda path: _resolve_record_strict_or_none(path, record),
        mode=ResolveMode.STRICT_RAISE,
    )
    return str(result)


def prompt_refs(text: str) -> list[str]:
    """Every declared-input reference PATH a prompt reads — the compile-time companion to
    `render_template_record`, used by `compose.validate` to name-check prompt scope.

    Scans `text` with the unified scanner and unions `template_refs` over its spans: each
    plain-span path plus each builtin-call argument path (literals contribute nothing), in
    source order (not deduped). Also rejects an unknown builtin callee — the loader relies
    on this compile-time check so a prompt naming a non-existent `TEMPLATE_FNS` formatter
    fails at load, not at render — and `isinstance`, which a prompt has no type table
    for. Raises `ExpressionError` on a malformed / unparseable span too."""
    segments = scan_template(text)
    for seg in segments:
        if isinstance(seg, Span):
            _reject_unknown_prompt_builtin(seg.tree)
    return template_refs(segments)


def _call_suffix_callees_absent_from_builtins(tree: "Tree | Token") -> list[str]:
    """Every callee named by a builtin-CALL inside a parsed `${...}` span `tree` whose
    name is NOT a `TEMPLATE_FNS` builtin or `isinstance` — i.e. a FLOW call embedded in a
    `${...}` expression. A `refcall` carrying a `call_suffix` child is a call; a bare
    reference (no suffix) names no callee and is skipped. Returned in tree-walk order (not
    deduped). An empty list means the span calls only known builtins (or no callee at all)."""
    return [
        callee for callee in _call_callees(tree)
        if callee not in TEMPLATE_FNS and callee != ISINSTANCE
    ]


def _call_callees(tree: "Tree | Token") -> list[str]:
    """Every callee named by a call inside a parsed `${...}` span, in tree-walk order."""
    from lark import Tree as _Tree  # runtime import: `Tree` is TYPE_CHECKING-only above

    if not isinstance(tree, _Tree):
        return []
    return [
        str(node.children[0])
        for node in tree.iter_subtrees()
        if node.data == "refcall"
        and any(isinstance(c, _Tree) and c.data == "call_suffix" for c in node.children)
    ]


def _reject_unknown_prompt_builtin(tree: Tree | Token) -> None:
    """Raise `ExpressionError` if any builtin CALL in a scanned prompt span names a callee
    absent from `TEMPLATE_FNS`. Walks the parse tree for `refcall`s carrying a `call_suffix`
    (a call); a bare reference (no suffix) names no builtin and is skipped. This is the
    compile-time mirror of the render-time `unknown expression function` raise, so the
    loader can reject an unknown prompt builtin without evaluating. `isinstance` is
    rejected too: prompt builtins only format, and a prompt has no type table."""
    if ISINSTANCE in _call_callees(tree):
        raise ExpressionError(
            "isinstance is not a prompt function; compute it in an input binding"
        )
    bad = _call_suffix_callees_absent_from_builtins(tree)
    if bad:
        raise ExpressionError(f"unknown prompt function {bad[0]!r}")


# --------------------------------------------------------------------------- #
# The ONE unified template scanner — splits a text string into literal runs +
# `${...}` spans, parsing ONLY the span interiors with the unified grammar
# (`expr.grammar.parse_expr`) and evaluating them via `expr.expressions.eval_expr`.
#
# This is the template layer OVER the one parser: outside `${...}`, text is LITERAL
# — operator characters (`|`, `+`, `[`, ...) in literal text are NEVER parsed as
# expression operators, which is what makes free text like
# `"stance (positive|negative|neutral)"` safe. Only span interiors go through
# `parse_expr`. `$$` -> a literal `$` everywhere (the universal-escape decision).
#
# The `TypedValue` types (`Literal` / `Span`) carry a `parse_expr` Lark tree per span —
# there is ONE grammar for a span interior, shared with conditions and prompts.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Literal:
    """A literal run of a template — raw text with `$$` already collapsed to `$`.
    Its `text` is emitted verbatim; it is never parsed as an expression."""

    text: str


@dataclass(frozen=True)
class Span:
    """A `${...}` span of a template — its interior parsed via `grammar.parse_expr`.
    `tree` is the resulting `Tree | Token` (a lone atom inlines to a `Token`),
    evaluated by `eval_expr`. ONLY span interiors are parsed."""

    tree: Tree | Token


# A template TypedValue is a literal run OR a parsed `${...}` span.
TypedValue = Union[Literal, Span]


def scan_template(text: str) -> list[TypedValue]:
    """
    Scan a template string into a list of `TypedValue`s — literal runs + `${...}` spans.

    Splits `text` on `${...}` spans, parsing ONLY each span's interior via
    `expr.grammar.parse_expr`. Literal text between spans is kept RAW (with `$$`
    collapsed to `$`) and is NEVER parsed — so operator characters (`|`, `+`, `[`,
    ...) in free text stay literal. `$$` -> `$` everywhere; a lone `$` not starting
    a `${` span stays literal.

    Args:
        text (`str`):
            The template source: literal text interspersed with `${...}` spans.

    Returns:
        `list[TypedValue]`:
            The ordered segments — a [`Literal`][yian.expr.template.Literal]
            per literal run and a [`Span`][yian.expr.template.Span] per
            `${...}` (interior already parsed). Adjacent literal runs are never
            emitted separately: each literal run is one `Literal`. A pure-literal
            template is a single `Literal`.

    Raises:
        `ExpressionError`:
            On an unbalanced `${...}` span, or if a span interior fails to parse.
    """
    segments: list = []
    buf: list = []  # accumulates the current literal run (with `$$` -> `$`)
    i, n = 0, len(text)
    while i < n:
        if text[i] == "$" and text[i + 1 : i + 2] == "$":
            buf.append("$")  # `$$` -> a literal `$`
            i += 2
        elif text[i] == "$" and text[i + 1 : i + 2] == "{":
            if buf:
                segments.append(Literal("".join(buf)))
                buf = []
            end = _find_span_end(text, i + 2)  # reuse the shared brace/quote matcher
            if end is None:
                raise ExpressionError(f"unbalanced '${{' in template {text!r}")
            segments.append(Span(grammar.parse_expr(text[i + 2 : end])))
            i = end + 1
        else:
            buf.append(text[i])
            i += 1
    if buf:
        segments.append(Literal("".join(buf)))
    return segments


def flow_call_callees_in_spans(text: str) -> list[str]:
    """Every FLOW callee (a callee absent from `TEMPLATE_FNS`) named by a call inside a
    `${...}` span of `text`. The loader uses this to REJECT the retired inline
    `${ flow(args) }` form: a flow call belongs in the whole-value `call(...)` directive,
    not inside a `${...}` expression. Pure builtins (`upper`, `join`, …) and the
    `isinstance(x, T)` type test return nothing — they stay legal inside `${...}`.
    Returns [] when no span names a flow callee.

    Raises `ExpressionError` if a span interior does not parse (the same parse the
    downstream template evaluator performs)."""
    out: list[str] = []
    for seg in scan_template(text):
        if isinstance(seg, Span):
            out.extend(_call_suffix_callees_absent_from_builtins(seg.tree))
    return out


def eval_template(
    segments: list,
    resolve: Callable[[str], Any],
    item: Any = None,
    mode: ResolveMode = ResolveMode.BINDING_NONE,
    *,
    trace: bool = False,
    types: dict[str, Type] | None = None,
) -> Any:
    """
    Evaluate scanned template `segments` to a value.

    A template that is EXACTLY one `${...}` span (no surrounding literal text)
    returns the TYPED value of `eval_expr` (a float stays a float, a list a list, a
    dict a dict). Otherwise every span is stringified and concatenated with the
    literal runs. In non-strict modes an embedded `None` becomes `""`; in
    `STRICT_RAISE`, a span whose final value is `None` raises before either return or
    concatenation. A pure-literal template returns its literal string. An empty template
    (`scan_template("") == []`, no segments at all) returns `""`.

    Args:
        segments (`list[TypedValue]`):
            The scan from [`scan_template`][yian.expr.template.scan_template].
        resolve (`Callable[[str], Any]`):
            Resolves a reference path to its value (pool-agnostic seam); a path it
            maps to `None` is a miss.
        item (`Any`, *optional*, defaults to `None`):
            The MAP-body-local scope for `${item}` / `${item.path}`.
        mode (`ResolveMode`, *optional*, defaults to `BINDING_NONE`):
            How a missing reference is treated inside a span (see `eval_expr`).
        trace (`bool`, *optional*, defaults to `False`):
            Internal retention switch. The resolver returns `ResolvedValue` and
            `item` may supply the same carrier. Record each embedded expression's
            route and half-open character span in the rendered output.
        types (`dict[str, Type] | None`, *optional*, defaults to `None`):
            The type table `isinstance(x, T)` looks `T` up in. `None` means no types.

    Returns:
        `Any`:
            The typed value for a whole-single-span template; otherwise the rendered
            string (pure-literal -> the literal string). With tracing enabled,
            return the same value inside `TracedValue`.

    Raises:
        `ExpressionError`:
            On any error `eval_expr` raises for a span interior, or when a span's final
            value is `None` in `STRICT_RAISE` mode.
        `RequiredError`:
            When a span's `head :? message` required atom misses / is None.
        `TypeError`:
            When a traced resolver does not return `ResolvedValue`.
    """
    def eval_span(span: Span) -> Any:
        result = eval_expr(span.tree, resolve, item=item, mode=mode, trace=trace, types=types)
        value = result.value if trace else result
        if value is None and mode is ResolveMode.STRICT_RAISE:
            raise ExpressionError("null value in strict template")
        return result

    if len(segments) == 1 and isinstance(segments[0], Span):
        return eval_span(segments[0])
    parts: list = []
    routes: list[ValueRoute] = []
    spans: list[tuple[int, int]] = []
    expressions: list[str] = []
    offset = 0
    for seg in segments:
        if isinstance(seg, Literal):
            parts.append(seg.text)
            if trace:
                expressions.append(seg.text)
        else:  # a Span embedded in surrounding text -> stringified
            evaluated = eval_span(seg)
            v = evaluated.value if trace else evaluated
            parts.append("" if v is None else str(v))
            if trace:
                routes.append(evaluated.route)
                spans.append((offset, offset + len(parts[-1])))
                expressions.append("${" + evaluated.route.expression + "}")
        offset += len(parts[-1])
    value = "".join(parts)
    if not trace:
        return value
    route = ValueRoute(
        "interpolation", value_json=json.dumps(value, ensure_ascii=False),
        children=tuple(routes), expression="".join(expressions), spans=tuple(spans),
    ) if routes else ValueRoute.literal(value)
    return TracedValue(value, route)


def template_refs(segments: list) -> list[str]:
    """
    Collect every reference PATH a template reads — union (source order) of
    `expr_refs` over the span segments.

    Literal runs contribute nothing. Matches the no-dedupe / source-order convention
    of [`expr_refs`][yian.expr.expressions.expr_refs] — the shared shape every
    ref-walk caller (`expr_refs_of`, `prompt_refs`) sees.

    Args:
        segments (`list[TypedValue]`):
            The scan from [`scan_template`][yian.expr.template.scan_template].

    Returns:
        `list[str]`:
            The reference paths across all span segments, in source order (NOT
            deduped).
    """
    refs: list = []
    for seg in segments:
        if isinstance(seg, Span):
            refs.extend(expr_refs(seg.tree))
    return refs
