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

Each synth node from a node binding records that node as its `host`; the host-edge step
(`compose.edges.hosts`) gives it the host's CASE routes, `depends_on`, and `runs_after`, and
gates it on the host's other inputs, so it runs only when the host can. A synth node from the
flow `output:`/`asserts:` has no host.

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

A keyword argument under a secret name (`yian.secret_names`) must be a binding or
`null`; a literal there raises `SecretArgumentError`, which every host formats with its
own label and without echoing the value.

Imports flow DOWN/peer only: `compose.parse.file` (descriptors), `compose.nodes.bindings` (the
kind-blind binding-field walk) + `compose.helpers` (errors, secret literals, YAML style and
quoting; peers), `compose.source` (a bare argument's plain-scalar reading), `expr` (the
ref-walk + the span flow-call detector), `diagnostics` (SourceRange), `secret_names`. The
loader, the case desugars, and `compose.validate` (the assert and secret-literal checks)
import this back.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import replace
from typing import Any, Callable, Collection, Optional

from yian.diagnostics import OriginRef, SourceRange
from yian.nodes.contracts import Assertion
from yian.expr import (
    ExpressionError,
    expr_refs_of,
)
from yian.expr.template import (
    InlineCall,
    _find_paren_end,
    _split_calls_aware,
    flow_call_callees_in_spans,
)
from yian.compose.nodes.bindings import map_binding_strings
from yian.compose.desugar.inline_bodies import INLINE_FLOW_PREFIX
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.secret_literals import (
    secret_literal_message,
    secret_literal_path,
)
from yian.compose.helpers.typed_bindings import CALL_HEAD, QUOTED_STYLES, yaml_quoted
from yian.compose.parse.file import (
    CallDescriptor,
    NodeDescriptor,
)
from yian.compose.source import SourceSyntaxError, read_plain_scalar
from yian.secret_names import is_secret_name

# The reserved id prefix for synthesized inline-call nodes. A monotonic counter per
# flow/def keeps them unique; the collision guard reserves the namespace from users.
_SYNTH_PREFIX = "__call_"
_FLOW_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
# Text before a lone `=` that ends in a separate word: `1 b` in `a=1 b=2`, a missing comma.
_WORD_BEFORE_EQUALS = re.compile(r"\s([A-Za-z_][A-Za-z0-9_-]*)\s*$")
# A name directly before `=`, anywhere in a directive's raw text.
_NAME_BEFORE_EQUALS = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)\s*=")


def holds_secret_keyword(text: str) -> bool:
    """Return whether `text` writes a name under a secret name directly before `=`.

    A malformed `call(...)` whose text holds such a keyword may hold a credential in a
    part the parse never reached, so its error must not quote the text.

    Args:
        text (`str`): The raw directive text.

    Returns:
        `bool`: `True` when some `name=` in `text` has a secret name.
    """
    return any(is_secret_name(m.group(1)) for m in _NAME_BEFORE_EQUALS.finditer(text))


def _quoted(part: str, text: str, lead: str = " ") -> str:
    """Return `part` quoted after `lead` for an error message, or `""` when `text`
    holds a secret keyword."""
    return "" if holds_secret_keyword(text) else f"{lead}{part!r}"


class SecretArgumentError(ExpressionError):
    """A `call(...)` keyword argument under a secret name holds a literal value.

    Its message never includes the value. A host formats it with `message_for`,
    which names the host and offers no quoting fix: quoting keeps the value literal.

    Args:
        name (`str`): The keyword argument's name.
        path (`tuple[str | int, ...]`, *optional*, defaults to `()`): The position
            of the literal, as `secret_literal_path` returns it.
    """

    def __init__(self, name: str, path: tuple[str | int, ...] = ()):
        self.name = name
        self.path = path
        super().__init__(secret_literal_message(self._slot(), name, path))

    def message_for(self, host: str) -> str:
        """Return the message prefixed with `host`, without a trailing `:` of its own.

        Args:
            host (`str`): The host label, e.g. `"node 'b'"` or `"case node 'c' then:"`.

        Returns:
            `str`: The located message, e.g. `"node 'b': call(...) argument ..."`.
        """
        return secret_literal_message(
            self._slot(), self.name, self.path, owner=host.rstrip(": ")
        )

    def _slot(self) -> str:
        """The argument as the message names it."""
        return f"call(...) argument {self.name!r}"


class ArgumentValueError(ExpressionError):
    """A `call(...)` keyword argument's bare value can't be read as a YAML scalar.

    Its message never includes the value; quoting the value keeps it as text.

    Args:
        flow_id (`str`): The called flow's id.
        name (`str`): The keyword argument's name.
        reason (`str`): Why the value can't be read.
    """

    def __init__(self, flow_id: str, name: str, reason: str):
        super().__init__(
            f"call({flow_id}, ...) keyword {name!r} value can't be read as YAML "
            f"({reason}); quote it to keep it as text"
        )


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
    typed_outputs: Collection[Any] = (),
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
    expression, so its quoting never makes it text. ``typed_outputs`` names the outputs
    written `Type = value` (`None` for a single output); like a typed node input, a
    malformed directive there gets no quoting fix, since quoting would drop the type.

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
                host_id=nid,
                range=origins.get(call.id, rng),
                diagnostics=diagnostics,
            )

    new_outputs, out_calls = _desugar_outputs(
        outputs_section,
        mint,
        range=outputs_range,
        style_at=output_style_at,
        typed=typed_outputs,
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
    return map_output_bindings(outputs, lambda _key, value: fn(value))


def map_output_bindings(outputs: Any, fn: Callable[[Any, str], Any]) -> Any:
    """Apply `fn(key, value)` to each STRING binding in the flow `output:` section; `key`
    is the record field name, or `None` for a bare single output."""
    if isinstance(outputs, dict):
        return {k: (fn(k, v) if isinstance(v, str) else v) for k, v in outputs.items()}
    if isinstance(outputs, str):
        return fn(None, outputs)
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
            # Quoting a typed binding would drop its type, so no quoting fix is offered.
            quotable=not (field == "inputs" and key in desc.input_types),
            owner=f"node {host!r}",
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
    expression: bool = False,
    owner: Optional[str] = None,
    diagnostics: LoadDiagnostics | None = None,
) -> str:
    """Desugar one binding string. A WHOLE-VALUE `call(<flow-id>, kw=...)` directive lifts
    to a synth call node (accumulated in `calls`, the value rewritten to `${<synth>.output}`).
    A flow call embedded INSIDE a `${...}` span is the retired `${flow(args)}` form — a
    located `LoadError` pointing at the `call(...)` directive. Any other value (plain refs,
    pure builtins, literals) is returned unchanged.

    ``style`` is the value's YAML scalar style. A quoted value is text: it is never a
    directive, and a quoted `call(...)` holding a span is a load error (its author most
    likely meant the call). ``expression`` is True for an assert, whose quoting does not
    make it text. A malformed directive's error suggests quoting only when that gives
    the text: not for an assert, a span-bearing value, a ``quotable=False`` (typed)
    binding, or an unreadable argument (`ArgumentValueError` says to quote that
    argument). Maps `ExpressionError` (a malformed directive or an unparseable span) to
    a located `LoadError` at ``range``; a literal secret argument (`SecretArgumentError`)
    is prefixed with ``owner`` (default ``host``) and gets no quoting fix."""
    diagnostics = diagnostics or LoadDiagnostics()
    quoted = not expression and style in QUOTED_STYLES
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
        except SecretArgumentError as exc:
            raise diagnostics.error(exc.message_for(owner or host), range=range) from exc
        except ExpressionError as exc:
            hint = (
                f". If you meant the text, quote the whole value: "
                f"`{yaml_quoted(value.strip())}`"
                if quotable
                and not expression
                and not isinstance(exc, ArgumentValueError)
                and "${" not in value
                and not holds_secret_keyword(value)
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
    Each keyword value passes through `_arg_source`: a bare value reads like a named
    `input:` value (`window=30` -> int 30, `1e3` -> float 1000.0), a quoted scalar
    unwraps to text, and a `${…}`-bearing value stays a binding string. An unquoted
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
    keyword passed more than once. A keyword under a secret name whose value is a literal
    raises `SecretArgumentError`, before any check that would echo the value. When the
    text holds a keyword under a secret name (`holds_secret_keyword`), a malformed
    directive's message does not quote any part of it. A bare keyword value the flow's
    YAML rules can't build (an impossible date) raises `ArgumentValueError`."""
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
                f"call({flow_id}, ...) passes a positional argument"
                f"{_quoted(pair.strip(), value)}; "
                "every argument after the flow id must be a keyword (name=value)"
            )
        name = parts[0].strip()
        if not name or any(ch.isspace() for ch in name):
            raise ExpressionError(
                f"call({flow_id}, ...) argument{_quoted(pair.strip(), value)} needs one "
                "name before "
                f"`=`; {'add the missing comma' if name else 'write the keyword name'}"
            )
        arg_value = "=".join(parts[1:])
        _check_secret_argument(name, arg_value)
        inner_keywords = _inner_keywords(parts[1:])
        if inner_keywords:
            for inner_name, inner_value in inner_keywords:
                if is_secret_name(inner_name):
                    # Read as the comma had been there; a binding falls to the error below.
                    _check_secret_argument(inner_name, inner_value)
            raise ExpressionError(
                f"call({flow_id}, ...) keyword {name!r} value"
                f"{_quoted(arg_value.strip(), value)} looks "
                "like two keywords; add the missing comma, or quote the value"
            )
        new_value, inner = desugar_call_directives(
            arg_value, mint
        )  # inner-first recursion
        calls.extend(inner)
        if name in args:
            raise ExpressionError(
                f"call({flow_id}, ...) passes keyword {name!r} more than once"
            )
        try:
            args[name] = _arg_source(new_value)
        except SourceSyntaxError as exc:
            raise ArgumentValueError(flow_id, name, str(exc)) from None
    cid = mint()
    calls.append(InlineCall(id=cid, callee=flow_id, args=args))
    return "${" + cid + ".output}", calls


def _check_secret_argument(name: str, arg_value: str) -> None:
    """Reject a literal keyword value under a secret name.

    A value that can't be read under another name is left for the directive's own
    read to report.

    Args:
        name (`str`): The keyword's name.
        arg_value (`str`): Its raw value text.

    Raises:
        `SecretArgumentError`: `name` is a secret name and the value, read as the
            directive reads it, is neither a binding nor `null`, or can't be read. A
            nested directive is a binding only if it parses: its parse error could
            quote the value.
        `ArgumentValueError`: A nested directive under a secret name passes a bare
            value that can't be read; its message does not echo the value.
    """
    if CALL_HEAD.match(arg_value.strip()):
        if is_secret_name(name):
            try:
                desugar_call_directives(arg_value, lambda: "_")
            except (SecretArgumentError, ArgumentValueError):
                raise
            except ExpressionError:
                raise SecretArgumentError(name) from None
        return  # its own arguments are checked when it is desugared
    try:
        value = _arg_source(arg_value)
    except SourceSyntaxError:
        if is_secret_name(name):
            raise SecretArgumentError(name) from None
        return
    path = secret_literal_path(name, value)
    if path is not None:
        raise SecretArgumentError(name, path)


def _arg_source(value: str) -> Any:
    """Read one `call(...)` keyword value the way the named form reads its `input:`.

    Args:
        value (`str`): The keyword's raw value text.

    Returns:
        `Any`: For a fully quoted scalar (`"…"`/`'…'`), its inner text, a binding
            template if it interpolates (`"hi ${name}"` -> `hi ${name}`); for an
            unquoted `${…}`-bearing value, the binding string; else the trimmed text
            read as one plain scalar like a named `input:` value (`30` -> int,
            `1e3` -> float, `~` or an empty value -> `None`, `None` -> text). YAML
            structure such as `[1]` stays text: a bare argument is one scalar.

    Raises:
        `SourceSyntaxError`: The bare text resolves to a type that can't build it,
            such as an impossible date; the message does not echo the text.
    """
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] in ("'", '"') and stripped[-1] == stripped[0]:
        return stripped[1:-1]  # a quoted scalar -> its inner text (template or literal)
    if "${" in stripped:
        return stripped
    return read_plain_scalar(stripped)


def _inner_keywords(parts: list[str]) -> list[tuple[str, str]]:
    """The would-be keywords a keyword value, split on its top-level `=`, holds as
    `... name=...`.

    Only a lone `=` after a separate word counts (`1 b=2`); `==`, `!=`, `<=`, `>=`, and a
    glued `a=b` stay part of the value's text.

    Args:
        parts (`list[str]`): The value split on each top-level `=`.

    Returns:
        `list[tuple[str, str]]`: The word before each such `=` and the text after that
            `=` to the end of the value; empty when there is none.
    """
    keywords = []
    for index, (left, right) in enumerate(zip(parts, parts[1:])):
        if not left or not right or left[-1] in "!<>":
            continue
        match = _WORD_BEFORE_EQUALS.search(left)
        if match:
            keywords.append((match.group(1), "=".join(parts[index + 1 :])))
    return keywords


def _match_directive(value: str) -> Optional[tuple]:
    """`value` -> `(flow_id, remaining_args_str)` if it is a whole-value `call(...)`
    directive, or None when its trimmed text does not start with `call(`.

    Returns `remaining_args_str` = the raw args substring AFTER the flow id's comma
    (empty when `call(f)` has no kwargs); `desugar_call_directives` checks those args.
    Raises `ExpressionError` when the value starts with `call(` but its parens are
    unbalanced, text follows the close paren, or the first arg is not a bare flow id;
    the message quotes no part of a value that `holds_secret_keyword`."""
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
            f"text follows the closing `)`{_quoted(trailing, s, ': ')}; a call(...) "
            "must be the whole value"
        )
    args_str = s[open_idx + 1 : close_idx]
    first = _split_calls_aware(args_str, ",")[0].strip()
    if not _FLOW_ID.fullmatch(first):
        raise ExpressionError(
            "a call(...) must start with a flow id (a bare name) and a comma"
            f"{_quoted(first, s, ', got ')}"
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
    typed: Collection[Any] = (),
    diagnostics: LoadDiagnostics,
) -> tuple:
    """Desugar the flow `output:` section, returning `(new_outputs, inline_calls)`.
    ``range`` is the `output:` section source range; ``style_at(key)`` gives an output's
    YAML scalar style (`key` is None for a single output; plain when absent), and
    ``typed`` names the outputs written `Type = value`."""
    calls: list = []
    new_outputs = map_output_bindings(
        outputs,
        lambda key, value: _desugar_value(
            value,
            mint,
            calls,
            "flow outputs",
            range=range,
            style=style_at(key) if style_at is not None else None,
            quotable=key not in typed,
            diagnostics=diagnostics,
        ),
    )
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
            expression=True,
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
    host_id: Optional[str] = None,
    range: Optional[SourceRange] = None,
    diagnostics: LoadDiagnostics | None = None,
) -> CallDescriptor:
    """One `InlineCall` -> a synth `CallDescriptor` (`over=None`), after its guards.

    Authored calls into the compiler-private inline-flow namespace are rejected at this
    shared boundary, before any synth descriptor can reach child resolution. The synth node
    carries its OWN provenance: `range` is the site it desugared from (a node input's binding
    line, a case `then:`/`else:` line, or a section item). It owns no authored fields, so
    `field_ranges` is empty and a bad ref inside the directive carets `range`, never a host
    line.

    Args:
        call (`InlineCall`):
            The recognized directive: synth id, callee, and keyword args.
        host (`str`):
            The site named in error messages (`node 'x'`, `flow outputs`, ...).
        host_id (`str`, *optional*, defaults to `None`):
            The node the call was lifted out of, recorded as the synth node's `host` so it
            takes that node's CASE routes, `depends_on`, and `runs_after`. `None` for a call
            with no host node (flow `output:`/`asserts:`, or a branch target).
        range (`SourceRange`, *optional*, defaults to `None`):
            The site the directive desugared from.
        diagnostics (`LoadDiagnostics`, *optional*, defaults to `None`):
            Locates load errors. `None` uses a range-only default.

    Returns:
        `CallDescriptor`:
            The synth call node.

    Raises:
        `LoadError`:
            Raised when the callee is in the private inline-flow namespace or an arg
            captures `${item}`.
    """
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
        host=host_id,
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
