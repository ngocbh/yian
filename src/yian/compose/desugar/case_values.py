"""Graph-stage: `case` then:/else: value branches -> synth CODE nodes.

A `case` branch is a sibling node, a `call(...)`, or a value. The YAML scalar style tells
them apart, so every branch is classified first:

- an unquoted (plain, or source-free) string is a node name unless it is a `call(...)`
  (a whole-value directive is left for `desugar_case_call_targets`; any other text starting
  with `call(` is a load error naming the problem), holds ` = ` after text that names a type
  (a load error: the case's `output:` gives the type, so `then: int = ${x}` is a mistake;
  quoting keeps the text), or holds a
  `${...}` span (a template value). A bare word that names no node is a load error that
  suggests the nearest node name and the quoted form, and for a YAML 1.1 boolean word
  (`yes`, `off`) the `true`/`false` to write when the case's type takes a boolean;
- a quoted string (`"..."`, `'...'`, or a block `|`/`>`) is text, so it is a value; a
  quoted `call(...)` holding a span is a load error, since quoting a call makes it text;
- a non-string scalar (`null`, bool, number, date), a mapping, or a list is a value.

A value branch needs the case's `output:` type. Each value lifts to one synth CODE node
`__value_<n>` whose inline code returns the value, and the branch is rewritten to that id
(recorded in `CaseDescriptor.value_branches`). Each string holding `${` becomes one input
binding `vN` (in `value_spans` order), so the binder evaluates it exactly like a node
input; everything else is a Python literal in the generated code (`prepared_literal`), so
a quoted text never reaches the inline-call desugar. A malformed span, a span that reads
the case itself, a `call(...)` nested inside a mapping or list, a non-finite number, and a
value that contains itself are load errors.

Runs in `_assemble` after the typed-binding splits and BEFORE `desugar_inline_calls`, so
the synth node's input bindings are walked by the inline-call and `${case.output}` passes
like any authored CODE input. Authored node ids may not use the reserved `__value_` and
`__case_` prefixes.

Located parse metadata: a synth value node and its inputs locate at the `then:`/`else:` it
was lifted from (the case's positional `then_ranges[N]`, or its `("field", "else")` range),
falling back to the case node's own range.

Imports flow down/peer only: `compose.parse.file` (descriptors), `compose.desugar.inline_calls`
(the `call(...)` recognizer, the reserved `__call_` prefix), `compose.helpers` (errors, YAML
quoting, YAML 1.1 boolean words), `expr`, `diagnostics`, and `nodes.contracts` and `typesys`
(the case's type, for the boolean fix). Only the loader pipeline, the `yian.compose`
re-export, and `compose.validate.case_values` (the shared value walk) import this back.
"""

from __future__ import annotations

import datetime
import difflib
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Optional

from yian.compose.desugar.inline_calls import (
    _SYNTH_PREFIX,
    ArgumentValueError,
    SecretArgumentError,
    desugar_call_directives,
    holds_secret_keyword,
)
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.directives import CALL_HEAD
from yian.compose.helpers.tags import QUOTED_STYLES, old_typed_text, yaml_quoted
from yian.compose.helpers.yaml_words import former_boolean_fix
from yian.compose.parse.file import CaseDescriptor, CodeDescriptor
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError, eval_binding, expr_refs_of
from yian.nodes.contracts import ValueContract
from yian.typesys.types import TypeRegistry
from yian.typesys.values import Type

VALUE_PREFIX = "__value_"
CASE_PREFIX = "__case_"
_RESERVED_PREFIXES = (VALUE_PREFIX, CASE_PREFIX)
_CLOSE_MATCH_CUTOFF = 0.6


def desugar_case_values(
    descriptors: dict,
    *,
    case_types: Mapping[str, ValueContract] | None = None,
    registry: TypeRegistry | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> dict:
    """Classify every `case` branch and lift each value branch to a synth CODE node.

    Args:
        descriptors (`dict`): The flow's parsed descriptors, keyed by node id.
        case_types (`Mapping[str, ValueContract] | None`, *optional*, defaults to `None`):
            The contract of each typed case's `output:`, keyed by case id; a bare
            YAML 1.1 boolean word (`yes`) that names no node is offered the boolean to
            write only when it fits this type. `None` offers it nowhere.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs; an unquoted branch whose text before ` = ` names a type is an
            error. `None` knows the builtin types only.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`): Source
            ownership for located errors; a fresh range-only owner when `None`.

    Returns:
        `dict`: A new descriptor map: each case with its value branches rewritten to synth
        ids, followed by one `CodeDescriptor` per value (`__value_0`, `__value_1`, ...).

    Raises:
        `LoadError`: An authored node id uses a reserved prefix; a bare word names no node;
            an unquoted branch is the old `Type = value` text; a quoted `call(...)`
            holds a span; a value branch sits on a case without `output:`; a value
            holds a key that is not a string, an unsupported scalar, a nested
            span-bearing `call(...)`, or a span that reads the case itself.

    Example:
        A `then: {a: 1, b: ${x.output}}` branch on a case with `output:` becomes a CODE node
        `__value_0` with `inputs == {"v0": "${x.output}"}` and code
        `return {'a': 1, 'b': inputs['v0']}`; the branch is rewritten to `__value_0`.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    case_types = case_types or {}
    registry = registry or {}
    for nid, desc in descriptors.items():
        for prefix in _RESERVED_PREFIXES:
            if nid.startswith(prefix):
                raise diagnostics.error(
                    f"node id {nid!r} uses the reserved prefix {prefix!r}",
                    range=getattr(desc, "range", None),
                )

    counter = itertools.count()
    suggestable = [
        nid
        for nid in descriptors
        if not nid.startswith((_SYNTH_PREFIX, *_RESERVED_PREFIXES))
    ]
    synth: dict[str, CodeDescriptor] = {}
    new_descriptors: dict = {}
    for nid, desc in descriptors.items():
        if not isinstance(desc, CaseDescriptor):
            new_descriptors[nid] = desc
            continue
        value_branches = dict(desc.value_branches)
        contract = case_types.get(nid)
        case_type = contract.type if contract is not None else None

        new_cases = list(desc.cases)
        for index, case in enumerate(desc.cases):
            if not isinstance(case, dict) or "then" not in case:
                continue
            rng = _nth(desc.then_ranges, index) or desc.range
            target = _branch_target(
                case["then"],
                style=_nth(desc.then_styles, index),
                desc=desc,
                case_type=case_type,
                label="then:",
                node_name=f"{nid} then[{index}]",
                rng=rng,
                descriptors=descriptors,
                suggestable=suggestable,
                registry=registry,
                counter=counter,
                synth=synth,
                value_branches=value_branches,
                diagnostics=diagnostics,
            )
            if target is not case["then"]:
                new_cases[index] = {**case, "then": target}
        new_else = desc.else_
        if desc.has_else:
            new_else = _branch_target(
                desc.else_,
                style=desc.else_style,
                desc=desc,
                case_type=case_type,
                label="else:",
                node_name=f"{nid} else",
                rng=desc.field_ranges.get(("field", "else")) or desc.range,
                descriptors=descriptors,
                suggestable=suggestable,
                registry=registry,
                counter=counter,
                synth=synth,
                value_branches=value_branches,
                diagnostics=diagnostics,
            )
        # Every rewritten branch records its value, so a grown map means a rewrite.
        if len(value_branches) > len(desc.value_branches):
            desc = replace(
                desc, cases=new_cases, else_=new_else, value_branches=value_branches
            )
        new_descriptors[nid] = desc
    new_descriptors.update(synth)
    return new_descriptors


def _nth(seq: list, index: int) -> Any:
    """Return the `index`-th item of a positional list, or None when it is absent."""
    return seq[index] if 0 <= index < len(seq) else None


def _branch_target(
    value: Any,
    *,
    style: Optional[str],
    desc: CaseDescriptor,
    case_type: Optional[Type],
    label: str,
    node_name: str,
    rng: Optional[SourceRange],
    descriptors: dict,
    suggestable: list[str],
    registry: TypeRegistry,
    counter: itertools.count,
    synth: dict,
    value_branches: dict,
    diagnostics: LoadDiagnostics,
) -> Any:
    """Return the branch's target: the value unchanged, or the id of its synth value node.

    Args:
        value (`Any`): The authored `then:`/`else:` value.
        style (`Optional[str]`): Its YAML scalar style; `None` (unknown) reads as plain.
        desc (`CaseDescriptor`): The case that owns the branch.
        case_type (`Optional[Type]`): The case's `output:` type, or `None` when it has
            none.
        label (`str`): `"then:"` or `"else:"`, for messages.
        node_name (`str`): Display name for a synth value node.
        rng (`Optional[SourceRange]`): The branch's source range.
        descriptors (`dict`): The flow's descriptors; a bare word must name one.
        suggestable (`list[str]`): Authored node ids offered as close matches.
        registry (`TypeRegistry`): The flow's typedefs, for the old `Type = value` text.
        counter (`itertools.count`): The synth-id counter of this pass.
        synth (`dict`): Collects synth value nodes by id.
        value_branches (`dict`): Collects synth id -> authored value for `desc`.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Returns:
        `Any`: The (possibly rewritten) branch target.

    Raises:
        `LoadError`: See `desugar_case_values`.
    """
    host = f"case node {desc.id!r} {label}"
    if not _is_value(
        value,
        style=style,
        case_type=case_type,
        host=host,
        label=label,
        descriptors=descriptors,
        suggestable=suggestable,
        registry=registry,
        rng=rng,
        diagnostics=diagnostics,
    ):
        return value
    if desc.outputs is None:
        if value is None:
            raise diagnostics.error(
                f"{host} is empty; name a node, or add `output: Optional(...)` for a null "
                "value",
                range=rng,
            )
        raise diagnostics.error(
            f"{host} a value branch needs the case's type; add `output: <Type>` to "
            f"{desc.id!r}",
            range=rng,
        )
    synth_id = f"{VALUE_PREFIX}{next(counter)}"
    node = _value_node(
        synth_id,
        value,
        node_name=node_name,
        range=rng,
        diagnostics=diagnostics,
        host=host,
        label=label,
    )
    for source in node.inputs.values():
        if _reads_case(source, desc.id):
            raise diagnostics.error(f"{host} the value reads {desc.id!r} itself", range=rng)
    synth[synth_id] = node
    value_branches[synth_id] = value
    return synth_id


def _is_value(
    value: Any,
    *,
    style: Optional[str],
    case_type: Optional[Type],
    host: str,
    label: str,
    descriptors: dict,
    suggestable: list[str],
    registry: TypeRegistry,
    rng: Optional[SourceRange],
    diagnostics: LoadDiagnostics,
) -> bool:
    """Classify one branch: True for a value, False for a node name or a `call(...)`.

    Args:
        value (`Any`): The authored `then:`/`else:` value.
        style (`Optional[str]`): Its YAML scalar style; `None` (unknown) reads as plain.
        case_type (`Optional[Type]`): The case's `output:` type, or `None`; a YAML 1.1
            boolean word that names no node is offered the boolean when it fits.
        host (`str`): Message prefix naming the branch, such as `"case node 'r' then:"`.
        label (`str`): `"then:"` or `"else:"`, for the quoted form a message suggests.
        descriptors (`dict`): The flow's descriptors; a bare word must name one.
        suggestable (`list[str]`): Authored node ids offered as close matches.
        registry (`TypeRegistry`): The flow's typedefs, for the old `Type = value` text.
        rng (`Optional[SourceRange]`): The branch's source range.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Returns:
        `bool`: Whether the branch is a value to lift.

    Raises:
        `LoadError`: An unquoted branch starts with `call(` but is not exactly one
            well-formed directive, is the old `Type = value` text, or names no node; a
            quoted `call(...)` holds a span.
    """
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    is_call = CALL_HEAD.match(stripped) is not None
    if style in QUOTED_STYLES:
        if is_call and "${" in value:
            raise diagnostics.error(
                f"{host} a quoted `call(...)` is text, so it can't hold `${{...}}` here; "
                "drop the quotes to call the flow, or reword the text",
                range=rng,
            )
        return True
    if is_call:
        # Checked here, where the style is known; lifted by `desugar_case_call_targets`.
        try:
            desugar_call_directives(value, lambda: "__check")
        except SecretArgumentError as exc:
            raise diagnostics.error(exc.message_for(host), range=rng) from exc
        except ExpressionError as exc:
            # Quoting a call that holds a span is its own error, so no quoting fix then;
            # nor for one with a secret keyword, whose text must not be echoed, nor for an
            # unreadable argument, whose own message says to quote that argument.
            hint = (
                f". If you meant the text, quote it: `{label} {yaml_quoted(stripped)}`"
                if "${" not in value
                and not holds_secret_keyword(value)
                and not isinstance(exc, ArgumentValueError)
                else ""
            )
            if isinstance(exc, ArgumentValueError):
                # The target is a single call; only one of its arguments is wrong.
                raise diagnostics.error(f"{host} {exc}", range=rng) from exc
            raise diagnostics.error(
                f"{host} a branch target must be a node id or a single call: {exc}{hint}",
                range=rng,
            ) from exc
        return False
    old = old_typed_text(value, registry)
    if old is not None:
        rewrite = f", so write `{label} {old[2]}`" if old[2] else ""
        raise diagnostics.error(
            f"{host} `{value}` is not a branch; the case's `output:` gives the "
            f"type{rewrite}. If you meant the text, quote it: "
            f"`{label} {yaml_quoted(value)}`",
            range=rng,
        )
    if "${" in value:
        return True
    if value not in descriptors:
        matches = difflib.get_close_matches(
            value, suggestable, n=1, cutoff=_CLOSE_MATCH_CUTOFF
        )
        suggestion = f" Did you mean `{matches[0]}`?" if matches else ""
        raise diagnostics.error(
            f"{host} `{value}` names no node.{suggestion} If you meant the text "
            f'"{value}", quote it: `{label} {yaml_quoted(value)}`'
            f"{_boolean_hint(value, case_type)}",
            range=rng,
        )
    return False


def _boolean_hint(word: str, case_type: Optional[Type]) -> str:
    """Return the boolean clause for a YAML 1.1 boolean word the case's type takes, else `""`.

    Args:
        word (`str`):
            The bare branch word that names no node.
        case_type (`Optional[Type]`):
            The case's `output:` type, or `None` when it has none.

    Returns:
        `str`:
            Such as `", or write `true` for the boolean"`, or `""`.
    """
    if case_type is None:
        return ""
    fix = former_boolean_fix(word, case_type)
    if fix is None:
        return ""
    return f", or write `{'true' if fix[0] else 'false'}` for the boolean"


def _value_node(
    synth_id: str,
    value: Any,
    *,
    node_name: str,
    range: Optional[SourceRange],
    diagnostics: LoadDiagnostics,
    host: str,
    label: str,
) -> CodeDescriptor:
    """Build the synth CODE node that produces one branch value.

    Args:
        synth_id (`str`): The node id (`__value_<n>`).
        value (`Any`): The authored value: a scalar, mapping, or list, possibly holding
            `${...}` spans in its strings.
        node_name (`str`): Display name, such as `"route then[0]"`.
        range (`Optional[SourceRange]`): The branch's source range; the node and each of its
            inputs locate here.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.
        host (`str`): Message prefix naming the branch, such as `"case node 'r' then:"`.
        label (`str`): The branch key, `"then:"` or `"else:"`, for the fix a message
            suggests.

    Returns:
        `CodeDescriptor`: An untyped inline CODE node. Each string holding `${` is one
        input `v0`, `v1`, ... (walk order); the code `return <expr>` builds the value with
        every other part written as a Python literal.

    Raises:
        `LoadError`: See `_literal`.

    Example:
        `_value_node("__value_0", {"t": "${input.t}", "done": True}, ...)` has
        `inputs == {"v0": "${input.t}"}` and code `return {'t': inputs['v0'], 'done': True}`.
    """
    names = {span.path: span.name for span in value_spans(value)}
    inputs: dict[str, str] = {}
    site = _Site(host=host, label=label, range=range, diagnostics=diagnostics)
    expr = _literal(value, names, inputs, path=(), site=site, containers=set())
    return CodeDescriptor(
        id=synth_id,
        node_name=node_name,
        code=f"return {expr}",
        inputs=inputs,
        outputs=None,
        range=range,
        field_ranges={("input", name): range for name in inputs} if range else {},
    )


ValuePath = tuple[Any, ...]
"""Mapping keys and list indices from a branch value's root to one of its parts."""


@dataclass(frozen=True)
class ValueSpan:
    """One string holding `${` in a branch value, bound to an input of its value node.

    Attributes:
        path (`ValuePath`): Where the string sits in the value; `()` for the whole value.
        name (`str`): The value node's input that binds it (`v0`, `v1`, ...).
        text (`str`): The string as authored.
    """

    path: ValuePath
    name: str
    text: str


def value_spans(value: Any) -> list[ValueSpan]:
    """Return the strings holding `${` in a branch value, in input order.

    The walk visits mapping values and list items in order; the value node's inputs are
    named `v0`, `v1`, ... in this order. A mapping or list that contains itself is not
    entered again (lifting it is a load error).

    Args:
        value (`Any`): An authored branch value.

    Returns:
        `list[ValueSpan]`: One entry per span-bearing string, in input order.
    """
    spans: list[ValueSpan] = []
    active: set[int] = set()

    def walk(item: Any, path: ValuePath) -> None:
        if isinstance(item, str):
            if "${" in item:
                spans.append(ValueSpan(path=path, name=f"v{len(spans)}", text=item))
            return
        if not isinstance(item, (dict, list)) or id(item) in active:
            return
        active.add(id(item))
        children = item.items() if isinstance(item, dict) else enumerate(item)
        for key, child in children:
            walk(child, path + (key,))
        active.discard(id(item))

    walk(value, ())
    return spans


def prepared_literal(value: Any) -> Any:
    """Return a span-free branch value as its value node produces it.

    A string is `$$`-unescaped, as the binder does for span-free text, and a date or
    datetime becomes its ISO text; mappings and lists are prepared item by item.

    Args:
        value (`Any`): A span-free branch value or a part of one.

    Returns:
        `Any`: The value the lifted code returns for it.
    """
    if isinstance(value, str):
        return eval_binding(value, lambda path: None)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: prepared_literal(item) for key, item in value.items()}
    if isinstance(value, list):
        return [prepared_literal(item) for item in value]
    return value


@dataclass(frozen=True)
class _Site:
    """Where a value sits, for located errors while it is written as Python."""

    host: str
    label: str
    range: Optional[SourceRange]
    diagnostics: LoadDiagnostics


def _literal(
    value: Any,
    names: dict[ValuePath, str],
    inputs: dict[str, str],
    *,
    path: ValuePath,
    site: _Site,
    containers: set[int],
) -> str:
    """Return Python source that builds `value`, adding each span-bearing string to `inputs`.

    Args:
        value (`Any`): A branch value or a part of one.
        names (`dict[ValuePath, str]`): The input name of each span-bearing string, by
            path (from `value_spans`).
        inputs (`dict[str, str]`): Collects input name -> template string.
        path (`ValuePath`): Where `value` sits in the branch value; `()` for the whole.
        site (`_Site`): Where the branch sits, for located errors.
        containers (`set[int]`): `id()`s of the mappings and lists enclosing `value`, so a
            YAML alias that contains itself is caught instead of recursing forever.

    Returns:
        `str`: A Python expression that evaluates to the value.

    Raises:
        `LoadError`: A nested string starts with `call(`, a span is malformed, a number is
            not finite, a mapping key is not a string, a value contains itself, or a scalar
            has no literal form.
    """
    if isinstance(value, str):
        if path and CALL_HEAD.match(value.strip()):
            raise site.diagnostics.error(
                f"{site.host} a `call(...)` nested in a value is not supported; write the "
                f"call as the branch (`{site.label} call(...)`) or in a node, or reword the "
                "text",
                range=site.range,
            )
        if "${" in value:
            try:
                expr_refs_of(value)
            except ExpressionError as exc:
                raise site.diagnostics.error(f"{site.host} {exc}", range=site.range) from exc
            name = names[path]
            inputs[name] = value
            return f"inputs[{name!r}]"
        return repr(prepared_literal(value))
    if value is None or isinstance(value, (bool, int)):
        return repr(value)
    if isinstance(value, float):
        # False for nan as well as for both infinities.
        if not float("-inf") < value < float("inf"):
            raise site.diagnostics.error(
                f"{site.host} numbers must be finite (got {value!r})", range=site.range
            )
        return repr(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return repr(prepared_literal(value))
    if isinstance(value, (dict, list)):
        if id(value) in containers:
            raise site.diagnostics.error(
                f"{site.host} a branch value can't contain itself", range=site.range
            )
        containers.add(id(value))
        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise site.diagnostics.error(
                        f"{site.host} record keys must be strings (got {key!r})",
                        range=site.range,
                    )
                item_expr = _literal(
                    item,
                    names,
                    inputs,
                    path=path + (key,),
                    site=site,
                    containers=containers,
                )
                parts.append(f"{key!r}: {item_expr}")
            expr = "{" + ", ".join(parts) + "}"
        else:
            items = [
                _literal(
                    item,
                    names,
                    inputs,
                    path=path + (index,),
                    site=site,
                    containers=containers,
                )
                for index, item in enumerate(value)
            ]
            expr = "[" + ", ".join(items) + "]"
        containers.discard(id(value))
        return expr
    raise site.diagnostics.error(
        f"{site.host} a branch value can't hold a {type(value).__name__}",
        range=site.range,
    )


def _reads_case(source: str, case_id: str) -> bool:
    """Return True when a template string (already parsed by `_literal`) reads
    `${<case_id>.output...}`."""
    for ref in expr_refs_of(source):
        parts = ref.split(".")
        if len(parts) >= 2 and parts[0] == case_id and parts[1] == "output":
            return True
    return False
