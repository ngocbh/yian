"""Graph-stage: a `case` branch names a node or calls a flow, and a `case` has no `output:`.

A `case` only routes. Each `then:`/`else:` is classified by its YAML scalar style:

- an unquoted string names a node unless it is a `call(...)`. A whole-value directive is
  left for `desugar_case_call_targets`; any other text starting with `call(` is a load
  error naming the problem. An unquoted bare word that names no node is a load error that
  suggests the nearest node and the value entry to write for the text;
- anything else is a value: quoted text (`"..."`, `'...'`, or a block `|`/`>`), `null` or
  an empty `else:`, a number, a bool, a date, a mapping, a list, and unquoted text holding
  `${`. A value branch is a load error that shows the rewrite: a value entry in `nodes:`
  (`route_then_0: !str approve`) and the branch naming it. The entry is tagged with the
  case's former `output:` type when a tag can spell it, else with a tag guessed from the
  YAML value. A quoted `call(...)` holding a span is its own error, since quoting a call
  makes it text and a value entry can't hold that text either.

A case `output:` (`output: str`, or tagged `output: !str`) is a load error at the
`output:` that says to tag the result where it is read (`x: !str ${route.output}`).
A case with both reports its first value branch first.

Runs in `_assemble` before `desugar_inline_calls`, so the branches are still as authored,
and after `check_authored_bindings`, so a secret nested in a value branch is reported as a
secret literal before a message here could echo it.

Imports flow down/peer only: `compose.desugar.inline_calls` (the `call(...)` check and the
synth call prefix), `compose.helpers` (errors, the `call(` head, YAML quoting, tag
spelling), `compose.parse` (descriptors, identifiers, the example type of a value),
`compose.source`, `diagnostics`, `expr`, and `typesys`. Only the loader pipeline imports
this back.
"""

from __future__ import annotations

import datetime
import difflib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from yian.compose.desugar.inline_calls import (
    _SYNTH_PREFIX,
    ArgumentValueError,
    SecretArgumentError,
    desugar_call_directives,
    holds_secret_keyword,
)
from yian.compose.helpers.directives import CALL_HEAD
from yian.compose.helpers.errors import LoadDiagnostics, LoadError, source_text
from yian.compose.helpers.tags import (
    QUOTED_STYLES,
    old_form_suggestion,
    old_typed_text,
    resolve_type_text,
    suggest_tag,
    suggested_literal,
    yaml_quoted,
)
from yian.compose.helpers.types import read_type
from yian.compose.parse.file import CaseDescriptor
from yian.compose.parse.nodes import example_type, is_identifier
from yian.compose.source import SourceCursor, SourceSyntaxError, parse_yaml_literal
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError
from yian.typesys import TypeCheckError
from yian.typesys.types import TypeRegistry
from yian.typesys.values import Type, canonical_type_label

_CLOSE_MATCH_CUTOFF = 0.6
# A value longer than this is cut in a message; the author writes it as a block.
_SHOWN_LIMIT = 60
# Characters that end or nest a plain scalar inside a YAML flow mapping or list.
_FLOW_SPECIAL = frozenset(",[]{}:#&*!|>'\"%@`")

# Returns the authored `output:` of a case, or `None` when it has none.
OutputReader = Callable[[str], Optional[SourceCursor]]


@dataclass(frozen=True)
class _Branch:
    """One authored `then:`/`else:` of a case.

    Attributes:
        label (`str`): `"then:"` or `"else:"`.
        value (`Any`): The authored value.
        style (`Optional[str]`): Its YAML scalar style; `None` (unknown) reads as plain.
        range (`Optional[SourceRange]`): The branch's range, else the case's range.
        entry (`str`): The value entry id a rewrite suggests, such as `route_then_0`.
    """

    label: str
    value: Any
    style: Optional[str]
    range: Optional[SourceRange]
    entry: str


@dataclass(frozen=True)
class _CaseOutput:
    """The `output:` a case declares, read for its migration message.

    Attributes:
        type (`Optional[Type]`): The declared type, or `None` when it does not resolve.
        text (`Optional[str]`): The type as written when it is text (a tag or a type
            name), else `None`.
        range (`Optional[SourceRange]`): The `output:` range, else the case's range.
    """

    type: Optional[Type]
    text: Optional[str]
    range: Optional[SourceRange]


def check_case_branches(
    descriptors: Mapping[str, Any],
    *,
    registry: TypeRegistry | None = None,
    output_at: OutputReader | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Reject every `case` branch that is not a node or a call, and every case `output:`.

    Args:
        descriptors (`Mapping[str, Any]`): The flow's parsed descriptors, keyed by node
            id; branches are as authored.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, for spelling a case's former `output:` type as a tag. `None` knows
            the builtin types only.
        output_at (`OutputReader | None`, *optional*, defaults to `None`): Returns a
            case's authored `output:` by node id, or `None` when it has none. It sees a
            tagged `output: !int`, which leaves the descriptor's `outputs` empty. `None`
            reads `outputs` from the descriptor.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`): Source
            ownership for located errors; a fresh range-only owner when `None`.

    Raises:
        `LoadError`: A branch holds a value, is an unquoted word that names no node, is
            text starting with `call(` that is not one well-formed directive, or is a
            quoted `call(...)` holding a span; or a case declares `output:`. Cases are
            checked in order, each one's `then:` branches, its `else:`, then its
            `output:`.

    Example:
        A case `route` with `then: "approve"` fails with ``case node 'route' then: a
        branch names a node or calls a flow; put the value in its own entry in `nodes:`
        and name it: `route_then_0: !str approve` and `then: route_then_0` ``.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    registry = registry or {}
    authored = [nid for nid in descriptors if not nid.startswith(_SYNTH_PREFIX)]
    for nid, desc in descriptors.items():
        if not isinstance(desc, CaseDescriptor):
            continue
        output = _read_output(desc, output_at, registry)
        for branch in _branches(desc):
            _check_branch(
                branch,
                case_id=nid,
                output=output,
                descriptors=descriptors,
                authored=authored,
                registry=registry,
                diagnostics=diagnostics,
            )
        if output is not None:
            raise diagnostics.error(
                f"case node {nid!r}: a `case` routes and has no `output:`; tag its result "
                f"where you read it{_read_example(nid, output, registry)}",
                range=output.range,
            )


def _branches(desc: CaseDescriptor) -> list[_Branch]:
    """Return the case's `then:` branches in order, then its `else:` when written."""
    branches = []
    for index, case in enumerate(desc.cases):
        if not isinstance(case, dict) or "then" not in case:
            continue
        branches.append(
            _Branch(
                label="then:",
                value=case["then"],
                style=_nth(desc.then_styles, index),
                range=_nth(desc.then_ranges, index) or desc.range,
                entry=f"{desc.id}_then_{index}",
            )
        )
    if desc.has_else:
        branches.append(
            _Branch(
                label="else:",
                value=desc.else_,
                style=desc.else_style,
                range=desc.field_ranges.get(("field", "else")) or desc.range,
                entry=f"{desc.id}_else",
            )
        )
    return branches


def _nth(seq: list, index: int) -> Any:
    """Return the `index`-th item of a positional list, or None when it is absent."""
    return seq[index] if 0 <= index < len(seq) else None


def _check_branch(
    branch: _Branch,
    *,
    case_id: str,
    output: Optional[_CaseOutput],
    descriptors: Mapping[str, Any],
    authored: list[str],
    registry: TypeRegistry,
    diagnostics: LoadDiagnostics,
) -> None:
    """Accept a node name or a `call(...)` branch; reject anything else with its rewrite.

    Args:
        branch (`_Branch`): The branch.
        case_id (`str`): The case that owns it.
        output (`Optional[_CaseOutput]`): The case's former `output:`, which tags the value
            entry a rewrite shows; `None` guesses the tag from the value.
        descriptors (`Mapping[str, Any]`): The flow's descriptors; a bare word must name
            one, and a suggested entry id must not.
        authored (`list[str]`): Authored node ids offered as close matches.
        registry (`TypeRegistry`): The flow's typedefs.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Raises:
        `LoadError`: See `check_case_branches`.
    """
    host = f"case node {case_id!r} {branch.label}"
    value = branch.value
    if isinstance(value, str) and branch.style not in QUOTED_STYLES:
        if CALL_HEAD.match(value.strip()) is not None:
            _check_call(branch, host, descriptors=descriptors, diagnostics=diagnostics)
            return
        old = old_typed_text(value, registry)
        if old is not None and old[2]:
            # The old `Type = value` text: the type it names tags the entry.
            entry = _free_id(branch.entry, descriptors)
            shown = old_form_suggestion(
                entry, old[0], old[1], old[2], registry=registry, placeholder=None
            )
            raise _value_error(host, branch, entry, shown, diagnostics)
        if "${" not in value:
            if value in descriptors:
                return
            raise _unknown_word_error(
                host,
                branch,
                descriptors=descriptors,
                authored=authored,
                diagnostics=diagnostics,
            )
    elif isinstance(value, str) and CALL_HEAD.match(value.strip()) and "${" in value:
        raise diagnostics.error(
            f"{host} a quoted `call(...)` is text, so it can't hold `${{...}}` here; "
            "drop the quotes to call the flow, or reword the text",
            range=branch.range,
        )
    entry = _free_id(branch.entry, descriptors)
    shown = _entry_text(
        entry, branch, output=output, registry=registry, diagnostics=diagnostics
    )
    raise _value_error(host, branch, entry, shown, diagnostics)


def _check_call(
    branch: _Branch,
    host: str,
    *,
    descriptors: Mapping[str, Any],
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject unquoted text starting with `call(` that is not one well-formed directive.

    Args:
        branch (`_Branch`): The branch; its value is unquoted text starting with `call(`.
        host (`str`): Message prefix naming the branch, such as `"case node 'r' then:"`.
        descriptors (`Mapping[str, Any]`): The flow's descriptors; a suggested entry id
            must not name one.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Raises:
        `LoadError`: The directive passes a literal secret, has an unreadable argument, or
            is not exactly one call. Unless the text holds a span or a secret keyword, or
            only one argument is wrong, the message shows the value entry to write for the
            text.
    """
    value = branch.value
    try:
        desugar_call_directives(value, lambda: "__check")
    except SecretArgumentError as exc:
        raise diagnostics.error(exc.message_for(host), range=branch.range) from exc
    except ArgumentValueError as exc:
        # The target is a single call; only one of its arguments is wrong.
        raise diagnostics.error(f"{host} {exc}", range=branch.range) from exc
    except ExpressionError as exc:
        # A value entry can't hold quoted call text with a span, and text with a secret
        # keyword must not be echoed, so neither gets the text's rewrite.
        hint = ""
        if "${" not in value and not holds_secret_keyword(value):
            entry = _free_id(branch.entry, descriptors)
            hint = (
                ". For the text, add a value entry and name it: "
                f"`{entry}: !str {_scalar_text(value.strip(), flow=False)}`"
            )
        raise diagnostics.error(
            f"{host} a branch target must be a node id or a single call: {exc}{hint}",
            range=branch.range,
        ) from exc


def _unknown_word_error(
    host: str,
    branch: _Branch,
    *,
    descriptors: Mapping[str, Any],
    authored: list[str],
    diagnostics: LoadDiagnostics,
) -> LoadError:
    """Build the error for an unquoted word that names no node.

    Args:
        host (`str`): Message prefix naming the branch.
        branch (`_Branch`): The branch; its value is unquoted text without a span.
        descriptors (`Mapping[str, Any]`): The flow's descriptors; a suggested entry id
            must not name one.
        authored (`list[str]`): Authored node ids offered as close matches.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Returns:
        `LoadError`: Located at the branch; it names the nearest node when one is close
        and shows the `!str` value entry for the text, named `<word>_text` when that is
        an identifier, else after the branch.
    """
    word = branch.value
    matches = difflib.get_close_matches(word, authored, n=1, cutoff=_CLOSE_MATCH_CUTOFF)
    suggestion = f" Did you mean `{matches[0]}`?" if matches else ""
    base = f"{word}_text" if is_identifier(f"{word}_text") else branch.entry
    entry = _free_id(base, descriptors)
    return diagnostics.error(
        f"{host} `{word}` names no node.{suggestion} For the text, add a value entry and "
        f"name it: `{entry}: !str {_scalar_text(word, flow=False)}`",
        range=branch.range,
    )


def _value_error(
    host: str, branch: _Branch, entry: str, shown: str, diagnostics: LoadDiagnostics
) -> LoadError:
    """Build the error for a value branch, showing its value entry and the branch.

    Args:
        host (`str`): Message prefix naming the branch.
        branch (`_Branch`): The branch.
        entry (`str`): The suggested value entry id.
        shown (`str`): The value entry as the message shows it.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Returns:
        `LoadError`: Located at the branch.
    """
    return diagnostics.error(
        f"{host} a branch names a node or calls a flow; put the value in its own entry in "
        f"`nodes:` and name it: {shown} and `{branch.label} {entry}`",
        range=branch.range,
    )


def _entry_text(
    entry: str,
    branch: _Branch,
    *,
    output: Optional[_CaseOutput],
    registry: TypeRegistry,
    diagnostics: LoadDiagnostics,
) -> str:
    """Spell the value entry a value branch becomes, for its error message.

    Args:
        entry (`str`): The entry id.
        branch (`_Branch`): The branch whose value the entry holds.
        output (`Optional[_CaseOutput]`): The case's former `output:`; its type tags the
            entry when a tag can spell it.
        registry (`TypeRegistry`): The flow's typedefs.
        diagnostics (`LoadDiagnostics`): Source ownership, for a scalar's authored text.

    Returns:
        `str`: Such as `` `route_then_0: !str approve` ``. A value past
        `_SHOWN_LIMIT` characters is cut with `…` and followed by `(write it as a block)`.
    """
    authored = _value_text(branch, diagnostics)
    if output is not None and (output.type is not None or output.text is not None):
        type_text = output.text
        if type_text is None and output.type is not None:
            type_text = canonical_type_label(output.type)
        text, note = authored, None
        if output.type is not None:
            # Spelled to fit the type where that is cheap: `3` for `str` becomes `"3"`.
            text, note = suggested_literal(text, output.type)
        shown = suggest_tag(
            entry, output.type, _cut(text), registry=registry, type_text=type_text
        )
        # Advice to name the type in `typedefs:` instead means no tag spells it.
        if shown.startswith("`"):
            return _with_notes(shown, text, note)
    shown = suggest_tag(
        entry, _guessed_type(branch.value), _cut(authored), registry=registry
    )
    return _with_notes(shown, authored, None)


def _with_notes(shown: str, text: str, note: Optional[str]) -> str:
    """Append to a shown entry the note on its literal and the advice for a cut value."""
    if note is not None:
        shown += f" ({note})"
    if len(text) > _SHOWN_LIMIT:
        shown += " (write it as a block)"
    return shown


def _cut(text: str) -> str:
    """Cut text past `_SHOWN_LIMIT` characters, ending it with `…`."""
    return text if len(text) <= _SHOWN_LIMIT else text[:_SHOWN_LIMIT] + "…"


def _guessed_type(value: Any) -> Type:
    """Return the tag type a value suggests: its YAML type, `object` for a mapping."""
    typ = example_type(value)
    if typ is None:
        typ = example_type({})
    assert typ is not None
    return typ


def _value_text(branch: _Branch, diagnostics: LoadDiagnostics) -> str:
    """Spell a branch value as YAML to follow a tag on one line.

    Text is plain when plain YAML reads it back unchanged, else quoted; a number keeps
    its authored text; a date is quoted, since a tag reads only a quoted date; a mapping
    or list is written in flow style.
    """
    value = branch.value
    if isinstance(value, str):
        return _scalar_text(value, flow=False)
    if isinstance(value, (dict, list)):
        return _flow_text(value, active=set())
    authored = source_text(diagnostics, branch.range)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return yaml_quoted(authored or value.isoformat())
    if isinstance(value, (int, float)) and not isinstance(value, bool) and authored:
        return authored
    return _scalar_text(value, flow=True)


def _flow_text(value: Any, *, active: set[int]) -> str:
    """Spell a mapping, list, or scalar in YAML flow style.

    Args:
        value (`Any`): The value or a part of one.
        active (`set[int]`): `id()`s of the containers enclosing `value`, so a YAML alias
            that contains itself is shown as `…` instead of recursing forever.

    Returns:
        `str`: Flow-style YAML, such as `{reply: "${input.text}", exited: true}`.
    """
    if not isinstance(value, (dict, list)):
        return _scalar_text(value, flow=True)
    if id(value) in active:
        return "…"
    active.add(id(value))
    if isinstance(value, dict):
        text = (
            "{"
            + ", ".join(
                f"{_scalar_text(key, flow=True)}: {_flow_text(item, active=active)}"
                for key, item in value.items()
            )
            + "}"
        )
    else:
        text = "[" + ", ".join(_flow_text(item, active=active) for item in value) + "]"
    active.discard(id(value))
    return text


def _scalar_text(value: Any, *, flow: bool) -> str:
    """Spell a scalar as YAML that reads back as the same value.

    Args:
        value (`Any`): The scalar.
        flow (`bool`): Whether it sits inside a flow mapping or list, where `,`, `{`,
            `}`, `[`, `]`, and `:` would end or nest it, so text holding them is quoted.

    Returns:
        `str`: `null`, `true`/`false`, a number, a quoted date, or text: plain when
        plain YAML reads it back unchanged and it is not a `call(...)`, else quoted.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return yaml_quoted(value.isoformat())
    text = str(value)
    if CALL_HEAD.match(text.strip()) or (flow and _FLOW_SPECIAL.intersection(text)):
        return yaml_quoted(text)
    try:
        plain = parse_yaml_literal(text) == text
    except (SourceSyntaxError, ValueError):
        plain = False
    return text if plain else yaml_quoted(text)


def _free_id(base: str, descriptors: Mapping[str, Any]) -> str:
    """Return `base`, or `base_2`, `base_3`, ... when it already names a node."""
    if base not in descriptors:
        return base
    suffix = 2
    while f"{base}_{suffix}" in descriptors:
        suffix += 1
    return f"{base}_{suffix}"


def _read_output(
    desc: CaseDescriptor, output_at: OutputReader | None, registry: TypeRegistry
) -> Optional[_CaseOutput]:
    """Read the `output:` a case declares, or return `None` when it has none.

    Args:
        desc (`CaseDescriptor`): The case.
        output_at (`OutputReader | None`): The authored `output:` reader, or `None` to
            read the descriptor's `outputs`.
        registry (`TypeRegistry`): The flow's typedefs the type resolves against.

    Returns:
        `Optional[_CaseOutput]`: The declared type, as far as it resolves, and the
        `output:` range.
    """
    cursor = output_at(desc.id) if output_at is not None else None
    if cursor is not None:
        value = cursor.value if cursor.has_value else None
        spec = value if value not in (None, "") else cursor.tag
        # The key line, which a block mapping's value range starts below.
        rng = cursor.key_range or cursor.range or desc.range
    elif desc.outputs is not None:
        spec = desc.outputs
        rng = desc.field_ranges.get(("field", "outputs")) or desc.range
    else:
        return None
    if isinstance(spec, str):
        resolved = resolve_type_text(spec, registry)
        if resolved is None:
            return _CaseOutput(type=None, text=spec.strip(), range=rng)
        return _CaseOutput(type=resolved[0], text=resolved[1], range=rng)
    if spec is None:
        return _CaseOutput(type=None, text=None, range=rng)
    # A record spelled as written (`{reply: str}`) reads better than its canonical label.
    text = _flow_text(spec, active=set())
    try:
        return _CaseOutput(type=read_type(spec, registry), text=text, range=rng)
    except (LoadError, TypeCheckError):
        return _CaseOutput(type=None, text=text, range=rng)


def _read_example(case_id: str, output: _CaseOutput, registry: TypeRegistry) -> str:
    """Return the example of reading a case's result with its former type as a tag.

    Args:
        case_id (`str`): The case.
        output (`_CaseOutput`): Its former `output:`.
        registry (`TypeRegistry`): The flow's typedefs.

    Returns:
        `str`: Such as ``, e.g. `x: !str ${route.output}` ``, or, for a type a tag can't
        spell, the advice to name it in `typedefs:`.
    """
    read = f"${{{case_id}.output}}"
    if output.type is None and output.text is None:
        return f", e.g. `x: !<Type> {read}`"
    type_text = output.text
    if type_text is None and output.type is not None:
        type_text = canonical_type_label(output.type)
    shown = suggest_tag("x", output.type, read, registry=registry, type_text=type_text)
    return f", e.g. {shown}" if shown.startswith("`") else f"; {shown}"
