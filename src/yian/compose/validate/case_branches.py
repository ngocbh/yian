"""Graph-stage: a `case` branch names a node or calls a flow, and a `case` has no `output:`.

A `case` only routes. Each `then:`/`else:` is classified by its YAML tag and scalar style:

- an unquoted string starting with `call(` calls a flow. A whole-value directive is left
  for `desugar_case_call_targets`; any other text starting with `call(` is a load error
  naming the problem. Any other untagged unquoted string without `${` that names a node
  names it; one that names no node is a load error that suggests the nearest node and
  the value entry to write for the text;
- anything else is a value: a tagged value (even a word naming a node, as in a value
  entry), quoted text (`"..."`, `'...'`, or a block `|`/`>`), `null` or an empty
  `else:`, a number, a bool, a date, a mapping, a list, and unquoted text holding `${`.
  A value branch is a load error that shows the rewrite: a value entry in `nodes:`
  (`route_then_0: !str approve`) and the branch naming it. The entry keeps the branch's
  own tag; else it takes the case's former `output:` type when a tag can spell it and
  the value fits it (a lone `${input.x}` or `${node.output}` reference fits when its
  type does); else the reference's type, or a tag guessed from the YAML value; else a
  `!<type>` placeholder (a lone span of another type, a list of mixed items).

A value no entry can hold is its own load error instead of a rewrite: a number that is
not finite, a value that contains itself, a written-out `!object` mapping with an empty
key, a `call(...)` nested in it, a flow call or a malformed span inside its `${...}`, a
span reading the case itself, and a quoted `call(...)` holding a span (quoting a call
makes it text).

A case `output:` (`output: str`, or tagged `output: !str`) is a load error at the
`output:` that says to tag the result where it is read (`x: !str ${route.output}`).
A case with both reports its first value branch first.

Runs in `_assemble` before `desugar_inline_calls`, so the branches are still as authored,
and after `check_authored_bindings`, so a secret nested in a value branch is reported as a
secret literal before a message here could echo it.

Imports flow down/peer only: `compile.validation` (type relations), `compose.desugar`
(the `call(...)` check, value entry descriptors and the values they can't hold),
`compose.helpers` (errors, the `call(` head, positional metadata, YAML quoting, tag
spelling, type reading), `compose.parse` (descriptors, identifiers, the example type of
a value), `compose.source`, `diagnostics`, `expr`, and `typesys`. Only the loader
pipeline imports this back.
"""

from __future__ import annotations

import datetime
import difflib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from yian.compile.validation import type_relation
from yian.compose.desugar.inline_calls import (
    ArgumentValueError,
    SecretArgumentError,
    desugar_call_directives,
    holds_secret_keyword,
)
from yian.compose.desugar.value_entries import (
    EMPTY_RECORD_KEY,
    ValueEntryDescriptor,
    infers_record,
    non_finite_reason,
    non_finite_text,
)
from yian.compose.helpers.directives import CALL_HEAD
from yian.compose.helpers.errors import LoadDiagnostics, LoadError, source_text
from yian.compose.helpers.source import nth
from yian.compose.helpers.tags import (
    CLOSE_MATCH_CUTOFF,
    QUOTED_STYLES,
    has_span,
    old_form_suggestion,
    old_typed_text,
    resolve_type_text,
    suggest_tag,
    suggested_literal,
    tag_text,
    yaml_quoted,
)
from yian.compose.helpers.types import read_type
from yian.compose.parse.file import CaseDescriptor
from yian.compose.parse.nodes import (
    MIXED_LIST_NOTE,
    contains_itself,
    example_type,
    is_identifier,
)
from yian.compose.source import SourceCursor, SourceSyntaxError, parse_yaml_literal
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError
from yian.expr.grammar import whole_span_interior
from yian.expr.template import expr_refs_of, flow_call_callees_in_spans
from yian.typesys import BUILTIN_TYPE_NAMES, TypeCheckError, build_value_as
from yian.typesys.types import TypeRegistry
from yian.typesys.values import Type

# A value longer than this is cut in a message; the author writes it as a block.
_SHOWN_LIMIT = 60
# Characters that end or nest a plain scalar inside a YAML flow mapping or list.
_FLOW_SPECIAL = frozenset(",[]{}:#&*!|>'\"%@`")
# The tag a branch's authored text starts with, such as `!int ` in `!int 3`.
_TAG_PREFIX = re.compile(r"!\S*\s*")
# A lone reference whose type is known before the build: `input.x` or `node.output`.
_REFERENCE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")
# Shown after an entry tagged `!<type>` for a lone span whose type isn't known yet.
_SPAN_TYPE_NOTE = "name its type for `<type>`"

# Returns the authored `output:` of a case, or `None` when it has none.
OutputReader = Callable[[str], Optional[SourceCursor]]


@dataclass(frozen=True)
class _Branch:
    """One authored `then:`/`else:` of a case.

    Attributes:
        label (`str`): `"then:"` or `"else:"`.
        value (`Any`): The authored value.
        style (`Optional[str]`): Its YAML scalar style; `None` (unknown) reads as plain.
        tag (`Optional[str]`): Its local YAML tag, such as `"int"`, or `None`.
        range (`Optional[SourceRange]`): The branch's range, else the case's range.
        entry_base (`str`): The value entry id a rewrite suggests before it is made
            unique among the node ids, such as `route_then_0`.
    """

    label: str
    value: Any
    style: Optional[str]
    tag: Optional[str]
    range: Optional[SourceRange]
    entry_base: str


@dataclass(frozen=True)
class _CaseOutput:
    """The `output:` a case declares, read for its migration message.

    Attributes:
        type (`Optional[Type]`): The declared type, or `None` when it does not resolve.
        text (`Optional[str]`): The type as written (a tag, a type name, or a record in
            flow style); `None` when `output:` is empty or contains itself.
        range (`Optional[SourceRange]`): The `output:` range, else the case's range.
    """

    type: Optional[Type]
    text: Optional[str]
    range: Optional[SourceRange]


def check_case_branches(
    descriptors: Mapping[str, Any],
    *,
    registry: TypeRegistry,
    input_types: Mapping[str, Type],
    output_at: OutputReader,
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject every `case` branch that is not a node or a call, and every case `output:`.

    Args:
        descriptors (`Mapping[str, Any]`): The flow's parsed descriptors, keyed by node
            id; branches are as authored.
        registry (`TypeRegistry`): The flow's typedefs, for spelling a type as a tag.
        input_types (`Mapping[str, Type]`): The flow's input types, which tag the value
            entry of a lone `${input.x}` branch.
        output_at (`OutputReader`): Returns a case's authored `output:` by node id, or
            `None` when it has none. It sees a tagged `output: !int`, which leaves the
            descriptor's `outputs` empty.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Raises:
        `LoadError`: A branch holds a value, is an unquoted word that names no node, or
            is text starting with `call(` that is not one well-formed directive; a
            branch's value can't be held by a value entry (a number that is not
            finite, a value that contains itself, a written-out `!object` mapping with
            an empty key, a nested `call(...)`, a malformed span, a flow call in a
            span, a span reading the case, a quoted `call(...)` holding a span); or a
            case declares `output:`. Cases are checked in order, each one's `then:`
            branches, its `else:`, then its `output:`.

    Example:
        A case `route` with `then: "approve"` fails with ``case node 'route' then: a
        branch names a node or calls a flow; put the value in its own entry in `nodes:`
        and name it: `route_then_0: !str approve` and `then: route_then_0` ``.
    """
    for nid, desc in descriptors.items():
        if not isinstance(desc, CaseDescriptor):
            continue
        output = _read_output(desc, output_at(desc.id), registry)
        for branch in _branches(desc):
            _check_branch(
                branch,
                case_id=nid,
                output=output,
                descriptors=descriptors,
                registry=registry,
                input_types=input_types,
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
                style=nth(desc.then_styles, index),
                tag=nth(desc.then_tags, index),
                range=nth(desc.then_ranges, index) or desc.range,
                entry_base=f"{desc.id}_then_{index}",
            )
        )
    if desc.has_else:
        branches.append(
            _Branch(
                label="else:",
                value=desc.else_,
                style=desc.else_style,
                tag=desc.else_tag,
                range=desc.field_ranges.get(("field", "else")) or desc.range,
                entry_base=f"{desc.id}_else",
            )
        )
    return branches


def _check_branch(
    branch: _Branch,
    *,
    case_id: str,
    output: Optional[_CaseOutput],
    descriptors: Mapping[str, Any],
    registry: TypeRegistry,
    input_types: Mapping[str, Type],
    diagnostics: LoadDiagnostics,
) -> None:
    """Accept a node name or a `call(...)` branch; reject anything else with its rewrite.

    Args:
        branch (`_Branch`): The branch.
        case_id (`str`): The case that owns it.
        output (`Optional[_CaseOutput]`): The case's former `output:`, which tags the value
            entry a rewrite shows when the value fits it.
        descriptors (`Mapping[str, Any]`): The flow's descriptors; a bare word must name
            one, and a suggested entry id must not.
        registry (`TypeRegistry`): The flow's typedefs.
        input_types (`Mapping[str, Type]`): The flow's input types.
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
        old = old_typed_text(value, registry) if branch.tag is None else None
        if old is not None and old[2]:
            # The old `Type = value` text: the type it names tags the entry.
            entry = _free_id(branch.entry_base, descriptors)
            shown = old_form_suggestion(
                entry, old[0], old[1], old[2], registry=registry, placeholder=None
            )
            raise _value_error(host, branch, entry, shown, diagnostics)
        # A tag makes the branch a value, as it makes a `nodes:` entry one.
        if branch.tag is None and "${" not in value:
            if value in descriptors:
                return
            raise _unknown_word_error(
                host, branch, descriptors=descriptors, diagnostics=diagnostics
            )
    elif isinstance(value, str) and CALL_HEAD.match(value.strip()) and "${" in value:
        raise diagnostics.error(
            f"{host} a quoted `call(...)` is text, so it can't hold `${{...}}` here; "
            "drop the quotes to call the flow, or reword the text",
            range=branch.range,
        )
    _reject_unholdable(
        value,
        host=host,
        branch=branch,
        case_id=case_id,
        type_names=frozenset(registry),
        diagnostics=diagnostics,
    )
    entry = _free_id(branch.entry_base, descriptors)
    shown = _entry_text(
        entry,
        branch,
        host=host,
        output=output,
        descriptors=descriptors,
        registry=registry,
        input_types=input_types,
        diagnostics=diagnostics,
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
            entry = _free_id(branch.entry_base, descriptors)
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
    diagnostics: LoadDiagnostics,
) -> LoadError:
    """Build the error for an unquoted word that names no node.

    Args:
        host (`str`): Message prefix naming the branch.
        branch (`_Branch`): The branch; its value is unquoted text without a span.
        descriptors (`Mapping[str, Any]`): The flow's descriptors, offered as close
            matches; a suggested entry id must not name one.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Returns:
        `LoadError`: Located at the branch; it names the nearest node when one is close
        and shows the `!str` value entry for the text, named `<word>_text` when that is
        an identifier, else after the branch.
    """
    word = branch.value
    matches = difflib.get_close_matches(
        word, list(descriptors), n=1, cutoff=CLOSE_MATCH_CUTOFF
    )
    suggestion = f" Did you mean `{matches[0]}`?" if matches else ""
    base = f"{word}_text" if is_identifier(f"{word}_text") else branch.entry_base
    entry = _free_id(base, descriptors)
    return diagnostics.error(
        f"{host} `{word}` names no node.{suggestion} For the text, add a value entry and "
        f"name it: `{entry}: !str {_scalar_text(word, flow=False)}`",
        range=branch.range,
    )


def _reject_unholdable(
    value: Any,
    *,
    host: str,
    branch: _Branch,
    case_id: str,
    type_names: frozenset[str],
    diagnostics: LoadDiagnostics,
    nested: bool = False,
) -> None:
    """Reject a branch value, or a part of one, that no value entry can hold.

    Args:
        value (`Any`): The branch value or a part of one.
        host (`str`): Message prefix naming the branch.
        branch (`_Branch`): The branch, for its label and range.
        case_id (`str`): The case that owns the branch.
        type_names (`frozenset[str]`): The flow's typedef names; a call to one inside
            `${...}` builds a record and is no flow call.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.
        nested (`bool`, *optional*, defaults to `False`): Whether `value` sits inside a
            mapping or list.

    Raises:
        `LoadError`: The value contains itself, a nested string starts with `call(`,
            a span is malformed, calls a flow, or reads the case, or a number is not
            finite.
    """
    if not nested and contains_itself(value):
        raise diagnostics.error(
            f"{host} a branch value can't contain itself", range=branch.range
        )
    if isinstance(value, str):
        if nested and CALL_HEAD.match(value.strip()):
            raise diagnostics.error(
                f"{host} a `call(...)` nested in a value is not supported; write the call "
                f"as the branch (`{branch.label} call(...)`) or in a node, or reword the "
                "text",
                range=branch.range,
            )
        if "${" in value:
            _reject_span(
                value,
                host=host,
                branch=branch,
                case_id=case_id,
                type_names=type_names,
                diagnostics=diagnostics,
            )
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise diagnostics.error(
            f"{host} {non_finite_reason(value)}", range=branch.range
        )
    if not isinstance(value, (dict, list)):
        return
    for item in value.values() if isinstance(value, dict) else value:
        _reject_unholdable(
            item,
            host=host,
            branch=branch,
            case_id=case_id,
            type_names=type_names,
            diagnostics=diagnostics,
            nested=True,
        )


def _reject_span(
    text: str,
    *,
    host: str,
    branch: _Branch,
    case_id: str,
    type_names: frozenset[str],
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject text whose `${...}` is malformed, calls a flow, or reads the case.

    Args:
        text (`str`): A string of the branch value holding `${`.
        host (`str`): Message prefix naming the branch.
        branch (`_Branch`): The branch, for its label and range.
        case_id (`str`): The case that owns the branch.
        type_names (`frozenset[str]`): The flow's typedef names.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Raises:
        `LoadError`: As `_reject_unholdable` says for a span.
    """
    try:
        refs = expr_refs_of(text)
        # A builtin type called like a function gets its own error at the entry.
        callees = [
            callee
            for callee in flow_call_callees_in_spans(text, type_names)
            if callee not in BUILTIN_TYPE_NAMES
        ]
    except ExpressionError as exc:
        raise diagnostics.error(f"{host} {exc}", range=branch.range) from exc
    if callees:
        raise diagnostics.error(
            f"{host} a flow call {callees[0]!r} inside `${{...}}` is not supported; call "
            f"the flow as the branch: `{branch.label} call({callees[0]}, ...)`",
            range=branch.range,
        )
    if any(ref.split(".", 1)[0] == case_id for ref in refs):
        raise diagnostics.error(
            f"{host} the value reads {case_id!r} itself", range=branch.range
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
    host: str,
    output: Optional[_CaseOutput],
    descriptors: Mapping[str, Any],
    registry: TypeRegistry,
    input_types: Mapping[str, Type],
    diagnostics: LoadDiagnostics,
) -> str:
    """Spell the value entry a value branch becomes, for its error message.

    Args:
        entry (`str`): The entry id.
        branch (`_Branch`): The branch whose value the entry holds.
        host (`str`): Message prefix naming the branch.
        output (`Optional[_CaseOutput]`): The case's former `output:`; its type tags the
            entry when a tag can spell it and the value fits it.
        descriptors (`Mapping[str, Any]`): The flow's descriptors, which type a lone
            `${node.output}`.
        registry (`TypeRegistry`): The flow's typedefs.
        input_types (`Mapping[str, Type]`): The flow's input types, which type a lone
            `${input.x}`.
        diagnostics (`LoadDiagnostics`): Source ownership, for a scalar's authored text.

    Returns:
        `str`: Such as `` `route_then_0: !str approve` ``, then any note in parentheses.
        A value past `_SHOWN_LIMIT` characters is cut with `…` and followed by
        `(write it as a block)`.

    Raises:
        `LoadError`: The entry would be a written-out `!object` over a mapping with an
            empty key.
    """
    tag, typ, text, note = _entry_tag(
        branch,
        _value_text(branch, diagnostics),
        output=output,
        descriptors=descriptors,
        registry=registry,
        input_types=input_types,
    )
    if (
        typ is not None
        and infers_record(typ, branch.value)
        and _holds_empty_key(branch.value)
    ):
        raise diagnostics.error(f"{host} {EMPTY_RECORD_KEY}", range=branch.range)
    return _shown(entry, tag, text, note)


def _entry_tag(
    branch: _Branch,
    authored: str,
    *,
    output: Optional[_CaseOutput],
    descriptors: Mapping[str, Any],
    registry: TypeRegistry,
    input_types: Mapping[str, Type],
) -> tuple[str, Optional[Type], str, Optional[str]]:
    """Choose the tag of the value entry a value branch becomes.

    Args:
        branch (`_Branch`): The branch whose value the entry holds.
        authored (`str`): The value's YAML, without the branch's own tag.
        output (`Optional[_CaseOutput]`): The case's former `output:`.
        descriptors (`Mapping[str, Any]`): The flow's descriptors.
        registry (`TypeRegistry`): The flow's typedefs.
        input_types (`Mapping[str, Type]`): The flow's input types.

    Returns:
        `tuple[str, Optional[Type], str, Optional[str]]`: The tag text (`<type>` when
        no tag is known), the type it names when known, the value's YAML after it, and
        any note to show after the entry.
    """
    if branch.tag is not None:
        resolved = resolve_type_text(branch.tag, registry)
        return branch.tag, None if resolved is None else resolved[0], authored, None
    interior = (
        whole_span_interior(branch.value) if isinstance(branch.value, str) else None
    )
    reference = (
        None
        if interior is None
        else _reference_type(
            interior,
            descriptors=descriptors,
            registry=registry,
            input_types=input_types,
        )
    )
    if output is not None:
        tag = tag_text(output.type, registry=registry, type_text=output.text)
        if output.type is not None and tag is not None:
            text, note = authored, None
            if branch.value is not None:
                # Spelled to fit the type where that is cheap: `3` for `str` is `"3"`.
                text, note = suggested_literal(text, output.type)
            if _fits(text, output.type, reference):
                return tag, output.type, text, note
    typ = reference if interior is not None else example_type(branch.value)
    tag = tag_text(typ, registry=registry)
    if tag is None:
        note = MIXED_LIST_NOTE if isinstance(branch.value, list) else _SPAN_TYPE_NOTE
        return "<type>", None, authored, note
    return tag, typ, authored, None


def _shown(entry: str, tag: str, text: str, note: Optional[str] = None) -> str:
    """Spell a value entry in backticks, then the note and the advice for a cut value."""
    cut = text if len(text) <= _SHOWN_LIMIT else text[:_SHOWN_LIMIT] + "…"
    shown = f"`{entry}: !{tag} {cut}`"
    if note is not None:
        shown += f" ({note})"
    if len(text) > _SHOWN_LIMIT:
        shown += " (write it as a block)"
    return shown


def _fits(text: str, typ: Type, reference: Optional[Type]) -> bool:
    """Report whether a tag of type `typ` reads the YAML `text`.

    Args:
        text (`str`): The value's YAML.
        typ (`Type`): The tag's type.
        reference (`Optional[Type]`): The known type of a lone reference `text` is,
            else `None`.

    Returns:
        `bool`: For a lone reference of known type, whether that type fits `typ`; for
        other text holding a span, True, since its type isn't known yet; else whether
        `typ` reads the value.
    """
    if reference is not None:
        return type_relation(reference, typ) == "fits"
    if has_span(text):
        return True
    try:
        build_value_as(typ, parse_yaml_literal(text))
    except (SourceSyntaxError, TypeCheckError, ValueError):
        return False
    return True


def _reference_type(
    interior: str,
    *,
    descriptors: Mapping[str, Any],
    registry: TypeRegistry,
    input_types: Mapping[str, Type],
) -> Optional[Type]:
    """Return the type a lone span reads when it is known before the build.

    Args:
        interior (`str`): The text inside a branch's lone `${...}`.
        descriptors (`Mapping[str, Any]`): The flow's descriptors.
        registry (`TypeRegistry`): The flow's typedefs.
        input_types (`Mapping[str, Type]`): The flow's input types.

    Returns:
        `Optional[Type]`: The type of the flow input `input.x`, or the declared output
        of the node `node.output`, reads; `None` for any other span.
    """
    found = _REFERENCE.fullmatch(interior.strip())
    if found is None:
        return None
    head, name = found.groups()
    if head == "input":
        return input_types.get(name)
    desc = descriptors.get(head)
    if name != "output" or desc is None:
        return None
    if isinstance(desc, ValueEntryDescriptor):
        return desc.tag.type
    outputs = getattr(desc, "outputs", None)
    resolved = resolve_type_text(outputs, registry) if isinstance(outputs, str) else None
    return None if resolved is None else resolved[0]


def _holds_empty_key(value: Any) -> bool:
    """Report whether a value that doesn't contain itself holds a mapping keyed `""`."""
    if isinstance(value, dict):
        return "" in value or any(_holds_empty_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_holds_empty_key(item) for item in value)
    return False


def _value_text(branch: _Branch, diagnostics: LoadDiagnostics) -> str:
    """Spell a branch value as YAML to follow a tag on one line.

    Text is plain when plain YAML reads it back unchanged, else quoted; a number keeps
    its authored text; a date is quoted, since a tag reads only a quoted date; a mapping
    or list is written in flow style.

    Args:
        branch (`_Branch`): The branch; its value holds no non-finite number and does
            not contain itself.
        diagnostics (`LoadDiagnostics`): Source ownership, for a scalar's authored text.

    Returns:
        `str`: The value's YAML, without the branch's own tag.
    """
    value = branch.value
    if isinstance(value, str):
        return _scalar_text(value, flow=False)
    if isinstance(value, (dict, list)):
        return _flow_text(value)
    authored = source_text(diagnostics, branch.range)
    if authored and branch.tag is not None:
        authored = _TAG_PREFIX.sub("", authored, count=1)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return yaml_quoted(authored or value.isoformat())
    if isinstance(value, (int, float)) and not isinstance(value, bool) and authored:
        return authored
    return _scalar_text(value, flow=True)


def _flow_text(value: Any) -> str:
    """Spell a mapping, list, or scalar that doesn't contain itself in YAML flow style.

    Args:
        value (`Any`): The value or a part of one.

    Returns:
        `str`: Flow-style YAML, such as `{reply: "${input.text}", exited: true}`.
    """
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{_scalar_text(key, flow=True)}: {_flow_text(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_flow_text(item) for item in value) + "]"
    return _scalar_text(value, flow=True)


def _scalar_text(value: Any, *, flow: bool) -> str:
    """Spell a scalar as YAML that reads back as the same value.

    Args:
        value (`Any`): The scalar.
        flow (`bool`): Whether it sits inside a flow mapping or list, where `,`, `{`,
            `}`, `[`, `]`, and `:` would end or nest it, so text holding them is quoted.

    Returns:
        `str`: `null`, `true`/`false`, a number (`.inf`, `-.inf`, `.nan` when not
        finite), a quoted date, or text: plain when plain YAML reads it back unchanged
        and it is not a `call(...)`, else quoted.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return non_finite_text(value)
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
    desc: CaseDescriptor, cursor: Optional[SourceCursor], registry: TypeRegistry
) -> Optional[_CaseOutput]:
    """Read the `output:` a case declares, or return `None` when it has none.

    Args:
        desc (`CaseDescriptor`): The case.
        cursor (`Optional[SourceCursor]`): The case's authored `output:`, or `None`.
        registry (`TypeRegistry`): The flow's typedefs the type resolves against.

    Returns:
        `Optional[_CaseOutput]`: The declared type, as far as it resolves, and the
        `output:` range. An empty `output:`, or one that contains itself, has no text.
    """
    if cursor is None:
        return None
    value = cursor.value if cursor.has_value else None
    spec = value if value not in (None, "") else cursor.tag
    # The key line, which a block mapping's value range starts below.
    rng = cursor.key_range or cursor.range or desc.range
    if spec is None or contains_itself(spec):
        return _CaseOutput(type=None, text=None, range=rng)
    if isinstance(spec, str):
        resolved = resolve_type_text(spec, registry)
        if resolved is None:
            return _CaseOutput(type=None, text=spec.strip(), range=rng)
        return _CaseOutput(type=resolved[0], text=resolved[1], range=rng)
    # A record spelled as written (`{reply: str}`) reads better than its canonical label.
    text = _flow_text(spec)
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
        `str`: Such as `` , e.g. `x: !str ${route.output}` ``, or, for a type a tag can't
        spell, the advice to name it in `typedefs:`.
    """
    read = f"${{{case_id}.output}}"
    if output.text is None:
        return f", e.g. `x: !<type> {read}`"
    shown = suggest_tag("x", output.type, read, registry=registry, type_text=output.text)
    if tag_text(output.type, registry=registry, type_text=output.text) is None:
        return f"; {shown}"
    return f", e.g. {shown}"
