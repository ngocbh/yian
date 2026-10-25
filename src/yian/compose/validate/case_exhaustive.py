"""Case exhaustiveness — the validate-phase check.

Two coverage laws for `case` nodes:

- Enum coverage, for every case: when an `on:` ref resolves to an ENUM node output or flow
  input, every enum tag must be covered by a case `when:` value OR by a present `else:`. A
  missing tag with no `else:` is a `LoadError`.
- Value-producing cases: a case that produces a value (it is read through
  `${<case>.output}`, or is a branch target of such a case) must be exhaustive. It
  has an `else:`, or it is an `on:` form whose `on:` type is non-nullable and fully listed
  by its `when:` values: an enum with every tag, or a `bool` with both `true` and `false`.
  `when:` values compare as typed values, so the text `"true"` does not cover `true`. Its
  `on:` resolves against node outputs and flow inputs, under the case's narrowing view.
  Otherwise it is a `LoadError`. A routing-only case with no match runs no branch, so only
  enum coverage applies to it.

One membership law for `on:`-form `when:` values (`check_case_when_values`): where the
`on:` type is known, every `when:` value must be a possible member of it (compared exactly:
no text conversion, a bool is not a number), even when the case has an `else:`. A value
that can never equal the `on:` value is a `LoadError` naming the value, the type, and the
fix. Two `when:` values that are equal (typed equality, `on_match_key`) are a `LoadError` at
the later one, for every `on:`-form case, since the later one never matches.

This lives in the validate phase (not in `CaseDescriptor.build`) because it needs the flow's
`producers` map — each producer node's `output_contract` — which only exists after every node is
built. `.build()` lowers the case unconditionally; the loader runs `check_case_exhaustive` in
the validate phase, alongside the other cross-node checks.

Imports flow down/peer only: `diagnostics` (SourceRange), `expr` (ref scan),
`nodes.contracts` (the output-contract sum), `typesys` (Type and the strict value check),
and the compose peers `helpers.output_contracts`, `helpers.errors`, `helpers.tags`,
`helpers.yaml_words`, `nodes.case`,
`source`, `parse.file`, `validate.narrowing`, and `validate.typed_bindings`.
Nothing in the engine imports this back.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Optional

from yian.compose.helpers.output_contracts import (
    assert_exact_contracts,
    resolve_output_binding_contract,
)
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError, expr_refs_of
from yian.nodes.contracts import OutputContract, ValueContract
from yian.typesys import TypeCheckError, build_value_as
from yian.typesys.values import Type, ValueKind, canonical_type_label
from yian.compose.helpers.errors import LoadDiagnostics, source_text
from yian.compose.helpers.tags import value_fits, yaml_quoted
from yian.compose.helpers.yaml_words import former_boolean_fix
from yian.compose.nodes.case import on_match_key, on_match_value
from yian.compose.parse.file import CaseDescriptor, NodeDescriptor
from yian.compose.source import SourceSyntaxError, parse_yaml_literal
from yian.compose.validate.narrowing import Narrowing
from yian.compose.validate.typed_bindings import shown_literal


def check_case_exhaustive(
    descriptors,
    producers: dict[str, OutputContract],
    *,
    narrowing: Narrowing | None = None,
    value_cases: frozenset[str] = frozenset(),
    diagnostics: LoadDiagnostics | None = None,
):
    """Check every case's enum coverage, and that each value-producing case is exhaustive.

    A known field projected from either arm of a partial record can drive enum coverage.
    With `narrowing`, each case resolves its `on:` ref against its own view of the
    producers and flow inputs, so a case inside a branch covers only the tags that can
    reach it, and enum coverage applies to an `on:` over a flow input too. Errors are
    located at the case node.

    Args:
        descriptors (`dict`): The flow's node descriptors keyed by node id.
        producers (`dict[str, OutputContract]`): Producer node id -> its exact or partial
            output contract.
        narrowing (`Narrowing | None`): Each node's narrowed producer and flow-input
            views. Without it, a case sees `producers` and no flow-input contracts.
        value_cases (`frozenset[str]`): The ids of the value-producing cases, which must
            have an `else:` or list every tag of a non-nullable enum `on:` type.
        diagnostics (`LoadDiagnostics | None`): Source ownership for located errors.

    Raises:
        `LoadError`: A case leaves an enum tag uncovered with no `else:`, or a
            value-producing case is not exhaustive.
        `TypeError`: A producer contract is not a `ValueContract` or
            `PartialRecordContract`.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    assert_exact_contracts(producers)
    for desc in descriptors.values():
        if not isinstance(desc, CaseDescriptor):
            continue
        case_producers, case_inputs = (
            narrowing.for_node(desc.id) if narrowing is not None else (producers, {})
        )
        if desc.id in value_cases:
            _check_value_exhaustive(desc, case_producers, case_inputs, diagnostics)
        _check_exhaustive(desc, _on_covered(desc), case_producers, case_inputs, diagnostics)


def check_case_when_values(
    descriptors: dict[str, NodeDescriptor],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    narrowing: Narrowing | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Reject an `on:`-form `when:` value that repeats an earlier one or can never equal
    the `on:` value.

    Two `when:` values are duplicates when they match the same `on:` values (typed
    equality: `1` and `1.0` are equal, `1`, `true`, and `"1"` are not); the later one never
    matches. A duplicate is reported before any membership error in the same case, so a
    repeated value is reported as such rather than as a non-member whose fix would repeat
    it.

    A `when:` value is a member of the `on:` type when the type accepts it as is: text is
    never converted, a bool is not a number, an int fits a float, and `Literal` tags and
    numeric bounds apply. The `on:` ref resolves against the unnarrowed producers and flow
    inputs first, then the case's narrowed view, so a field only a branch's members have is
    checked too. A case whose `on:` type is unknown or `object` gets no membership check
    (duplicates are still rejected).

    Args:
        descriptors (`dict[str, NodeDescriptor]`):
            The level's node id -> descriptor. Non-CASE and searched CASE descriptors are
            ignored.
        producers (`dict[str, OutputContract]`):
            Producer node id -> output contract.
        flow_inputs (`dict[str, ValueContract]`):
            Flow input name -> contract.
        narrowing (`Narrowing | None`, *optional*, defaults to `None`):
            Per-node narrowed views, tried when the unnarrowed `on:` type is unknown.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Source ownership for located errors and the authored `when:` text.

    Raises:
        `LoadError`:
            Two `when:` values are equal (typed equality); located at the later one. Or a
            `when:` value can never be a member of its case's `on:` type; located at the
            first such value in source order.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for desc in descriptors.values():
        if not isinstance(desc, CaseDescriptor) or desc.on is None:
            continue
        _check_duplicate_whens(desc, diagnostics)
        typ = _resolve_on_type(desc.on, producers, flow_inputs)
        if typ is None and narrowing is not None:
            typ = _resolve_on_type(desc.on, *narrowing.for_node(desc.id))
        if typ is None or typ.kind is ValueKind.OBJECT:
            continue
        for index, case in enumerate(desc.cases):
            when_range = _when_range(desc, index)
            message = _when_value_error(
                desc.on, case.get("when"), typ, source_text(diagnostics, when_range)
            )
            if message is not None:
                raise diagnostics.error(
                    f"case node {desc.id!r}: {message}", range=when_range or desc.range
                )


def _check_duplicate_whens(desc: CaseDescriptor, diagnostics: LoadDiagnostics) -> None:
    """Reject an `on:`-form case's `when:` value that equals an earlier one.

    Args:
        desc (`CaseDescriptor`):
            The `on:`-form case.
        diagnostics (`LoadDiagnostics`):
            Source ownership for located errors and the authored `when:` text.

    Raises:
        `LoadError`:
            Two `when:` values match the same `on:` values (`on_match_key`). The error is
            located at the later one, the first such repeat in source order.
    """
    seen: dict[tuple[str, Any], int] = {}
    for index, case in enumerate(desc.cases):
        value = case.get("when")
        if type(value) not in _VALUE_PHRASES:
            continue
        key = on_match_key(value)
        if key not in seen:
            seen[key] = index
            continue
        first = seen[key]
        when_range = _when_range(desc, index)
        first_range = _when_range(desc, first)
        message = _duplicate_when_message(
            value,
            source_text(diagnostics, when_range),
            desc.cases[first].get("when"),
            source_text(diagnostics, first_range),
            first_range=first_range,
            first_index=first,
        )
        raise diagnostics.error(f"case node {desc.id!r}: {message}", range=when_range or desc.range)


def _when_range(desc: CaseDescriptor, index: int) -> SourceRange | None:
    """Return the source range of a case's `index`-th `when:` value, when known."""
    return desc.when_ranges[index] if index < len(desc.when_ranges) else None


def _duplicate_when_message(
    value: Any,
    src: str | None,
    first_value: Any,
    first_src: str | None,
    *,
    first_range: SourceRange | None,
    first_index: int,
) -> str:
    """Return the message for a `when:` value that repeats an earlier one.

    Args:
        value (`Any`):
            The later, repeated `when:` value.
        src (`str | None`):
            Its authored source text, when known.
        first_value (`Any`):
            The earlier `when:` value it equals.
        first_src (`str | None`):
            The earlier value's authored source text, when known.
        first_range (`SourceRange | None`):
            The earlier value's source range, when known.
        first_index (`int`):
            The earlier value's index in `cases`, shown when its range is unknown.

    Returns:
        `str`:
            The message, without the `case node` prefix.
    """
    shown = _authored_or_spelling(value, src)
    first_shown = _authored_or_spelling(first_value, first_src)
    where = (
        f"first at line {first_range.start_line}"
        if first_range is not None
        else f"first at cases[{first_index}]"
    )
    return (
        f"`when: {shown}` duplicates `when: {first_shown}` ({where}), "
        "so it never matches; remove one"
    )


def _authored_or_spelling(value: Any, src: str | None) -> str:
    """Return a `when:` value's authored text, or its spelling when that text is unknown or
    starts with an anchor, alias, or tag (which is not part of the value's spelling)."""
    if src is not None and not src.startswith(("&", "*", "!")):
        return src
    return _spelling(value)


# The scalar kind of each `when:` value type, for "this value has the right kind but is
# still not a member" (an out-of-bounds number).
_VALUE_KINDS: dict[type, tuple[ValueKind, ...]] = {
    # Text is how a date or datetime `on:` value is written, so wrong text is a bad value.
    str: (ValueKind.STRING, ValueKind.DATE, ValueKind.DATETIME),
    bool: (ValueKind.BOOLEAN,),
    int: (ValueKind.INTEGER, ValueKind.NUMBER),
    float: (ValueKind.NUMBER,),
    datetime.date: (ValueKind.DATE,),
}

_VALUE_PHRASES: dict[type, str] = {
    str: "text",
    bool: "a bool",
    int: "an int",
    float: "a float",
    datetime.date: "a date",
}


def _when_value_error(on: str, value: Any, typ: Type, src: str | None) -> str | None:
    """Return why one `when:` value can never equal an `on:` value of type `typ`.

    Args:
        on (`str`):
            The case's `on:` template, as authored.
        value (`Any`):
            The authored `when:` value.
        typ (`Type`):
            The resolved `on:` type.
        src (`str | None`):
            The authored source text of the `when:` scalar, when known.

    Returns:
        `str | None`:
            The message (without the `case node` prefix), or `None` when the value is a
            member or is not a matchable scalar (the build already rejects those).
    """
    match = on_match_value(value)
    if type(value) not in _VALUE_PHRASES:
        return None
    try:
        build_value_as(typ, match)
        return None
    except TypeCheckError as exc:
        error = exc
    label = canonical_type_label(typ)
    if isinstance(value, str) and typ.tags is not None:
        tags = ", ".join(_text_shown(tag) for tag in sorted(typ.tags))
        shown = src or _text_shown(value)
        return f"`when: {shown}` is not a tag of `on: {on}` (`{label}`); use one of: {tags}"
    shown = src or _spelling(value)
    if typ.kind in _VALUE_KINDS[type(value)]:
        return f"`when: {shown}` is not a value of `on: {on}` (`{label}`): {error}"
    return (
        f"`when: {shown}` is {_VALUE_PHRASES[type(value)]} but `on: {on}` is `{label}`"
        + _when_fix(value, typ, src)
    )


def _when_fix(value: Any, typ: Type, src: str | None) -> str:
    """Return the `; <fix>` suffix for a `when:` value of the wrong kind, or `""`.

    Args:
        value (`Any`):
            The authored `when:` value.
        typ (`Type`):
            The resolved `on:` type.
        src (`str | None`):
            The authored source text of the `when:` scalar, when known.

    Returns:
        `str`:
            A quote, write-as-an-int, write-the-boolean, or drop-the-quotes hint, or `""`
            when none applies.
    """
    if not isinstance(value, str):
        # An anchor, alias, or tag is not part of the value's spelling, so quoting the
        # source would change the text; the value's own spelling is quoted instead.
        written = _authored_or_spelling(value, src)
        if value_fits(typ, written):
            return f"; quote it: `when: {yaml_quoted(written)}`"
        if isinstance(value, float) and value.is_integer() and value_fits(typ, int(value)):
            return f"; write it as an int: `when: {int(value)}`"
        return ""
    former = former_boolean_fix(value, typ)
    if former is not None:
        return f"; {former[1]}"
    if src is not None and not src.startswith(("'", '"')):
        # Text written without quotes (`!!str 1`) has no quotes to drop.
        return ""
    text = value.strip()
    reading = _yaml_reading(text)
    if reading is not None and value_fits(typ, reading):
        return f"; drop the quotes: `when: {text}`"
    return ""


def _yaml_reading(text: str) -> bool | int | float | None:
    """Return the bool or finite number `text` reads as unquoted, else `None`.

    Only the values `on:` matching lowers as numbers or bools count, so a hint to drop the
    quotes never suggests text that YAML still reads as text, `null`, a date, or a
    non-finite number.
    """
    try:
        reading = parse_yaml_literal(text)
    except SourceSyntaxError:
        return None
    if isinstance(reading, (bool, int)):
        return reading
    if isinstance(reading, float) and math.isfinite(reading):
        return reading
    return None


def _text_shown(text: str) -> str:
    """Return `text` bare when YAML reads it back as the same text, else YAML-quoted."""
    try:
        reads_back = parse_yaml_literal(text) == text
    except SourceSyntaxError:
        reads_back = False
    return text if reads_back else yaml_quoted(text)


def _spelling(value: Any) -> str:
    """Return a `when:` value as YAML would spell it, for when its source text is unknown."""
    return _text_shown(value) if isinstance(value, str) else shown_literal(value)


def _check_value_exhaustive(
    desc: CaseDescriptor,
    producers: dict[str, OutputContract],
    inputs: dict[str, ValueContract],
    diagnostics: LoadDiagnostics,
) -> None:
    """Require a value-producing case to have an `else:` or to list every enum tag.

    Args:
        desc (`CaseDescriptor`): The value-producing case.
        producers (`dict[str, OutputContract]`): The producer contracts the case sees.
        inputs (`dict[str, ValueContract]`): The flow-input contracts the case sees.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Raises:
        `LoadError`: The case has no `else:` and its `on:` type is unknown, neither an enum
            nor a `bool`, nullable, or has a member no `when:` lists (or it is a searched
            case).
    """
    if case_is_exhaustive(desc, producers, inputs):
        return
    raise diagnostics.error(
        f"case node {desc.id!r} produces a value, so it must be exhaustive: "
        f"add an `else:` ({_not_exhaustive_reason(desc, producers, inputs)})",
        range=desc.range,
    )


def _not_exhaustive_reason(
    desc: CaseDescriptor,
    producers: dict[str, OutputContract],
    inputs: dict[str, ValueContract],
) -> str:
    """Return why a case that `case_is_exhaustive` rejected may match no branch."""
    if desc.on is None:
        return "a searched case needs an else:"
    typ = _resolve_on_type(desc.on, producers, inputs)
    if typ is None:
        return f"the type of on: {desc.on} is not known"
    covered = _on_covered(desc)
    if typ.kind is ValueKind.BOOLEAN and typ.tags is None:
        if typ.nullable:
            return f"{desc.on} may be null"
        missing = [
            name
            for name, value in (("true", True), ("false", False))
            if not any(v is value for v in covered)
        ]
        return f"bool value(s) [{', '.join(missing)}] not covered"
    if typ.tags is None:
        return f"on: {desc.on} is {canonical_type_label(typ)}, not a Literal or bool"
    if typ.nullable:
        return f"{desc.on} may be null"
    return f"enum tag(s) {sorted(typ.tags - _covered_tags(covered))} not covered"


def case_is_exhaustive(
    desc: CaseDescriptor,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract] | None = None,
) -> bool:
    """Return whether a `case` always takes a branch when it runs.

    Args:
        desc (`CaseDescriptor`):
            The `case` descriptor.
        producers (`dict[str, OutputContract]`):
            Producer node id -> output contract, as the case sees them.
        flow_inputs (`dict[str, ValueContract] | None`, *optional*, defaults to `None`):
            Flow input name -> contract, as the case sees them. Without them an `on:` over
            a flow input is never exhaustive.

    Returns:
        `bool`:
            True when the case has an `else:`, or it is the `on:` form over a non-nullable
            `Literal` or `bool` node output or flow input whose every member is a `when:`
            value. A searched case without `else:` is never exhaustive.
    """
    if desc.else_ is not None:
        return True
    if desc.on is None:
        return False
    typ = _resolve_on_type(desc.on, producers, flow_inputs or {})
    return typ is not None and _covers(typ, _on_covered(desc))


def _covers(typ: Type, covered: list[Any]) -> bool:
    """Return whether the typed `when:` values `covered` list every member of `typ`.

    Args:
        typ (`Type`):
            The resolved `on:` type.
        covered (`list[Any]`):
            The case's typed `when:` match values.

    Returns:
        `bool`:
            True when `typ` is non-nullable and is a `Literal` whose every tag is covered,
            or a `bool` with both `true` and `false` covered.
    """
    if typ.nullable:
        return False
    if typ.tags is not None:
        return typ.tags <= _covered_tags(covered)
    return (
        typ.kind is ValueKind.BOOLEAN
        and any(v is True for v in covered)
        and any(v is False for v in covered)
    )


def exhaustive_cases(
    descriptors: dict[str, NodeDescriptor],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract] | None = None,
    *,
    narrowing: Narrowing | None = None,
) -> set[str]:
    """Return the ids of the CASE descriptors that always take a branch when they run.

    Args:
        descriptors (`dict[str, NodeDescriptor]`):
            The level's node id -> descriptor. Non-CASE descriptors are ignored.
        producers (`dict[str, OutputContract]`):
            Producer node id -> output contract.
        flow_inputs (`dict[str, ValueContract] | None`, *optional*, defaults to `None`):
            Flow input name -> contract.
        narrowing (`Narrowing | None`, *optional*, defaults to `None`):
            Per-node narrowed views; each CASE resolves its `on:` against its own view.

    Returns:
        `set[str]`:
            The exhaustive CASE ids (see `case_is_exhaustive`).
    """
    exhaustive = set()
    for desc in descriptors.values():
        if not isinstance(desc, CaseDescriptor):
            continue
        views = narrowing.for_node(desc.id) if narrowing is not None else (producers, flow_inputs)
        if case_is_exhaustive(desc, *views):
            exhaustive.add(desc.id)
    return exhaustive


def _on_covered(desc: CaseDescriptor) -> list[Any]:
    """Return the typed match values of an `on:`-form case's `when:`s (empty when searched).

    A list, not a set: a set would merge `1` and `True`, which never match each other.
    """
    if desc.on is None:
        return []
    return [on_match_value(c["when"]) for c in desc.cases if c.get("when") is not None]


def _covered_tags(covered: list[Any]) -> set[str]:
    """Return the text values among typed `when:` values, the only ones a tag can equal."""
    return {v for v in covered if isinstance(v, str)}


def _resolve_on_type(
    on_ref: str,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract] | None = None,
) -> Optional[Type]:
    """Return the `Type` an `on:` ref resolves to, else None.

    Resolves the binding through the shared output-contract projector, so an exact
    contract or a known field of a partial record can supply the enum Type.

    Args:
        on_ref (`str`):
            The case's `on:` template.
        producers (`dict[str, OutputContract]`):
            Producer node id -> output contract.
        flow_inputs (`dict[str, ValueContract] | None`, *optional*, defaults to `None`):
            Flow input name -> contract. With them an `${input.<name>[.<field>…]}` ref
            resolves too; without them only `${<id>.output[.<field>…]}` does.

    Returns:
        `Optional[Type]`:
            The resolved type, or ``None`` for an unsupported head, an opaque result, or a
            non-record step (lenient: no exhaustiveness check).
    """
    try:
        refs = expr_refs_of(on_ref)
    except ExpressionError:
        return None
    if len(refs) != 1:
        return None
    parts = refs[0].split(".")
    node_output = len(parts) >= 2 and parts[1] == "output"
    flow_input = flow_inputs is not None and len(parts) >= 2 and parts[0] == "input"
    if not (node_output or flow_input):
        return None
    contract = resolve_output_binding_contract(on_ref, producers, flow_inputs or {})
    return contract.type if type(contract) is ValueContract else None


def _check_exhaustive(
    desc: CaseDescriptor,
    covered: list[Any],
    producers: dict[str, OutputContract],
    inputs: dict[str, ValueContract],
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Enum-exhaustiveness for the `on:` form.

    Resolves the `on:` ref to a `Type`; if it carries `.tags` (an enum node output or flow
    input), every tag must be a text `when:` match value OR be covered by a present
    `else:`. A non-enum or unresolved `on:` is lenient (no check).

    Args:
        desc (`CaseDescriptor`):
            The case descriptor.
        covered (`list[Any]`):
            The case's typed `when:` match values.
        producers (`dict[str, OutputContract]`):
            The producer contracts the case sees.
        inputs (`dict[str, ValueContract]`):
            The flow-input contracts the case sees.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Source ownership for located errors.

    Raises:
        `LoadError`:
            An enum tag is not covered and the case has no `else:`; located at the case.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    if desc.on is None or desc.else_ is not None:
        return  # searched form, or an else: that satisfies coverage
    typ = _resolve_on_type(desc.on, producers, inputs)
    if typ is None or typ.tags is None:
        return  # not a checked enum -> lenient
    missing = sorted(typ.tags - _covered_tags(covered))
    if missing:
        raise diagnostics.error(
            f"case node {desc.id!r} on {desc.on}: non-exhaustive — enum tag(s) "
            f"{missing} not covered by a case `when:` or an `else:`",
            range=desc.range,
        )
