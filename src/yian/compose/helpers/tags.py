"""Read a slot's type and value through its YAML tags.

A slot's type is a local YAML tag on its value (`qty: !int ${size.output.qty}`). A tag
ends at the first space and, in strict YAML 1.2, can't contain `,`, `[`, `]`, `{` or
`}`, so a type with several arguments is named in `typedefs:` and tagged by its name.
`read_tag_type` turns tag text into a `Type` and explains each way a tag goes wrong;
`suggest_tag` spells a tagged slot for error messages; `read_tagged_value` reads one
slot's authored value, with tags inside it, into a binding source or a literal, and
records each tag it reads in the load's ledger. `reject_unread_tags` then rejects
every tag no slot read (on `kind:`, a typedef, a mapping key, ...), and
`reject_tagged_output` a tag on a node's `output:` type declaration.

A tag only checks a value and declares its type: it never adds or changes a field.
"""

from __future__ import annotations

import difflib
import json
import re
import string
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from yian.compose.helpers.directives import CALL_HEAD
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.source import is_extension_key
from yian.compose.helpers.yaml_words import former_boolean_fix
from yian.compose.source import (
    SourceAccessError,
    SourceCursor,
    SourceDocument,
    SourceSyntaxError,
    parse_yaml_literal,
)
from yian.diagnostics import OriginRef, SourceRange
from yian.expr import ExpressionError, eval_binding, scan_template
from yian.expr.template import Span
from yian.nodes.contracts import (
    Binding,
    ContractOrigins,
    ValueContract,
    ValuePath,
    ValueTemplate,
)
from yian.typesys import (
    BUILTIN_TYPE_NAMES,
    TypeCheckError,
    TypeRegistry,
    build_value_as,
    type_for,
)
from yian.typesys.types import EnumExpr, format_type_expr, legacy_respelling, parse_type
from yian.typesys.values import (
    IntTooLongError,
    Type,
    ValueKind,
    canonical_type_label,
    requires_file,
)

if TYPE_CHECKING:
    from yian.compose.source.catalog import _SourceCatalogBuilder

_HEAD_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# The `difflib` ratio a name must reach to be offered as a close match.
CLOSE_MATCH_CUTOFF = 0.6
_FLOW_INDICATORS = frozenset("[]{}")
# The characters PyYAML reads into a tag, minus YAML 1.2's flow indicators, `!`, and
# the `%` that starts an escape: text made only of these reads back as one tag.
_TAG_CHARACTERS = frozenset(string.ascii_letters + string.digits + "-;/?:@&=+$_.~*'()")
_TYPEDEF_EXAMPLE = (
    "(e.g. `Decision: Union(Buy, Sell)`) and tag with its name (`!Decision`)"
)
# YAML scalar styles that make a value text, never a `call(...)` directive.
QUOTED_STYLES = frozenset({"single", "double", "literal", "folded"})
_OLD_SEPARATOR = " = "
# YAML drops the trailing space of `qty: int = `, delivering "int =".
_OLD_TRAILING_SEPARATOR = " ="


def read_tag_type(
    tag: str,
    registry: TypeRegistry,
    *,
    label: str,
    diagnostics: LoadDiagnostics,
    range: SourceRange | None = None,
) -> Type:
    """Resolve a slot's tag text into its type.

    Args:
        tag (`str`): The tag text without the leading `!`, e.g. `"list(Buy)"`.
        registry (`TypeRegistry`): The flow's typedefs the tag resolves against.
        label (`str`): The slot's name, used in error messages.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        range (`SourceRange | None`, *optional*, defaults to `None`): The tagged
            value's location, attached to any error.

    Returns:
        `Type`: The type the tag names.

    Raises:
        `LoadError`: The tag holds a bracket or brace (the message shows the
            parenthesized spelling), is cut short by a space (it ends in `,` or has an
            unclosed `(`), holds a `,`, or does not name a type (the message offers a
            close match).
    """
    if _FLOW_INDICATORS.intersection(tag):
        message = (
            f"{label!r}: a tag can't contain `[`, `]`, `{{` or `}}` (strict YAML 1.2)"
        )
        spelled = _respelled(tag, registry)
        if spelled is not None and _writable_as_tag(spelled):
            message += f"; write `!{spelled}`"
        elif spelled is not None:
            message += f"; name `{spelled}` in `typedefs:` and tag with its name"
        raise diagnostics.error(message, range=range)
    if tag.endswith(",") or tag.count("(") > tag.count(")"):
        raise diagnostics.error(
            f"{label!r}: `!{tag}` is not a complete type: a YAML tag ends at the first "
            "space and can't contain `,`; name a type with several arguments in "
            f"`typedefs:` {_TYPEDEF_EXAMPLE}",
            range=range,
        )
    if "," in tag:
        raise diagnostics.error(
            f"{label!r}: `!{tag}`: a tag can't contain `,`; name it in `typedefs:` "
            f"{_TYPEDEF_EXAMPLE}",
            range=range,
        )
    try:
        return type_for(tag, registry)
    except TypeCheckError as exc:
        # A long int's tag would repeat every digit; the range still points at it.
        shown = "the tag" if isinstance(exc.__cause__, IntTooLongError) else f"`!{tag}`"
        message = f"{label!r}: {shown} is not a type ({exc})"
        match = close_type_name(tag, registry)
        if match is not None:
            message += f". Did you mean `{match}`?"
        raise diagnostics.error(message, range=range) from None


def _respelled(text: str, registry: TypeRegistry) -> str | None:
    """Spell bracket-style type text in the parenthesized spelling, if it resolves.

    A `dict[K, V]` becomes `object`, the lenient mapping it resolves to.
    """
    try:
        spelled = legacy_respelling(text)
        type_for(spelled, registry)
    except TypeCheckError:
        return None
    return spelled


def split_typed_text(value: Any) -> tuple[str, str] | None:
    """Split text at its first ` = ` into the part before and the part after.

    A `call(...)` directive is not split, since its arguments may hold ` = `.

    Args:
        value (`Any`): The authored value.

    Returns:
        `tuple[str, str] | None`: The stripped parts before and after the separator
            (the second empty for `int =`), or `None` when `value` is not text, is a
            `call(...)`, or holds no separator.
    """
    if not isinstance(value, str) or CALL_HEAD.match(value.strip()):
        return None
    if _OLD_SEPARATOR in value:
        left, right = value.split(_OLD_SEPARATOR, 1)
    elif value.endswith(_OLD_TRAILING_SEPARATOR):
        left, right = value[: -len(_OLD_TRAILING_SEPARATOR)], ""
    else:
        return None
    return left.strip(), right.strip()


def old_typed_text(
    value: Any, registry: TypeRegistry
) -> tuple[Type, str, str] | None:
    """Recognize the old `Type = value` text, whose type is now written as a tag.

    Only text whose part before the first ` = ` names a type (in either spelling)
    is the old form; any other text holding ` = ` is plain text, and so is a
    `call(...)` directive, whose arguments may hold ` = `. The caller decides
    whether the value's YAML style makes it text.

    Args:
        value (`Any`): The authored value.
        registry (`TypeRegistry`): The flow's typedefs the left side resolves against.

    Returns:
        `tuple[Type, str, str] | None`: The type, its parenthesized spelling, and the
            text after the separator (empty for `int =`), or `None` when `value` is
            not the old form.
    """
    split = split_typed_text(value)
    if split is None:
        return None
    resolved = resolve_type_text(split[0], registry)
    if resolved is None:
        return None
    return resolved[0], resolved[1], split[1]


def resolve_type_text(text: str, registry: TypeRegistry) -> tuple[Type, str] | None:
    """Resolve type text written in either the parenthesized or the bracket spelling.

    Args:
        text (`str`): Authored type text, e.g. `"int"` or `"Optional[int]"`.
        registry (`TypeRegistry`): The flow's typedefs the text resolves against.

    Returns:
        `tuple[Type, str] | None`: The type and its parenthesized spelling, or `None`
            when the text names no type in either spelling.
    """
    text = text.strip()
    spelled = text if _resolves(text, registry) else _respelled(text, registry)
    if spelled is None:
        return None
    return type_for(spelled, registry), spelled


def spell_type_text(text: str) -> str | None:
    """Spell text that parses as a type, in either spelling, with parentheses.

    Names are not resolved, so `list[message]` gives `list(message)` even with no
    `message` typedef.

    Args:
        text (`str`): Authored type text.

    Returns:
        `str | None`: The parenthesized spelling, or `None` when the text does not
            parse as a type expression.
    """
    text = text.strip()
    try:
        return format_type_expr(parse_type(text))
    except TypeCheckError:
        pass
    try:
        return legacy_respelling(text)
    except TypeCheckError:
        return None


def suggested_literal(text: str, typ: Type) -> tuple[str, str | None]:
    """Spell a literal for a tag suggestion so that it fits `typ`, where that is cheap.

    A former YAML boolean word becomes the boolean it meant when that fits, with the
    `former_boolean_fix` note; a non-string literal whose text fits becomes quoted
    text (`30` for `str` gives `"30"`). Templates, directives, and anything else
    are kept as written.

    Args:
        text (`str`): The literal text after the type.
        typ (`Type`): The type the tag declares.

    Returns:
        `tuple[str, str | None]`: The text to show after the tag, and a note to show
            after the suggestion, if any.
    """
    if not text or has_span(text) or CALL_HEAD.match(text):
        return text, None
    try:
        parsed = parse_yaml_literal(text)
        build_value_as(typ, parsed)
    except (SourceSyntaxError, TypeCheckError, ValueError):
        pass
    else:
        return text, None
    fix = former_boolean_fix(parsed, typ)
    if fix is not None:
        return ("true" if fix[0] else "false"), fix[1]
    if not isinstance(parsed, str):
        try:
            build_value_as(typ, text)
        except (TypeCheckError, ValueError):
            return text, None
        return yaml_quoted(text), None
    return text, None


def old_form_suggestion(
    name: str,
    typ: Type | None,
    type_text: str,
    value: str,
    *,
    registry: TypeRegistry,
    placeholder: str | None,
) -> str:
    """Spell the tagged slot an old `name: T = value` form should be written as.

    Args:
        name (`str`): The slot's name.
        typ (`Type | None`): The type, or `None` when its text names no type.
        type_text (`str`): The type in the parenthesized spelling.
        value (`str`): The authored text after ` = `, or `""` for none.
        registry (`TypeRegistry`): The flow's typedefs.
        placeholder (`str | None`): The text shown when there is no value, or `None`
            to show the tag alone.

    Returns:
        `str`: The suggestion from `suggest_tag`, with a literal spelled to fit the
            type where that is cheap, and the reason when the literal changed.
    """
    note = None
    if typ is not None and value:
        value, note = suggested_literal(value, typ)
    suggestion = suggest_tag(
        name, typ, value or placeholder, registry=registry, type_text=type_text
    )
    return suggestion if note is None else f"{suggestion} ({note})"


def _resolves(text: str, registry: TypeRegistry) -> bool:
    """Whether `text` names a type."""
    try:
        type_for(text, registry)
    except TypeCheckError:
        return False
    return True


def close_type_name(text: str, registry: TypeRegistry) -> str | None:
    """Return the type name closest to the name `text` starts with, if any.

    Args:
        text (`str`): Authored type text that failed to resolve.
        registry (`TypeRegistry`): The flow's typedefs, offered with the builtins.

    Returns:
        `str | None`: A builtin or typedef name close to the leading name, or `None`
            when there is no leading name, it is already a type name (the error lies
            later in the text, e.g. `list(Strr)`), or nothing is close.
    """
    head = _HEAD_IDENTIFIER.match(text)
    names = BUILTIN_TYPE_NAMES | registry.keys()
    if head is None or head.group() in names:
        return None
    matches = difflib.get_close_matches(
        head.group(), sorted(names), n=1, cutoff=CLOSE_MATCH_CUTOFF
    )
    if matches:
        return matches[0]
    # A wrong capital (`Strr`, `INT`) costs a whole character in the ratio, so a
    # case-blind second pass finds what the first misses.
    by_lower: dict[str, str] = {}
    for name in sorted(names):
        by_lower.setdefault(name.lower(), name)
    matches = difflib.get_close_matches(
        head.group().lower(), sorted(by_lower), n=1, cutoff=CLOSE_MATCH_CUTOFF
    )
    return by_lower[matches[0]] if matches else None


def suggest_tag(
    name: str,
    typ: Type | None,
    value: str | None,
    *,
    registry: TypeRegistry | None = None,
    type_text: str | None = None,
) -> str:
    """Spell a tagged slot for an error message.

    Args:
        name (`str`): The slot's name.
        typ (`Type | None`): The slot's type, or `None` when its text names no type;
            `type_text` is then shown as written.
        value (`str | None`): The value text after the tag, or `None` for none.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs; a typedef that resolves to `typ` (or to its list element or
            non-null form) is spelled by its name.
        type_text (`str | None`, *optional*, defaults to `None`): The authored
            spelling of `typ`, used when a tag can write it and in the advice to
            name it otherwise.

    Returns:
        `str`: `` `name: !<type> value` `` in backticks, or, for a type a tag can't
            spell (an unnamed record or union, or anything needing a `,` or quotes),
            advice to name the type in `typedefs:` and tag with its name.
    """
    text = tag_text(typ, registry=registry, type_text=type_text)
    if text is None:
        if type_text is None:
            assert typ is not None, "suggest_tag needs a type or its text"
            type_text = canonical_type_label(typ)
        return f"name `{type_text}` in `typedefs:` and tag with its name"
    if value is None:
        return f"`{name}: !{text}`"
    return f"`{name}: !{text} {value}`"


def any_source_hint(
    name: str, source: Type, sink: Type, *, registry: TypeRegistry | None = None
) -> str | None:
    """Advise how to fill a slot of type `sink` from an `Any` source, when that can work.

    Load can't tell which JSON value an `Any` holds, so a tag checks it at runtime or a
    `case` narrows it. There is no advice when the source isn't `Any`, the slot is `Any`
    or `Optional(Any)` (only nullability failed), or every non-null value of the slot
    holds a file: load treats such a slot as disjoint from `Any`, although a `file` slot
    reads a file's JSON form at runtime, where load can't tell.

    Args:
        name (`str`): The slot's name.
        source (`Type`): The source's type.
        sink (`Type`): The slot's type.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, used to spell the tag by a typedef name.

    Returns:
        `str | None`: Advice to tag the slot with `sink` or to narrow the value with
            `isinstance` in a `case`, or `None` when neither can help.
    """
    if source.kind is not ValueKind.ANY or sink.kind is ValueKind.ANY or requires_file(sink):
        return None
    tag = suggest_tag(name, sink, "${...}", registry=registry)
    return (
        f"tag the slot ({tag}) to check it at runtime, or narrow it with `isinstance` "
        "in a `case`"
    )


def tag_text(
    typ: Type | None,
    *,
    registry: TypeRegistry | None = None,
    type_text: str | None = None,
) -> str | None:
    """Spell a type as the text after `!`, or return `None` when no tag can spell it.

    Args:
        typ (`Type | None`): The type, or `None` when its text names no type.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs; a typedef that resolves to `typ` (or to its list element or
            non-null form) is spelled by its name.
        type_text (`str | None`, *optional*, defaults to `None`): The authored
            spelling of `typ`, used as written when a tag can write it.

    Returns:
        `str | None`: Such as `"list(int)"` or `"Reply"`, or `None` for a type a tag
            can't spell (an unnamed record or union, or anything needing a `,` or
            quotes).
    """
    if type_text is not None and _writable_as_tag(type_text):
        return type_text
    if typ is None:
        return None
    return _tag_text(typ, registry or {})


def _tag_text(typ: Type, registry: TypeRegistry) -> str | None:
    """Spell `typ` as tag text, or return `None` when no tag can spell it.

    The type is spelled directly where a tag can; a typedef name stands in only
    where it can't (a record, a union, a multi-value `Literal`, anything needing a
    comma or quotes).
    """
    direct = _direct_tag_text(typ, registry)
    if direct is not None:
        return direct
    named = _typedef_name(replace(typ, nullable=False), registry)
    if named is None:
        return None
    text = f"Optional({named})" if typ.nullable else named
    return text if _writable_as_tag(text) else None


def _direct_tag_text(typ: Type, registry: TypeRegistry) -> str | None:
    """Spell `typ` by its structure; only a list element may fall back to a typedef name."""
    if typ.variants is not None or typ.fields is not None:
        return None
    if typ.tags is not None:
        if len(typ.tags) != 1:
            return None
        text = format_type_expr(EnumExpr(tuple(typ.tags)))
    elif typ.kind.is_list() and typ.element is not None:
        element = _tag_text(typ.element, registry)
        if element is None:
            return None
        text = f"list({element})"
    else:
        # A flat list, bounded number, or plain scalar has no part a tag can't spell
        # except its own characters, checked below.
        text = canonical_type_label(Type(kind=typ.kind, bounds=typ.bounds))
    if typ.nullable:
        text = f"Optional({text})"
    return text if _writable_as_tag(text) else None


def named_type_label(typ: Type, registry: TypeRegistry) -> str:
    """Label a type for a message, naming a record or union by its typedef.

    Args:
        typ (`Type`): The type to label.
        registry (`TypeRegistry`): The flow's typedefs.

    Returns:
        `str`: The typedef's name for a record or union one resolves to exactly
            (also under `Optional` or as a list element, as in `list(Buy)`), else
            `canonical_type_label`.
    """
    named = _named_record(typ, registry)
    return named if named is not None else canonical_type_label(typ)


def _named_record(typ: Type, registry: TypeRegistry) -> str | None:
    """Spell a record or union type (or a list of one) by its typedef, else `None`."""
    if typ.kind.is_list() and typ.element is not None:
        inner = _named_record(typ.element, registry)
        text = None if inner is None else f"list({inner})"
    elif typ.fields is not None or typ.variants is not None:
        text = _typedef_name(replace(typ, nullable=False), registry)
    else:
        text = None
    if text is None:
        return None
    return f"Optional({text})" if typ.nullable else text


def _typedef_name(typ: Type, registry: TypeRegistry) -> str | None:
    """Return the first typedef a tag can name that resolves to exactly `typ`."""
    for name in registry:
        if not _writable_as_tag(name):
            continue
        try:
            if type_for(name, registry) == typ:
                return name
        except TypeCheckError:
            continue
    return None


def _writable_as_tag(text: str) -> bool:
    """Whether `text` reads back as exactly one YAML tag when written after `!`."""
    return bool(text) and all(char in _TAG_CHARACTERS for char in text)


@dataclass(frozen=True)
class TaggedSlot:
    """One slot's authored value, read through its tags.

    Attributes:
        value (`Any`): The binding source or literal: a template string, a
            `ValueTemplate`, or a plain value; `None` when `has_value` is false.
        contract (`ValueContract | None`): The top-level tag's contract, or `None`
            for an untagged value.
        has_value (`bool`): False for `name: !T` with nothing after the tag.
    """

    value: Any
    contract: ValueContract | None
    has_value: bool


def read_tagged_value(
    cursor: SourceCursor,
    registry: TypeRegistry,
    *,
    label: str,
    site_path: tuple[str | int, ...],
    catalog_builder: "_SourceCatalogBuilder",
    diagnostics: LoadDiagnostics,
    ledger: set[int],
    literal_only: bool = False,
    nested_call_fix: str = "bind it at the top level, or reword the text",
) -> TaggedSlot:
    """Read one slot's authored value and the tags on and inside it.

    The top-level tag becomes the slot's contract. A top-level string is kept as
    written (a template, a `call(...)`, or text the binder unescapes). A tagged node
    inside a mapping or list becomes a typed `Binding` at its position, and a
    container holding one, or holding a string with a `${...}` span, becomes a
    `ValueTemplate` with raw leaf strings; any other container is a plain literal
    with its leaf strings `$$`-unescaped once. Every span-free tagged literal is
    checked against its tag here; a tag never fills in a missing field.

    Catalog sites: the top-level contract at `(*site_path, "type", "root")`, a
    nested binding at `(*site_path, "type", "path", *path)`, and a template's leaf
    string at `(*site_path, "path", *path)`.

    Args:
        cursor (`SourceCursor`): The slot's value.
        registry (`TypeRegistry`): The flow's typedefs the tags resolve against.
        label (`str`): The slot's name, used in messages and suggested spellings.
        site_path (`tuple[str | int, ...]`): Catalog site prefix for this slot.
        catalog_builder (`_SourceCatalogBuilder`): File-owned origin catalog builder.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        ledger (`set[int]`): Consumed tags; the reader adds the definition id
            (`SourceCursor.tag_definition_id`, matching `TaggedNode.definition_id`)
            of every tag it reads.
        literal_only (`bool`, *optional*, defaults to `False`): Read a literal (a
            default): strings are text, never templates, and are not unescaped;
            nested tags are checked and dropped, so the value is plain.
        nested_call_fix (`str`, *optional*, defaults to binding it at the top level):
            The fix a nested string starting with `call(` is told to make.

    Returns:
        `TaggedSlot`: The value, the top-level contract, and whether a value follows
            the tag.

    Raises:
        `LoadError`: A tag is malformed or names no type, a value is missing after a
            nested tag, a space splits a tag from its arguments, a template has a
            non-string record key, a nested string starts with `call(`, a span-free
            tagged literal does not fit its tag, or the value can't be walked (a
            recursive alias).
    """
    reader = _TaggedValueReader(
        registry,
        label=label,
        site_path=site_path,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
        ledger=ledger,
        literal_only=literal_only,
        nested_call_fix=nested_call_fix,
    )
    try:
        return reader.read(cursor)
    except SourceAccessError as exc:
        raise diagnostics.error(f"{label!r}: {exc}", range=exc.range) from exc


@dataclass
class _Walked:
    """One walked node: its tree, and what in it needs the binder."""

    tree: Any
    has_span: bool = False
    has_binding: bool = False

    @property
    def dynamic(self) -> bool:
        """Whether the node must be a `ValueTemplate` part rather than a literal."""
        return self.has_span or self.has_binding

    def absorb(self, part: _Walked) -> None:
        """Carry a child's span and binding flags up to this container."""
        self.has_span |= part.has_span
        self.has_binding |= part.has_binding


class _TaggedValueReader:
    """Walk one slot's value with its cursors, reading the tags inside it."""

    def __init__(
        self,
        registry: TypeRegistry,
        *,
        label: str,
        site_path: tuple[str | int, ...],
        catalog_builder: "_SourceCatalogBuilder",
        diagnostics: LoadDiagnostics,
        ledger: set[int],
        literal_only: bool,
        nested_call_fix: str,
    ) -> None:
        self.registry = registry
        self.label = label
        self.site_path = site_path
        self.catalog_builder = catalog_builder
        self.diagnostics = diagnostics
        self.ledger = ledger
        self.literal_only = literal_only
        self.nested_call_fix = nested_call_fix
        self.leaf_sites: list[tuple[ValuePath, SourceCursor]] = []

    def read(self, cursor: SourceCursor) -> TaggedSlot:
        """Read the slot at `cursor`; see `read_tagged_value`."""
        contract = None
        tag = cursor.tag
        if tag is not None:
            # A space-split tag (`!Optional (str)`) usually names no type on its own,
            # so the spacing is reported before the type is resolved.
            self._reject_spaced_tag(cursor, ())
            typ = self._tag_type(cursor, ())
            origin = self.catalog_builder.register(
                (*self.site_path, "type", "root"), cursor
            )
            contract = ValueContract(typ, ContractOrigins(default=origin))
            if not cursor.has_value:
                return TaggedSlot(None, contract, False)
        walked = self._walk(cursor, cursor.value, ())
        if isinstance(walked.tree, str) or not walked.dynamic or self.literal_only:
            literal = walked.tree
            if not self._is_template_text(literal, cursor):
                literal = self._literal(literal)
                if contract is not None:
                    self._check(cursor, (), contract.type, literal)
            # A top-level string stays as written: the binder unescapes it.
            return TaggedSlot(
                walked.tree if isinstance(walked.tree, str) else literal,
                contract,
                True,
            )

        if contract is not None and not walked.has_span:
            self._check(cursor, (), contract.type, self._literal(walked.tree))
        origins: list[tuple[ValuePath, OriginRef]] = []
        for path, leaf in self.leaf_sites:
            origin = self.catalog_builder.register((*self.site_path, "path", *path), leaf)
            if origin is not None:
                origins.append((path, origin))
        return TaggedSlot(ValueTemplate(walked.tree, tuple(origins)), contract, True)

    def _walk(self, cursor: SourceCursor, value: Any, path: ValuePath) -> _Walked:
        """Walk one node, reading a tag on it unless it is the slot itself."""
        if path and cursor.tag is not None:
            return self._walk_tagged(cursor, value, path)
        return self._walk_value(cursor, value, path)

    def _walk_tagged(self, cursor: SourceCursor, value: Any, path: ValuePath) -> _Walked:
        """Read a nested tagged node into a typed `Binding`, or check and fold it."""
        self._reject_spaced_tag(cursor, path)
        typ = self._tag_type(cursor, path)
        if not cursor.has_value:
            raise self.diagnostics.error(
                f"{path_label(self.label, path)!r}: a nested tag needs a value: "
                f"`!{cursor.tag} <value>`",
                range=cursor.range,
            )
        inner = self._walk_value(cursor, value, path)
        if not inner.has_span:
            self._check(cursor, path, typ, self._literal(inner.tree))
        if self.literal_only:
            return inner
        origin = self.catalog_builder.register(
            (*self.site_path, "type", "path", *path), cursor
        )
        binding = Binding(
            source=inner.tree,
            origin=origin,
            contract=ValueContract(typ, ContractOrigins(default=origin)),
        )
        return _Walked(binding, has_span=inner.has_span, has_binding=True)

    def _walk_value(self, cursor: SourceCursor, value: Any, path: ValuePath) -> _Walked:
        """Walk one node's own value, reading the tags on its children."""
        if isinstance(value, str):
            if self.literal_only or not path:
                return _Walked(value)
            if CALL_HEAD.match(value.strip()):
                raise self.diagnostics.error(
                    f"{path_label(self.label, path)!r}: a `call(...)` inside a value "
                    f"isn't supported; {self.nested_call_fix}",
                    range=cursor.range,
                )
            self.leaf_sites.append((path, cursor))
            return _Walked(value, has_span=has_span(value))
        if type(value) is dict:
            tree: dict[Any, Any] = {}
            walked = _Walked(tree)
            for key, item in value.items():
                child = cursor.child(key)
                part = self._walk(child, item, (*path, key))
                tree[key] = part.tree
                walked.absorb(part)
            return walked
        if type(value) is list:
            items: list[Any] = []
            walked = _Walked(items)
            for index, item in enumerate(value):
                part = self._walk(cursor.child(index), item, (*path, index))
                items.append(part.tree)
                walked.absorb(part)
            return walked
        return _Walked(value)

    def _tag_type(self, cursor: SourceCursor, path: ValuePath) -> Type:
        """Record the tag on `cursor` as consumed and return the type it names."""
        definition_id = cursor.tag_definition_id
        if definition_id is not None:
            self.ledger.add(definition_id)
        assert cursor.tag is not None
        return read_tag_type(
            cursor.tag,
            self.registry,
            label=path_label(self.label, path),
            diagnostics=self.diagnostics,
            range=cursor.range,
        )

    def _reject_spaced_tag(self, cursor: SourceCursor, path: ValuePath) -> None:
        """Reject plain text starting with `(` right after a tag.

        When the tag and the parenthesized part together name a type
        (`!int (1..12) 5`), a space cut the tag's arguments off; otherwise the text
        really starts with `(` (`!str (555) 123-4567`) and must be quoted so a reader
        doesn't take it for arguments. A tag that names no type either way is left
        for `read_tag_type` to report.
        """
        if cursor.scalar_style != "plain":
            return
        value = cursor.value
        if not isinstance(value, str) or not value.startswith("("):
            return
        tag = cursor.tag
        assert tag is not None
        close = value.find(")")
        joined = tag + (value if close < 0 else value[: close + 1])
        label = path_label(self.label, path)
        if _writable_as_tag(joined) and _resolves(joined, self.registry):
            message = (
                f"{label!r}: a YAML tag ends at the first space; "
                f"write `!{joined}` without spaces"
            )
        elif _resolves(tag, self.registry):
            quoted = f"!{tag} {yaml_quoted(value)}"
            message = (
                f"{label!r}: a YAML tag ends at the first space; quote the text: "
                + (f"`{self.label}: {quoted}`" if not path else f"`{quoted}`")
            )
        else:
            return
        raise self.diagnostics.error(message, range=cursor.range)

    def _is_template_text(self, value: Any, cursor: SourceCursor) -> bool:
        """Whether a top-level value is binding text resolved once sources are known.

        Unquoted text starting with `call(` is left to the inline-call desugar, which
        requires exactly one well-formed directive; quoted, it is text.
        """
        if self.literal_only or not isinstance(value, str):
            return False
        if has_span(value):
            return True
        return cursor.scalar_style == "plain" and CALL_HEAD.match(value.strip()) is not None

    def _literal(self, tree: Any) -> Any:
        """Return the value a span-free tree binds: no bindings, strings unescaped."""
        if isinstance(tree, Binding):
            return self._literal(tree.source)
        if isinstance(tree, str):
            return tree if self.literal_only else eval_binding(tree, lambda path: None)
        if type(tree) is dict:
            return {key: self._literal(item) for key, item in tree.items()}
        if type(tree) is list:
            return [self._literal(item) for item in tree]
        return tree

    def _check(self, cursor: SourceCursor, path: ValuePath, typ: Type, value: Any) -> None:
        """Check a span-free literal against its tag's type, explaining a mismatch.

        Raises:
            LoadError: The literal does not fit `typ`.
        """
        try:
            build_value_as(typ, value)
        except TypeCheckError as exc:
            raise self.diagnostics.error(
                self._mismatch(cursor, path, typ, value, exc), range=cursor.range
            ) from None
        except ValueError as exc:
            # The value models validate on their own too; whatever they reject that
            # the structural check let through is still the author's value.
            raise self.diagnostics.error(
                f"{path_label(self.label, path)!r}: the value is not a valid "
                f"{cursor.tag}: {exc}",
                range=cursor.range,
            ) from None

    def _mismatch(
        self,
        cursor: SourceCursor,
        path: ValuePath,
        typ: Type,
        value: Any,
        error: TypeCheckError,
    ) -> str:
        """Explain why a tagged literal does not fit, suggesting a rewrite when one helps.

        A former YAML boolean word such as `yes`, now text, written bare gets the
        `true`/`false` to write when that boolean fits; a non-string scalar whose text
        would fit gets the quoted form.
        """
        label = path_label(self.label, path)
        tag = cursor.tag
        text = cursor.scalar_text
        if text is None:
            message = f"{label!r}: the value is not a valid {tag}: {error}"
            if error.path:
                message += f" (at `{path_label(label, error.path)}`)"
            return message
        # Quoted and block text is shown quoted, so its line breaks stay escaped.
        written = text if cursor.scalar_style == "plain" else yaml_quoted(text)
        message = f"{label!r}: `{written}` is not a valid {tag}"
        if error.path:
            message += f" ({error})"
        else:
            reason = _mismatch_reason(error, value)
            if reason is not None:
                message += f" ({reason})"
        fix = former_boolean_fix(value, typ)
        if fix is not None:
            return f"{message}; {fix[1]}"
        # Quoting only helps when YAML read the text as a non-string and the text
        # itself would fit, e.g. `!str 30` or `!str 2026-01-01`.
        if isinstance(value, str):
            return message
        if value is None and typ.kind is ValueKind.ANY and not typ.nullable:
            # `Any` would read the quoted text as the string "null", which is rarely
            # what a null under `!Any` means.
            optional = f"!Optional({tag}) null"
            return message + "; to allow null, write " + (
                f"`{self.label}: {optional}`" if not path else f"`{optional}`"
            )
        try:
            build_value_as(typ, text)
        except TypeCheckError:
            return message
        quoted = f"!{tag} {yaml_quoted(text)}"
        message += "; quote it to keep it as text: " + (
            f"`{self.label}: {quoted}`" if not path else f"`{quoted}`"
        )
        if isinstance(value, bool):
            message += " (YAML reads unquoted true/false as booleans)"
        return message


def _mismatch_reason(error: TypeCheckError, value: Any) -> str | None:
    """Return the reason a root type check gives beyond the plain mismatch, else `None`.

    A date or datetime check, for example, says what the text must look like.

    Args:
        error (`TypeCheckError`): The failed check of `value`, at the root.
        value (`Any`): The checked literal.

    Returns:
        `str | None`: The reason, or `None` when the error only says the value does
            not match.
    """
    head = f"{value!r} does not match declared type "
    text = str(error)
    if not text.startswith(head):
        return None
    reason = text[len(head) :].partition(": ")[2]
    return reason or None


def reject_tagged_output(cursor: SourceCursor, diagnostics: LoadDiagnostics) -> None:
    """Reject a tag on a node's `output:` type declaration or inside it.

    An `output:` declaration is a type, so a tag there would type nothing.

    Args:
        cursor (`SourceCursor`): The node's `output:` value.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: At the first tag on or inside the declaration.
    """
    # Each entry: a cursor, the key it sits under, and whether it is a list item.
    stack: list[tuple[SourceCursor, str, bool]] = [(cursor, "output", False)]
    while stack:
        current, key, is_item = stack.pop()
        tag = current.tag
        value = current.value
        if tag is not None:
            if is_item:
                message = (
                    f"`output:` declares a type; an item of `{key}:` is tagged "
                    f"`!{tag}`; write the type without a tag"
                )
            else:
                written = value if current.has_value and isinstance(value, str) else tag
                message = (
                    f"`output:` declares a type; write `{key}: {written}`, without a tag"
                )
            raise diagnostics.error(message, range=current.range)
        if type(value) is dict:
            children = [current.child(name) for name in value]
            stack.extend(
                (child, _authored_key(child), False) for child in reversed(children)
            )
        elif type(value) is list:
            stack.extend(
                (current.child(index), key, True)
                for index in reversed(range(len(value)))
            )


def _authored_key(cursor: SourceCursor) -> str:
    """Spell the mapping key selecting `cursor` as written."""
    text = cursor.key_text
    return text if text is not None else str(cursor.source_key_value)


def reject_unread_tags(
    document: SourceDocument,
    consumed: set[int],
    diagnostics: LoadDiagnostics,
    *,
    typedefs_only: bool = False,
) -> None:
    """Reject the first tag in `document` that no slot reader consumed.

    A tag types a flow input, a binding, a flow output, an agent variable, an
    `llm_config` value (the whole record or a field), a value entry in `nodes:`, or a
    value inside one; anywhere else it would be silently ignored. Tags reached only
    through top-level `x-*` keys are exempt. Tags under `typedefs:` and on mapping keys
    get their own messages. Keys are spelled as written, so `on:` stays `on:` although
    YAML reads it as `True`.

    Args:
        document (`SourceDocument`): The loaded document.
        consumed (`set[int]`): Definition ids of the tags slot readers read.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        typedefs_only (`bool`, *optional*, defaults to `False`): Check only tags
            under `typedefs:`, which are read before any slot.

    Raises:
        `LoadError`: At the first offending tag, in document order.
    """
    for node in document.tagged_nodes(skip_top_level=is_extension_key):
        path = node.key_path
        in_typedefs = len(path) >= 2 and path[0] == "typedefs"
        if typedefs_only and not in_typedefs:
            continue
        key = next((step for step in reversed(path) if isinstance(step, str)), None)
        if node.is_key and key is not None:
            raise diagnostics.error(
                f"`!{node.tag} {key}:` tags a mapping key; a tag types a value, so "
                f"put it after the colon: `{key}: !{node.tag} <value>`",
                range=node.range,
            )
        if in_typedefs:
            raise diagnostics.error(
                f"typedef {path[1]!r}: a type definition is written as a type, "
                "without a tag",
                range=node.range,
            )
        if node.definition_id in consumed:
            continue
        where = f"`{key}:`" if key is not None else "the document root"
        raise diagnostics.error(
            "a tag types a value — a flow input, a binding, a flow output, an agent "
            "variable, an `llm_config` value, a value entry in `nodes:`, or a value "
            f"inside one; {where} is not one",
            range=node.range,
        )


def path_label(name: str, path: ValuePath) -> str:
    """Spell a slot name and a path inside its value, as in `x.items[0]`."""
    return name + "".join(
        f".{part}" if isinstance(part, str) else f"[{part}]" for part in path
    )


def has_span(text: str) -> bool:
    """Whether `text` holds a `${...}` span; malformed text counts, for later checks."""
    try:
        return any(isinstance(segment, Span) for segment in scan_template(text))
    except ExpressionError:
        return True


def yaml_quoted(text: str) -> str:
    """Quote `text` as a YAML scalar that reads back as exactly `text`."""
    # A single-quoted scalar folds line breaks and cannot escape, so text with a
    # control character needs a double-quoted scalar with escapes.
    if not text.isprintable():
        return '"' + "".join(_yaml_escaped(ch) for ch in text) + '"'
    # A double-quoted scalar treats backslashes as escapes; a single-quoted one
    # only needs its own quote doubled.
    if '"' in text or "\\" in text:
        return "'" + text.replace("'", "''") + "'"
    return f'"{text}"'


def _yaml_escaped(ch: str) -> str:
    """Escape one character for a YAML double-quoted scalar."""
    if ch.isprintable() and ch not in '"\\':
        return ch
    if ord(ch) > 0xFFFF:
        # JSON would write a surrogate pair, which YAML does not join.
        return f"\\U{ord(ch):08X}"
    # JSON's escapes for the basic plane are all valid YAML escapes.
    return json.dumps(ch)[1:-1]
