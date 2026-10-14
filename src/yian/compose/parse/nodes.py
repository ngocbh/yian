"""Graph-stage: parse the `nodes:` mapping into typed per-kind descriptors.

The first cross-node assembly pass. Consumes a source cursor over the flow's
`nodes:` mapping and classifies each entry by its YAML tag: a tagged entry is a value
(`total: !int ${a.output + b.output}`) handed to the caller's `read_value` reader, and an
untagged mapping with `kind:` is a node body dispatched to its kind's `parse` via the
`KINDS` table. Any other entry is a located error that says to add `kind:` for a node or
a tag for a value.

Only the generic, kind-independent pre-checks live here (node id shape and reserved ids,
entry classification, the `input:`/`output:` -> plural back-map, `kind:` resolution);
each kind's own `parse` owns its field legality (allowed/required/asserts/env) and its
descriptor-local `SourceRange` recording (each field tagged `("field", name)`). Two
kind-blind enrichments run after every kind parse: each `input:` binding's own range is
stamped into the descriptor's `field_ranges` under `("input", name)`, and each `asserts:`
item range is appended positionally to its `assert_ranges`. This metadata remains inside
Compose; `_finish()` copies node/field locations to `BuildResult` for validation, while
runtime diagnosis uses only opaque semantic-carrier origins. Source-order origin
registration keeps duplicate assertions distinct without an aligned range list on the
semantic node.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import re
from collections.abc import Mapping
from typing import Any, Callable, NoReturn

from yian.compile.model import END_ID, START_ID
from yian.compose.helpers.directives import CALL_HEAD
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.source import one_line_text
from yian.compose.helpers.tags import QUOTED_STYLES, suggest_tag, tag_text, yaml_quoted
from yian.compose.nodes import KINDS, REMOVED_KINDS
from yian.compose.nodes.base import not_text_error
from yian.compose.source import (
    SourceAccessError,
    SourceCursor,
    SourceSyntaxError,
    parse_yaml_literal,
)
from yian.expr import ExpressionError, scan_template
from yian.expr.grammar import EXPRESSION_KEYWORDS
from yian.expr.template import Span
from yian.typesys import TypeCheckError, build_value
from yian.typesys.values import Type, ValueKind, list_kind_for_element, value_repr

# Singular source keys are projected onto the descriptor dataclasses' plural
# field names while retaining their original child sources.
_INPUT_OUTPUT_REMAP = {"input": "inputs", "output": "outputs"}

# Explicit ASCII classes, matching the expression `NAME` terminal: `str.isidentifier`
# and `\w` also accept non-ASCII letters (`café`), which `${...}` can't reference.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NON_IDENTIFIER_CHAR = re.compile(r"[^A-Za-z0-9_]")

IDENTIFIER_RULE = "ASCII letters, digits, and _, not starting with a digit"

# Item types a list of numbers joins to `float`.
_INT = Type(kind=ValueKind.INTEGER)
_FLOAT = Type(kind=ValueKind.NUMBER)
# A null while list items are joined: its type is the other items' type, made optional.
_NULL = Type(kind=ValueKind.NONE, nullable=True)
# The item type of a list whose items share no type.
_ANY = Type(kind=ValueKind.ANY)


def is_identifier(name: object) -> bool:
    """Report whether `name` is an identifier a `${...}` reference can name.

    Args:
        name (`object`): The candidate name; any non-`str` value is not an identifier.

    Returns:
        `bool`: True when `name` is a `str` matching `[A-Za-z_][A-Za-z0-9_]*`.
    """
    return isinstance(name, str) and _IDENTIFIER.fullmatch(name) is not None


def identifier_suggestion(name: str) -> str:
    """Spell `name` as an identifier, for a rename hint.

    Args:
        name (`str`): The non-identifier name to respell.

    Returns:
        `str`: `name` with each character outside `[A-Za-z0-9_]` replaced by `_` and a `_`
        prefixed when it starts with a digit, or `""` for an empty name.
    """
    suggestion = _NON_IDENTIFIER_CHAR.sub("_", name)
    if suggestion[:1].isdigit():
        suggestion = "_" + suggestion
    return suggestion


_NODE_OR_VALUE = (
    "node {node_id!r}: an entry in `nodes:` is a node or a value; add `kind:` for a "
    "node, or a tag (`!object`, `!Buy`, ...) for a value"
)


# Reads one tagged `nodes:` entry: `(entry id, entry cursor) -> descriptor`.
ValueReader = Callable[[str, SourceCursor], Any]


def parse_nodes(
    nodes_cursor: SourceCursor,
    *,
    read_value: ValueReader | None = None,
) -> dict[str, Any]:
    """Parse one `nodes:` source cursor into typed per-kind descriptors.

    The source is physical for a full graph, source-aligned for compact/def
    projections, or virtual for a programmatic/test mapping. A tagged entry is a
    value and is read by `read_value`; every other entry must be a node body with
    `kind:`.

    Args:
        nodes_cursor (`SourceCursor`):
            Cursor over the `nodes:` mapping. A non-mapping value parses to no nodes.
        read_value (`ValueReader | None`, *optional*, defaults to `None`):
            Reads one tagged entry, given its id and cursor, into its descriptor.
            `None` is only valid for a mapping with no tagged entry.

    Returns:
        `dict[str, Any]`:
            The descriptors keyed by entry id, in source order, each anchored on its
            entry's key line whenever an authored source exists.

    Raises:
        `LoadError`:
            Raised at the entry for a bad or reserved id, an untagged entry that is
            not a node body, or a node body that fails its kind's parse.
        `TypeError`:
            Raised when a tagged entry is found and `read_value` is `None`.
    """
    diagnostics = LoadDiagnostics.from_cursor(nodes_cursor)
    out: dict[str, Any] = {}
    try:
        nodes = nodes_cursor._value_view()
        if not isinstance(nodes, Mapping):
            return out
        for node_id in list(nodes):
            desc = _parse_node(node_id, nodes_cursor.child(node_id), read_value)
            # Anchor the descriptor's node-level `.range` on the node's HEADER line (its
            # `key:` token), not its body/value block. The header is the node-level
            # diagnostic anchor a cycle, an unresolved ref, or a namespaced runtime failure
            # boxes.
            header = nodes_cursor.key_range_of(node_id)
            if header is not None and getattr(desc, "range", None) is not None:
                desc = dataclasses.replace(desc, range=header)
            out[node_id] = desc
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc
    return out


def _parse_node(
    node_id: str, cursor: SourceCursor, read_value: ValueReader | None
) -> Any:
    """Read one `nodes:` entry into its descriptor (located).

    A tagged entry goes to `read_value`; an untagged one must be a node body with
    `kind:` and is dispatched to its kind's `parse`.
    """
    diagnostics = LoadDiagnostics.from_cursor(cursor)
    _reject_reserved_node_id(node_id, cursor)
    # A tag makes the entry a value even when its mapping has a `kind:` key.
    if cursor.tag is not None:
        if read_value is None:
            raise TypeError(
                f"node {node_id!r} is a tagged value entry; parse_nodes needs "
                "read_value to read it"
            )
        return read_value(node_id, cursor)
    body = cursor._value_view()
    if not isinstance(body, Mapping):
        _reject_untagged_value(node_id, cursor, diagnostics)
    cursor = normalize_node_signature_cursor(cursor)

    kind = cursor.get("kind")
    if kind is None:
        raise diagnostics.error(
            _NODE_OR_VALUE.format(node_id=node_id),
            range=cursor.range or cursor.key_range,
        )
    if not isinstance(kind, str):
        raise diagnostics.error(
            f"node {node_id!r}: "
            + not_text_error("kind", kind, fix=f"write one of {', '.join(sorted(KINDS))}"),
            range=cursor.range_of("kind"),
        )
    if kind in REMOVED_KINDS:
        raise diagnostics.error(
            f"node {node_id!r}: kind {kind!r} was removed; {REMOVED_KINDS[kind]}",
            range=cursor.range_of("kind"),
        )
    cls = KINDS.get(kind)
    if cls is None:
        raise diagnostics.error(
            f"node {node_id!r}: unknown kind {value_repr(kind)} "
            f"(allowed: {', '.join(sorted(KINDS))})",
            range=cursor.range_of("kind"),
        )
    desc = cls.parse(node_id, cursor)
    # Compose-only range enrichments retained on parsed descriptors, keyed by `(kind, key)`
    # rather than a bare name so namespaces cannot collide (the kind already stamped
    # its own fields as `("field", name)` in `cls.parse`):
    #   - each `input:` binding -> its own range, tagged `("input", name)` and distinct
    #     from a like-named field.
    #   - each `asserts:` list item -> its POSITIONAL range, appended to the descriptor's
    #     `assert_ranges` (aligned 1:1 with the authored `asserts:` list). Asserts are located by
    #     INDEX, not by their (non-unique) text, so two identical asserts get distinct carets — hence
    #     they live in a list, NOT in `field_ranges`.
    # Runtime diagnosis does not consume these ranges, and the semantic Node never receives
    # them. Both no-op for a source-free cursor or a descriptor without the field.
    # `input:` is read AFTER
    # `normalize_node_signature_cursor` has renamed the author's singular
    # `input:` key to plural.
    field_ranges = getattr(desc, "field_ranges", None)
    if isinstance(field_ranges, dict):
        field_ranges.update(
            {
                ("input", name): rng
                for name, rng in cursor.mapping_value_ranges("inputs").items()
            }
        )
    assert_ranges = getattr(desc, "assert_ranges", None)
    if isinstance(assert_ranges, list):
        assert_ranges.extend(cursor.item_range_list("asserts"))
    return desc


def _reject_untagged_value(
    node_id: str, cursor: SourceCursor, diagnostics: LoadDiagnostics
) -> NoReturn:
    """Reject an untagged scalar, list, or null entry, showing the tag to write.

    A literal gets an example of the entry with its tag. A `${...}` or `call(...)`
    value is not a literal, so it gets no example.

    Args:
        node_id (`str`): The entry id.
        cursor (`SourceCursor`): The entry's value.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: Always, at the entry.
    """
    value = cursor.value
    rng = cursor.range or cursor.key_range
    message = _NODE_OR_VALUE.format(node_id=node_id)
    if _is_reference(value, cursor.scalar_style):
        raise diagnostics.error(message, range=rng)
    if not _writable(value):
        raise diagnostics.error(message, range=rng)
    typ = example_type(value)
    if typ is not None:
        message += f", e.g. {suggest_tag(node_id, typ, _example_text(value, cursor))}"
    else:
        quoted = tag_text(quoted_dates_type(value))
        if quoted is not None:
            # The example shows the list as written, so its dates are still unquoted.
            message += (
                f", e.g. `{node_id}: !<type> {_example_text(value, cursor)}` "
                f"(quote its dates to tag it `{quoted}`)"
            )
    raise diagnostics.error(message, range=rng)


def _is_reference(value: Any, style: str | None) -> bool:
    """Report whether an untagged value is a `${...}` template or a `call(...)` directive."""
    if not isinstance(value, str):
        return False
    if _is_binding_text(value):
        return True
    # A quoted `call(...)` is text, as in a binding.
    return style not in QUOTED_STYLES and CALL_HEAD.match(value.strip()) is not None


def _is_binding_text(value: Any) -> bool:
    """Return whether `value` is template text holding at least one `${...}` span.

    Args:
        value (`Any`): An authored value.

    Returns:
        `bool`: `True` for a string with a well-formed `${...}` span. Plain text, a
            `$${...}` escape (literal text) and malformed template text are `False`.
    """
    if not isinstance(value, str):
        return False
    try:
        return any(isinstance(segment, Span) for segment in scan_template(value))
    except ExpressionError:
        return False


def example_type(value: Any) -> Type | None:
    """Return the widened type of a value to tag it with in an example.

    Args:
        value (`Any`): An authored YAML value that does not contain itself.

    Returns:
        `Type | None`: `Optional(str)` for null, `date` or `datetime` for an unquoted
            date, `object` for a mapping, a list of its items' joined type, else the
            widened type of the value; `None` for a number that is not finite, a value
            no type holds, or a list whose items share no type and that holds an
            unquoted date (see `quoted_dates_type`). Items join recursively: a null
            makes the type optional, ints and floats join to `float`, and an empty
            list joins any list, so `[[1], [null]]` is `list(list(Optional(int)))` and
            `[[], [1]]` is `list(list(int))`. A list of nulls only is
            `list(Optional(str))`. Items that share no type are `Any`, or
            `Optional(Any)` when one of them is null, so `[1, a]` is `list(Any)` and
            `[null, 1, a]` is `list(Optional(Any))`.
    """
    typ = _item_type(value)
    return None if typ is None else _settled(typ)


def _item_type(value: Any) -> Type | None:
    """Return `example_type(value)`, but with `_NULL` for a null left to be joined."""
    if value is None:
        return _NULL
    # YAML reads an unquoted `2024-01-01` as a date, which no value type holds.
    if isinstance(value, datetime.datetime):
        return Type(kind=ValueKind.DATETIME)
    if isinstance(value, datetime.date):
        return Type(kind=ValueKind.DATE)
    if isinstance(value, dict):
        return Type(kind=ValueKind.OBJECT)
    if isinstance(value, list) and value:
        types = [_item_type(item) for item in value]
        if any(typ is None for typ in types):
            return None
        element: Type | None = types[0]
        for typ in types[1:]:
            element = None if element is None else _joined(element, typ)
        if element is not None:
            return _list_of(element)
        if _holds_date(value):
            return None
        return _list_of(dataclasses.replace(_ANY, nullable=None in value))
    try:
        # An empty list is a `list(Optional(Any))` with no element type, which joins any list.
        return Type.scalar(build_value(value).kind)
    except TypeCheckError:
        return None


def quoted_dates_type(value: Any) -> Type | None:
    """Return `example_type(value)` as if each unquoted date in it were quoted text.

    `Any` holds a mixed list's items, but not an unquoted date (YAML reads it as a
    date, not text), so `example_type` gives no type for a mixed list holding one.
    Quoted, the date is text, and the list is `list(Any)` or a narrower list.

    Args:
        value (`Any`): An authored YAML value that does not contain itself.

    Returns:
        `Type | None`: The type of `value` with its dates read as text, such as
            `list(Any)` for `[1, 2024-01-01]`, or `None` as for `example_type`.
    """
    return example_type(_dates_as_text(value))


def _dates_as_text(value: Any) -> Any:
    """Return `value`, which doesn't contain itself, with each date as its ISO text."""
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _dates_as_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dates_as_text(item) for item in value]
    return value


def _holds_date(value: Any) -> bool:
    """Report whether a value that doesn't contain itself holds an unquoted date."""
    if isinstance(value, datetime.date):
        return True
    if isinstance(value, dict):
        return any(_holds_date(item) for item in value.values())
    if isinstance(value, list):
        return any(_holds_date(item) for item in value)
    return False


def _joined(first: Type, second: Type) -> Type | None:
    """Return the one type two item types widen to, or `None` when there is none."""
    if first.kind is ValueKind.NONE:
        return dataclasses.replace(second, nullable=True)
    if second.kind is ValueKind.NONE:
        return dataclasses.replace(first, nullable=True)
    nullable = first.nullable or second.nullable
    first = dataclasses.replace(first, nullable=False)
    second = dataclasses.replace(second, nullable=False)
    if first == second:
        joined = first
    elif first in (_INT, _FLOAT) and second in (_INT, _FLOAT):
        joined = _FLOAT
    elif first.kind.is_list() and second.kind.is_list():
        if first.element is None or second.element is None:
            joined = second if first.element is None else first
        else:
            element = _joined(first.element, second.element)
            if element is None:
                return None
            joined = _list_of(element)
    else:
        return None
    return dataclasses.replace(joined, nullable=True) if nullable else joined


def _settled(typ: Type) -> Type:
    """Return `typ` with each `_NULL` left in it read as `Optional(str)`."""
    if typ.kind is ValueKind.NONE:
        return Type(kind=ValueKind.STRING, nullable=True)
    if typ.element is None:
        return typ
    element = _settled(typ.element)
    return dataclasses.replace(
        typ, kind=list_kind_for_element(element), element=element
    )


def _list_of(element: Type) -> Type:
    """Return the list type of an element type."""
    return Type(kind=list_kind_for_element(element), element=element)


def _writable(value: Any) -> bool:
    """Report whether a value entry can hold a value: no non-finite number, no cycle.

    Args:
        value (`Any`): An authored YAML value.

    Returns:
        `bool`: False when a number is `.inf`, `-.inf` or `.nan`, or a mapping or list
            holds itself through a YAML alias.
    """
    return not contains_itself(value) and _finite(value)


def _finite(value: Any) -> bool:
    """Report whether every number in a value that doesn't contain itself is finite."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def contains_itself(value: Any, *, active: frozenset[int] = frozenset()) -> bool:
    """Report whether a mapping or list holds itself through a YAML alias (`&R [*R]`).

    Args:
        value (`Any`): An authored YAML value or a part of one.
        active (`frozenset[int]`, *optional*, defaults to `frozenset()`): The `id()`s
            of the mappings and lists enclosing `value`.

    Returns:
        `bool`: True when a mapping or list holds itself at some depth.
    """
    if not isinstance(value, (dict, list)):
        return False
    if id(value) in active:
        return True
    items = value.values() if isinstance(value, dict) else value
    inner = active | {id(value)}
    return any(contains_itself(item, active=inner) for item in items)


def _example_text(value: Any, cursor: SourceCursor) -> str:
    """Spell an untagged entry's value for its example.

    A value written over several lines (a block list) is shown as `...`. A date is
    quoted, since a tag reads only a quoted date, and so is text that YAML would not
    read back as the same text (`!!str 3` is shown as `"3"`).
    """
    if value is None:
        return "null"
    if isinstance(value, str):
        text = cursor.scalar_text
        if cursor.scalar_style == "plain" and text is not None and _reads_as(text, value):
            return text
        return yaml_quoted(value)
    text = cursor.scalar_text
    if isinstance(value, datetime.date) and text is not None:
        return yaml_quoted(text)
    if text is not None:
        return text
    return one_line_text(cursor.document, cursor.range) or "..."


def _reads_as(text: str, value: str) -> bool:
    """Report whether the plain scalar `text` reads back as the string `value`."""
    try:
        return parse_yaml_literal(text) == value
    except SourceSyntaxError:
        return False


def _reject_reserved_node_id(node_id: str, cursor: SourceCursor) -> None:
    """Reject a node id that isn't a referenceable identifier or collides with a reserved name.

    A node id must be an identifier (`${<node>.output}` names it) that is neither an
    expression keyword, the synthesized boundary, nor a resolver head; `#`, `/`, and `.`
    get their own reserved-separator message.

    Mirrors the legacy `parser._parse_node` head guards; anchors each error at the node
    body's `SourceRange` (the closest located anchor for a bad node key).
    """
    diagnostics = LoadDiagnostics.from_cursor(cursor)
    rng = cursor.range
    if "#" in node_id or "/" in node_id or "." in node_id:
        raise diagnostics.error(
            f"node id {node_id!r} uses a reserved separator ('#' and '/' are reserved "
            f"for runtime graph expansion; '.' is reserved for reference paths)",
            range=rng,
        )
    if not is_identifier(node_id):
        reference = f" so ${{{node_id}.output}} can reference it" if node_id else ""
        suggestion = identifier_suggestion(node_id)
        hint = f"; try {suggestion!r}" if suggestion else ""
        raise diagnostics.error(
            f"node id {node_id!r} must be an identifier ({IDENTIFIER_RULE}){reference}{hint}",
            range=rng,
        )
    if node_id in (START_ID, END_ID):
        raise diagnostics.error(
            f"node id {node_id!r} is reserved for the synthesized START_ID/END_ID boundary"
            f"; authors write `input:`/`output:`, not a boundary node",
            range=rng,
        )
    if node_id in EXPRESSION_KEYWORDS:
        raise diagnostics.error(
            f"node id {node_id!r} is an expression keyword, so ${{{node_id}.output}} "
            f"can't reference it; pick a different node id",
            range=rng,
        )
    # Reserve the four singular head literals so the node-first ref `${<node>.output.k}` is
    # unambiguous, and the plural forms as migration typo-catchers.
    if node_id in ("input", "output", "system", "item"):
        raise diagnostics.error(
            f"node id {node_id!r} is reserved (used as `${{{node_id}.X}}` as a "
            f"resolver head — pick a different node id)",
            range=rng,
        )
    if node_id in ("inputs", "outputs"):
        raise diagnostics.error(
            f"node id {node_id!r} is reserved (the plural head is retired — "
            f"rename to `{node_id[:-1]}` or pick a different node id)",
            range=rng,
        )


def normalize_node_signature_cursor(cursor: SourceCursor) -> SourceCursor:
    """Project authored node-signature keys onto descriptor field names.

    The projection is the single crossing from the singular Compose surface to
    the plural Python carrier fields. Later phases that correlate a descriptor
    with its source must reuse this cursor so collision and source-selection
    semantics cannot drift from parsing.

    Args:
        cursor (`SourceCursor`):
            Cursor over one node body mapping in authored or projected form.

    Returns:
        `SourceCursor`:
            Pure projection with signature keys normalized to carrier names.

    Raises:
        `SourceAccessError`:
            Raised when `cursor` is not a traversable mapping.
    """

    return cursor.renamed(_INPUT_OUTPUT_REMAP)
