"""Values (`${...}`) in an AGENT's `llm_config` layers: leaves, contribution, carrying, resolving.

A layer is a plain ``dict`` (all literal), a ``ValueTemplate`` (fields, block
fields or whole blocks from values), or a ``Binding`` (the whole record from one
value). The helpers here find a layer's value leaves and the references they read,
decide which lower-layer values a higher layer leaves able to contribute, carry an
agent's node and own-flow layers as one reserved binding, and resolve that binding
against a pool. Precedence and merge granularity follow ``merge_llm_config``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, TypeAlias

from yian.expr.expressions import ExpressionError
from yian.expr.template import Span, expr_refs_of, scan_template
from yian.llm_clients.config import PROVIDER_BLOCKS
from yian.nodes.binding import BindingError, _resolve_source
from yian.nodes.contracts import Binding, ValuePath, ValueTemplate, iter_source_strings
from yian.typesys import Type, ValueKind
from yian.typesys.pool import VariablePool
from yian.typesys.values import FieldRead, union_field

Nullable: TypeAlias = Callable[[Any, ValuePath], bool]
"""Says whether a value leaf's value, or the part at a path inside it, may be null or absent."""

LLM_CONFIG_KEY = "__llm_config"
"""The reserved AGENT wiring key that carries the node and own-flow layers."""


def is_value(leaf: Any) -> bool:
    """Return whether a layer leaf is a value rather than a literal.

    A value is a template string holding a `${...}` span, or a typed ``Binding``
    whose source holds one. A tagged literal and span-free text are literals.

    Args:
        leaf (`Any`): A position of a layer tree.

    Returns:
        `bool`: ``True`` for a value.
    """
    if isinstance(leaf, str):
        return _has_span(leaf)
    return isinstance(leaf, Binding) and _dynamic(leaf.source)


def value_leaves(layer: Any) -> dict[ValuePath, Any]:
    """Map each value in a layer to its field path.

    A whole record is the empty path, a whole block ``(block,)``, a field
    ``(field,)``, and a block field ``(block, field, ...)`` down to the value. A
    plain ``dict`` or ``None`` layer has no values.

    Args:
        layer (`Any`): A layer: ``dict``, ``ValueTemplate``, ``Binding`` or ``None``.

    Returns:
        `dict[ValuePath, Any]`: Field path to its leaf (the template string or typed
            ``Binding``), in tree order.
    """
    leaves: dict[ValuePath, Any] = {}
    if isinstance(layer, ValueTemplate):
        _collect_leaves(layer.value, (), leaves)
    elif isinstance(layer, Binding) and is_value(layer):
        leaves[()] = layer
    return leaves


def layer_refs(layer: Any) -> tuple[str, ...]:
    """Return the reference paths a layer's values read, once each, in source order.

    Args:
        layer (`Any`): A layer: ``dict``, ``ValueTemplate``, ``Binding`` or ``None``.

    Returns:
        `tuple[str, ...]`: Paths such as ``input.m`` or ``pick.output``.

    Raises:
        `ExpressionError`: A `${...}` span does not parse.
    """
    if isinstance(layer, dict) or layer is None:
        return ()
    refs = [ref for _, text in iter_source_strings(layer) for ref in expr_refs_of(text)]
    return tuple(dict.fromkeys(refs))


def reads_only_inputs(layer: Any) -> bool:
    """Return whether every value in a layer reads only its flow's inputs (``input.*``).

    A layer without values reads nothing, so it qualifies.

    Args:
        layer (`Any`): A layer: ``dict``, ``ValueTemplate``, ``Binding`` or ``None``.

    Returns:
        `bool`: ``True`` when no reference has a head other than ``input``.

    Raises:
        `ExpressionError`: A `${...}` span does not parse.
    """
    return all(ref.split(".", 1)[0] == "input" for ref in layer_refs(layer))


def type_nullable(typ: Optional[Type], inner: ValuePath = ()) -> bool:
    """Return whether a value of ``typ`` may be null, or lack the part at ``inner``.

    Each step reads a record field, which must be required and non-``Optional``,
    or a field every member of a union of records has. Any other step, and an
    unknown type, may be absent.

    Args:
        typ (`Type | None`): The value's type; ``None`` when it is not known.
        inner (`ValuePath`, *optional*, defaults to `()`): Field names inside the value.

    Returns:
        `bool`: ``True`` unless the value, and every step down ``inner``, is present.
    """
    if typ is None:
        return True
    for segment in inner:
        if _admits_none(typ):
            return True
        if typ.variants is not None:
            read = union_field(typ, str(segment))
            if not isinstance(read, FieldRead) or read.type is None:
                return True
            typ = read.type
            continue
        fields = typ.fields or {}
        if segment not in fields or segment not in (typ.required or frozenset()):
            return True
        typ = fields[segment]
    return _admits_none(typ)


def contract_nullable(leaf: Any, inner: ValuePath) -> bool:
    """A ``Nullable`` that knows only the types tags declare.

    A typed ``Binding`` answers from its contract; any other leaf may be null.

    Args:
        leaf (`Any`): A value leaf.
        inner (`ValuePath`): Field names inside its value.

    Returns:
        `bool`: Whether the value, or its part at ``inner``, may be null or absent.
    """
    if isinstance(leaf, Binding) and leaf.contract is not None:
        return type_nullable(leaf.contract.type, inner)
    return True


def types_nullable(layer: Any, types: Mapping[ValuePath, Optional[Type]]) -> Nullable:
    """A ``Nullable`` that answers from the static type of each value in ``layer``.

    Args:
        layer (`Any`): The layer whose value leaves are asked about.
        types (`Mapping[ValuePath, Type | None]`): Each value leaf's type by its field
            path; ``None`` is not proven, so the value may be null. A leaf whose path is
            missing answers as ``contract_nullable`` does.

    Returns:
        `Nullable`: Whether a leaf of ``layer``, or its part at a path, may be null or
            absent.
    """
    # Leaves are matched by identity; two leaves with equal text read the same value.
    paths = {id(leaf): path for path, leaf in value_leaves(layer).items()}

    def nullable(leaf: Any, inner: ValuePath) -> bool:
        path = paths.get(id(leaf))
        if path is None or path not in types:
            return contract_nullable(leaf, inner)
        return type_nullable(types[path], inner)

    return nullable


def contributing(lower: Any, higher: Any, nullable: Nullable) -> Any:
    """Return ``lower`` without the values ``higher`` always overrides.

    A value sets one merge field: a top-level key, or a provider block's field
    (a deeper value sets the block field it is inside). It can't contribute when
    ``higher`` sets that field to a non-null literal, or to a value that can't be
    null. A higher whole block or whole record shields the field only when it can't
    be null and its type declares the field as present and non-null. A lower whole
    block or whole record always contributes, since a partial higher layer leaves
    the rest of it.

    Args:
        lower (`Any`): The lower-precedence layer.
        higher (`Any`): The layer above it for the same agent.
        nullable (`Nullable`): Whether a higher value leaf, or a part of it, may be
            null or absent.

    Returns:
        `Any`: ``lower`` itself when every value can contribute; otherwise a
            ``ValueTemplate`` without the shielded fields and their origins.
    """
    shielded: set[ValuePath] = set()
    for path in value_leaves(lower):
        field = _merge_field(path)
        if field is not None and _shields(higher, field, nullable):
            shielded.add(field)
    if not shielded:
        return lower
    assert isinstance(lower, ValueTemplate)  # only template layers have field values
    return ValueTemplate(
        _pruned(lower.value, shielded, ()),
        tuple(
            (path, origin)
            for path, origin in lower.origins
            if not any(path[: len(field)] == field for field in shielded)
        ),
    )


def split_layers(
    node_layer: Any, flow_layer: Any, inherit: bool, nullable: Nullable
) -> Optional[Binding]:
    """Build the reserved binding that carries an agent's node and own-flow layers.

    The template is ``{"node": <node layer>, "flow": <flow layer>}``, each carried
    whole with its literal fields; only flow values the node layer always overrides
    are dropped. Under ``inherit: false`` the flow part is left out, as it is for an
    empty flow layer. Literal text is escaped so it resolves to itself, and origins
    are re-keyed under ``node`` and ``flow``.

    Args:
        node_layer (`Any`): The agent's own layer.
        flow_layer (`Any`): The layer of the flow the agent is written in.
        inherit (`bool`): The agent's literal ``inherit``.
        nullable (`Nullable`): Whether a node-layer value leaf may be null or absent.

    Returns:
        `Binding | None`: The binding, or ``None`` when no value can reach the agent.
    """
    flow = contributing(flow_layer, node_layer, nullable) if inherit else None
    if not value_leaves(node_layer) and not value_leaves(flow):
        return None
    tree: dict[str, Any] = {}
    origins: list[tuple[ValuePath, Any]] = []
    for part, layer in (("node", node_layer), ("flow", flow)):
        if part == "flow" and not flow:
            continue
        tree[part], part_origins = _carried(layer)
        origins.extend(((part, *path), origin) for path, origin in part_origins)
    return Binding(ValueTemplate(tree, tuple(origins)))


def resolve_layers(
    template: Binding, pool: VariablePool, *, label: Optional[str] = None
) -> tuple[dict, dict]:
    """Resolve a ``split_layers`` binding into the node and flow layers.

    A null field stays null, which ``merge_llm_config`` reads as unset; a null whole
    record is an empty layer.

    Args:
        template (`Binding`): The binding ``split_layers`` built.
        pool (`VariablePool`): The pool the values read.
        label (`str`, *optional*, defaults to `None`): Names the config in an error
            message in place of ``binding '__llm_config'``.

    Returns:
        `tuple[dict, dict]`: The node layer and the flow layer.

    Raises:
        `BindingError`: A value fails to evaluate, a typed value violates its tag
            (``binding.type``), or a whole record resolves to something other than a
            mapping or null (``contract.input``).
    """
    resolved = _resolve_source(template, pool, name=LLM_CONFIG_KEY, label=label)
    tree = template.source.value
    return (
        _resolved_layer(resolved.get("node"), tree.get("node"), label),
        _resolved_layer(resolved.get("flow"), tree.get("flow"), label),
    )


def keep_values(layer: Any, keep: Callable[[Any], bool]) -> Any:
    """Return ``layer`` without the merge fields that hold a value ``keep`` rejects.

    A rejected value takes its whole merge field with it (a top-level key, or a
    provider block's field), so no partial field is left; a rejected whole block takes
    the block, and a rejected whole record leaves an empty layer. Literal fields stay.

    Args:
        layer (`Any`): A layer: ``dict``, ``ValueTemplate``, ``Binding`` or ``None``.
        keep (`Callable[[Any], bool]`): Whether a value leaf stays.

    Returns:
        `Any`: ``layer`` itself when every value stays; otherwise a ``ValueTemplate``
            without the dropped fields and their origins, or ``{}``.
    """
    dropped: set[ValuePath] = set()
    for path, leaf in value_leaves(layer).items():
        if not keep(leaf):
            field = _merge_field(path)
            dropped.add(path if field is None else field)
    if not dropped:
        return layer
    if () in dropped:
        return {}
    assert isinstance(layer, ValueTemplate)  # only template layers have field values
    return ValueTemplate(
        _pruned(layer.value, dropped, ()),
        tuple(
            (path, origin)
            for path, origin in layer.origins
            if not any(path[: len(field)] == field for field in dropped)
        ),
    )


def resolve_layer(layer: Any, pool: VariablePool, *, label: Optional[str] = None) -> dict:
    """Resolve one layer against a pool into a plain configuration dict.

    Literal text resolves to itself (``$$`` reads as ``$``), and typed values are
    checked against their tags. A null whole record is an empty layer.

    Args:
        layer (`Any`): A layer: ``dict``, ``ValueTemplate``, ``Binding`` or ``None``.
        pool (`VariablePool`): The pool the values read.
        label (`str`, *optional*, defaults to `None`): Names the config in an error
            message in place of ``binding '__llm_config'``.

    Returns:
        `dict`: The resolved layer.

    Raises:
        `BindingError`: As `resolve_layers` raises it.
    """
    part, origins = _carried(layer)
    template = Binding(
        ValueTemplate({"layer": part}, tuple((("layer", *path), o) for path, o in origins))
    )
    resolved = _resolve_source(template, pool, name=LLM_CONFIG_KEY, label=label)
    return _resolved_layer(resolved.get("layer"), part, label)


def _has_span(text: str) -> bool:
    """Whether ``text`` holds a `${...}` span; malformed text counts, for later checks."""
    try:
        return any(isinstance(segment, Span) for segment in scan_template(text))
    except ExpressionError:
        return True


def _whole_span(text: str) -> bool:
    """Whether ``text`` is exactly one `${...}` span, whose value keeps its type."""
    try:
        segments = scan_template(text)
    except ExpressionError:
        return True
    return len(segments) == 1 and isinstance(segments[0], Span)


def _dynamic(node: Any) -> bool:
    """Whether a tree, looking through nested bindings, holds a `${...}` span."""
    if isinstance(node, str):
        return _has_span(node)
    if isinstance(node, Binding):
        return _dynamic(node.source)
    if isinstance(node, ValueTemplate):
        return _dynamic(node.value)
    if type(node) is dict:
        return any(_dynamic(item) for item in node.values())
    if type(node) is list:
        return any(_dynamic(item) for item in node)
    return False


def _collect_leaves(node: Any, path: ValuePath, leaves: dict[ValuePath, Any]) -> None:
    """Add every value under ``node`` to ``leaves``, keyed by its path."""
    if is_value(node):
        leaves[path] = node
    elif type(node) is dict:
        for key, item in node.items():
            _collect_leaves(item, (*path, key), leaves)
    elif type(node) is list:
        for index, item in enumerate(node):
            _collect_leaves(item, (*path, index), leaves)


def _admits_none(typ: Type) -> bool:
    """Whether ``None`` satisfies ``typ``."""
    return typ.nullable or typ.kind is ValueKind.NONE


def _merge_field(path: ValuePath) -> Optional[ValuePath]:
    """Return the field a value at ``path`` sets, or ``None`` for a whole record or block."""
    if not path:
        return None
    if path[0] in PROVIDER_BLOCKS:
        return path[:2] if len(path) >= 2 else None
    return path[:1]


def _shields(higher: Any, field: ValuePath, nullable: Nullable) -> bool:
    """Whether ``higher`` always sets ``field`` to a non-null value."""
    node = higher.value if isinstance(higher, ValueTemplate) else higher
    for index, segment in enumerate(field):
        if is_value(node):
            return not nullable(node, ()) and not nullable(node, field[index:])
        if isinstance(node, Binding):
            # A tagged literal: its source is the value.
            node = node.source
            if isinstance(node, ValueTemplate):
                node = node.value
        if type(node) is not dict or segment not in node:
            return False
        node = node[segment]
    return _always_set(node, nullable)


def _always_set(node: Any, nullable: Nullable) -> bool:
    """Whether a higher layer's field value is never null."""
    if node is None:
        return False
    if isinstance(node, str):
        # Text around a span stringifies the value, so the result is never null.
        return not _has_span(node) or not _whole_span(node) or not nullable(node, ())
    if isinstance(node, Binding):
        return not nullable(node, ()) if is_value(node) else node.source is not None
    return True


def _pruned(node: Any, fields: set[ValuePath], path: ValuePath) -> Any:
    """Rebuild a template tree without the entries at ``fields``."""
    if type(node) is not dict:
        return node
    out: dict[str, Any] = {}
    for key, item in node.items():
        here = (*path, key)
        if here in fields:
            continue
        inside = any(field[: len(here)] == here for field in fields)
        out[key] = _pruned(item, fields, here) if inside else item
    return out


def _carried(layer: Any) -> tuple[Any, tuple[tuple[ValuePath, Any], ...]]:
    """Return a layer as a template part and the origins inside it."""
    if layer is None:
        return {}, ()
    if isinstance(layer, ValueTemplate):
        return layer.value, layer.origins
    if isinstance(layer, Binding):
        if isinstance(layer.source, ValueTemplate):
            return (
                Binding(layer.source.value, origin=layer.origin, contract=layer.contract),
                layer.source.origins,
            )
        return layer, ()
    return _escaped(layer), ()


def _escaped(node: Any) -> Any:
    """Escape every `$` in a literal tree's text, so the binder reads it back unchanged."""
    if isinstance(node, str):
        return node.replace("$", "$$")
    if isinstance(node, Mapping):
        return {key: _escaped(item) for key, item in node.items()}
    if isinstance(node, list):
        return [_escaped(item) for item in node]
    return node


def _resolved_layer(value: Any, carried: Any, label: Optional[str]) -> dict:
    """Return one resolved layer, with a null whole record as an empty layer."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    where = label if label is not None else f"binding {LLM_CONFIG_KEY!r}"
    raise BindingError(
        f"{where}: llm_config must resolve to a mapping or null, "
        f"got {type(value).__name__}",
        failure_code="contract.input",
        origin=carried.origin if isinstance(carried, Binding) else None,
    )
