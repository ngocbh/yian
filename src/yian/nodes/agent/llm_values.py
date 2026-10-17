"""Values (`${...}`) in an AGENT's `llm_config` layers: leaves, contribution, carrying, resolving.

A layer is a plain ``dict`` (all literal), a ``ValueTemplate`` (fields, block
fields or whole blocks from values), or a ``Binding`` (the whole record from one
value). The helpers here find a layer's value leaves and the references they read,
decide which lower-layer values a higher layer leaves able to contribute, carry an
agent's node and own-flow layers as one reserved binding, resolve that binding
against a pool, check the resolved layers, and build a call-time agent's config.
Precedence and merge granularity follow ``merge_llm_config``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, TypeAlias

from pydantic import ValidationError

from yian.diagnostics import OriginRef
from yian.expr.expressions import ExpressionError
from yian.expr.template import Span, expr_refs_of, scan_template
from yian.llm_clients.config import (
    PROVIDER_BLOCKS,
    SUPPORTED_LLM_PROVIDERS,
    HostConnection,
    LLMConfig,
    host_only_field_error,
    merge_llm_config,
)
from yian.llm_clients.connection import (
    HOST_KEY_MARKER,
    ConnectionSelection,
    block_problem,
    check_model_config,
    refused_block,
)
from yian.llm_clients.layers import (
    require_concrete_selection,
    resolve_agent_config,
    validate_layer,
)
from yian.llm_clients.provider_blocks import located_block_problems
from yian.nodes.binding import BindingError, _resolve_source
from yian.nodes.contracts import Binding, ValuePath, ValueTemplate, iter_source_strings
from yian.typesys import Type, ValueKind
from yian.typesys.pool import VariablePool
from yian.typesys.values import FieldRead, union_field

Nullable: TypeAlias = Callable[[Any, ValuePath], bool]
"""Says whether a value leaf's value, or the part at a path inside it, may be null or absent."""

FieldWhere: TypeAlias = Callable[[ValuePath], str]
"""Names a field of a layer in an error message, from its path in the layer."""

LLM_CONFIG_KEY = "__llm_config"
"""The reserved AGENT wiring key that carries the node and own-flow layers."""

HOST_ONLY_FROM_VALUE = "credentials and where they are sent stay with the host or yian auth"
"""Why a host-only field may not come from a value."""


@dataclass(frozen=True)
class CallTimeLLM:
    """What run start fixes for an agent whose ``llm_config`` resolves only when it runs.

    Attributes:
        path (`str`): The agent's static path.
        base (`dict[str, Any]`): The layers below the agent's node and own-flow layers:
            the enclosing flows' layers and the host layer, merged; empty under
            ``inherit: false``. It holds no key or header value.
        selection (`ConnectionSelection`): The run's recorded selection: the defaults
            and saved models the resolved config falls back to, and the connection it
            binds.
    """

    path: str
    base: dict[str, Any]
    selection: ConnectionSelection


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
    are dropped. Under ``inherit: false`` the flow part is left out, as it is when
    nothing of the flow layer is left. Literal text is escaped so it resolves to
    itself, and origins are re-keyed under ``node`` and ``flow``.

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
        carried, part_origins = _carried(layer)
        if part == "flow" and carried == {}:
            continue
        tree[part] = carried
        origins.extend(((part, *path), origin) for path, origin in part_origins)
    return Binding(ValueTemplate(tree, tuple(origins)))


def resolve_layers(
    template: Binding,
    pool: VariablePool,
    *,
    node_where: FieldWhere,
    flow_where: FieldWhere,
) -> tuple[dict, dict]:
    """Resolve a ``split_layers`` binding into the node and flow layers.

    Each layer resolves on its own, so an error names the field under that layer's
    subject. A null field stays null, which ``merge_llm_config`` reads as unset; a null
    whole record is an empty layer.

    Args:
        template (`Binding`): The binding ``split_layers`` built.
        pool (`VariablePool`): The pool the values read.
        node_where (`FieldWhere`): Names a node-layer field in an error message, from
            its path in the layer, such as ``node 'chat' llm_config.model``.
        flow_where (`FieldWhere`): Names a flow-layer field the same way, such as
            ``flow llm_config.model``.

    Returns:
        `tuple[dict, dict]`: The node layer and the flow layer.

    Raises:
        `BindingError`: A value fails to evaluate, a typed value violates its tag
            (``binding.type``), or a whole record resolves to something other than a
            mapping or null (``contract.input``).
    """
    source = template.source
    return (
        _resolve_part(source, "node", pool, node_where),
        _resolve_part(source, "flow", pool, flow_where),
    )


def _resolve_part(
    source: ValueTemplate, part: str, pool: VariablePool, where: FieldWhere
) -> dict:
    """Resolve the ``part`` layer of a `split_layers` template, naming fields with ``where``."""
    if part not in source.value:
        return {}
    layer = source.value[part]
    template = Binding(ValueTemplate(
        {part: layer},
        tuple((path, origin) for path, origin in source.origins if path[:1] == (part,)),
    ))
    resolved = _resolve_source(
        template, pool, name=LLM_CONFIG_KEY, label=lambda path: where(path[1:])
    )
    return _resolved_layer(resolved[part], layer, where(()))


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


def resolve_layer(
    layer: Any, pool: VariablePool, *, where: Optional[FieldWhere] = None
) -> dict:
    """Resolve one layer against a pool into a plain configuration dict.

    Literal text resolves to itself (``$$`` reads as ``$``), and typed values are
    checked against their tags. A null whole record is an empty layer.

    Args:
        layer (`Any`): A layer: ``dict``, ``ValueTemplate``, ``Binding`` or ``None``.
        pool (`VariablePool`): The pool the values read.
        where (`FieldWhere`, *optional*, defaults to `None`): Names a field in an error
            message, from its path in the layer, such as ``flow llm_config.model``.

    Returns:
        `dict`: The resolved layer.

    Raises:
        `BindingError`: As `resolve_layers` raises it.
    """
    part, origins = _carried(layer)
    template = Binding(
        ValueTemplate({"layer": part}, tuple((("layer", *path), o) for path, o in origins))
    )
    label = None if where is None else (lambda path: where(path[1:]))
    resolved = _resolve_source(template, pool, name=LLM_CONFIG_KEY, label=label)
    return _resolved_layer(resolved.get("layer"), part, None if where is None else where(()))


def refused_resolved_field(
    written: Any, resolved: Mapping[str, Any], subject: str
) -> Optional[str]:
    """Return why a layer resolved from values sets a field no flow file may set, or `None`.

    ``_connection`` is internal to recorded runs. A host-only field (``api_key``,
    ``base_url``, ``organization``, ``project``, ``headers``) that a value set, as the
    whole record, a whole block or the field itself, is refused whatever it resolved
    beside, since the same value may have set the provider too. A host-only field
    written as a literal key is checked as an authored layer is, against the layer's
    literal keys only: ``base_url`` passes only beside a literal ``provider: azure``.

    Args:
        written (`Any`): The layer as written: a layer (``dict``, ``ValueTemplate``,
            ``Binding`` or ``None``), or its part of a `split_layers` binding.
        resolved (`Mapping[str, Any]`): What ``written`` resolved to.
        subject (`str`): Names the layer, such as ``node 'chat' llm_config``.

    Returns:
        `str | None`: The message, naming the field under ``subject``; it never echoes
            a value.
    """
    refusal = _refusal(written, resolved)
    if refusal is None:
        return None
    field, text = refusal
    return f"{field_where(subject, field)}: {text}"


class LayerFieldError(ValueError):
    """A layer resolved from values, refused at one of its fields.

    Attributes:
        field (`ValuePath`): The field at fault; empty for the layer as a whole. A
            provider block's problem names the deepest key path it reports, such as
            ``("anthropic", "thinking", "budget_tokens")``.
    """

    def __init__(self, message: str, field: ValuePath) -> None:
        """Hold the message and the field it names."""
        super().__init__(message)
        self.field = field


def check_resolved_layer(
    layer: dict[str, Any], subject: str, *, written: Any, callsite: str = ""
) -> None:
    """Check a layer resolved from values as an authored layer is checked, naming the field.

    Messages read ``<subject>.<field><callsite>: ...`` for the first field at fault (a
    provider block's problems name their key path), or ``<subject><callsite>: ...`` for
    a problem of the layer as a whole, and never echo a value. A host-only field or
    ``_connection`` is refused first (`refused_resolved_field`).

    Args:
        layer (`dict[str, Any]`): The resolved layer.
        subject (`str`): Names the layer, such as ``node 'chat' llm_config``.
        written (`Any`): The layer as written, which tells a literal field from one a
            value set.
        callsite (`str`, *optional*, defaults to `""`): Follows the field, such as
            `` in 'sub'`` for a synthesized agent's layer inside a call.

    Raises:
        `LayerFieldError`: The layer sets a field a flow file may not set, an unknown or
            retired field, an unknown provider, or a field or block its schema refuses.
    """
    refusal = _refusal(written, layer)
    if refusal is not None:
        field, text = refusal
        raise LayerFieldError(f"{field_where(subject, field)}{callsite}: {text}", field)
    try:
        validate_layer(layer)
    except ValueError as exc:
        raise LayerFieldError(f"{subject}{callsite}: {exc}", ()) from None
    for block in PROVIDER_BLOCKS:
        if layer.get(block) is None:
            continue
        problems = located_block_problems(block, layer[block], lambda _leaf: False)
        if problems:
            raise LayerFieldError(
                block_problem(subject, block, problems[0].message, callsite),
                (block, *problems[0].keys),
            )
    try:
        LLMConfig(**layer)
    except ValidationError as exc:
        error = exc.errors(include_input=False, include_url=False)[0]
        loc = tuple(error["loc"])
        field = loc[:1]
        if loc == ("provider",) and error["type"] == "literal_error":
            text = f"not a known provider; use one of: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
        elif error["type"] == "value_error":
            text = str(error["ctx"]["error"])
        else:
            # A block's field is named in full, as in `<subject>.openai.temperature`.
            text = error["msg"]
            field = loc
        raise LayerFieldError(
            f"{field_where(subject, field)}{callsite}: {text}", field
        ) from None


def agent_label(label: Optional[str], parent: str) -> Optional[str]:
    """Return a synthesized agent's ``llm_config`` label, with its callsite when nested.

    Args:
        label (`str | None`): The agent's ``llm_config_label``; ``None`` for an
            authored agent.
        parent (`str`): The static path of the callsite the agent is in; empty at the
            run's flow.

    Returns:
        `str | None`: ``label``, or ``<label> in '<parent>'`` when nested; ``None`` for
            an authored agent, whose messages name its path instead.
    """
    if label is None:
        return None
    return f"{label} in {parent!r}" if parent else label


def field_where(subject: str, field: ValuePath) -> str:
    """Name a field under an ``llm_config`` subject, as in ``node 'a' llm_config.model``."""
    return subject + "".join(f".{part}" for part in field)


def invocation_config(
    canonical: Mapping[str, Any], credentials: Optional[HostConnection]
) -> dict[str, Any]:
    """Return the config one model call uses: ``canonical`` with the host's secrets joined.

    The host key and header values are joined only when ``canonical`` is marked for the
    host key (``HOST_KEY_MARKER``). The result is for that call alone; it is never
    stored or recorded.

    Args:
        canonical (`Mapping[str, Any]`): A resolved config with no secrets.
        credentials (`HostConnection | None`): The run's host key and header values.

    Returns:
        `dict[str, Any]`: A new config.
    """
    config = copy.deepcopy(dict(canonical))
    if config.get(HOST_KEY_MARKER) is True and credentials is not None:
        if credentials.api_key is not None:
            config["api_key"] = credentials.api_key
        if credentials.headers:
            config["headers"] = dict(credentials.headers)
    return config


def call_time_config(
    template: Binding,
    pool: VariablePool,
    call_time: CallTimeLLM,
    *,
    label: Optional[str],
    credentials: Optional[HostConnection],
) -> dict[str, Any]:
    """Resolve and check a call-time agent's config; return it with no secrets.

    The node and own-flow layers resolve against ``pool`` and are each checked as
    an authored layer is. They merge over the recorded base, the defaults and saved
    models of the recorded selection fill the gaps, and the selection binds the
    connection: an agent of the host key's provider is marked for it with the host's
    base_url and nonsecret settings, as run start marks a static agent. The result
    has the shape of a ``RunStarted`` map entry. It is then checked as the model call
    would check it, with the host's secrets joined only when it is marked.

    Args:
        template (`Binding`): The agent's reserved ``__llm_config`` binding.
        pool (`VariablePool`): The pool its values read.
        call_time (`CallTimeLLM`): What run start fixed for the agent.
        label (`str | None`): The authored site of a synthesized agent's
            ``llm_config``, named instead of its path.
        credentials (`HostConnection | None`): The run's host key and header values.

    Returns:
        `dict[str, Any]`: The canonical config: no key, header value or ``SecretStr``.

    Raises:
        `BindingError`: A value breaks its tag (``binding.type``) or fails to resolve
            (its own code), or the resolved config is refused (``contract.input``).
            The message names the field, under ``node '<path>' llm_config`` or the
            label, and the origin is that field's authored site when known.
        `AuthError`: The config is marked for the host key, but the run holds no key,
            or header names other than those run start recorded.
    """
    path = call_time.path
    parent = path.rpartition("/")[0]
    site = label or f"node {path!r} llm_config"
    subject = agent_label(label, parent) or site
    # A nested label names its callsite after the field: `<label>.model in 'sub'`.
    callsite = subject[len(site):]
    flow_subject = f"flow {parent!r} llm_config" if parent else "flow llm_config"
    tree = template.source.value
    node_layer, flow_layer = resolve_layers(
        template,
        pool,
        node_where=lambda field: field_where(site, field) + callsite,
        flow_where=lambda field: field_where(flow_subject, field),
    )
    for part, layer, where, after in (
        ("node", node_layer, site, callsite), ("flow", flow_layer, flow_subject, ""),
    ):
        try:
            check_resolved_layer(layer, where, written=tree.get(part), callsite=after)
        except LayerFieldError as exc:
            raise _refused(str(exc), _field_origin(template, (part, *exc.field))) from None
    selection = call_time.selection
    snapshot = selection.snapshot
    try:
        config = resolve_agent_config(
            merge_llm_config(node_layer, merge_llm_config(flow_layer, call_time.base)),
            defaults=snapshot.defaults,
            provider_models=snapshot.provider_models,
            stray_model=snapshot.stray_model,
            path=path,
            label=label,
        )
        canonical = selection.bind(config)
        require_concrete_selection(canonical, path=path, label=label)
    except ValueError as exc:
        raise _refused(str(exc), None) from None
    probe = invocation_config(canonical, credentials)
    try:
        check_model_config(probe)
    except ValueError as exc:
        # An `AuthError` (the host key or a header missing) propagates: it is the host's
        # to fix by resuming with the key, not a failure of this node.
        block = refused_block(probe)
        origin = None
        if block:
            origin = _field_origin(template, ("node", block[1:])) or _field_origin(
                template, ("flow", block[1:])
            )
        raise _refused(block_problem(site, block[1:], str(exc), callsite), origin) from None
    return canonical


def _refused(message: str, origin: Optional[OriginRef]) -> BindingError:
    """Return the ``contract.input`` failure of a resolved config refused at a field."""
    return BindingError(message, failure_code="contract.input", origin=origin)


def _field_origin(template: Binding, field: ValuePath) -> Optional[OriginRef]:
    """Return the authored site of ``field`` (under ``node`` or ``flow``), or its nearest.

    The deepest written key on the way down that has a site is used; a part written
    as one whole-record value is that value's site.
    """
    source = template.source
    for end in range(len(field), 1, -1):
        origin = source.origin_at(field[:end])
        if origin is not None:
            return origin
    part = source.value.get(field[0]) if field else None
    return part.origin if isinstance(part, Binding) else None


def _refusal(written: Any, resolved: Mapping[str, Any]) -> Optional[tuple[ValuePath, str]]:
    """Return the field `refused_resolved_field` refuses and why, or `None`."""
    if "_connection" in resolved:
        return ("_connection",), "internal to recorded runs"
    literal = _literal_fields(written)
    for field in resolved:
        if host_only_field_error({field: None}) is None:
            continue
        if literal is None or field not in literal:
            return (field,), HOST_ONLY_FROM_VALUE
    message = host_only_field_error(literal or {})
    if message is None:
        return None
    return (), message


def _literal_fields(written: Any) -> Optional[dict[str, Any]]:
    """Return a written layer's literal top-level fields, or `None` for a whole-record value.

    A field holding a value anywhere inside it is left out; a tagged literal counts
    as the literal it tags.
    """
    tree = _unwrapped(written)
    if is_value(tree):
        return None
    if type(tree) is not dict:
        return {}
    return {key: _unwrapped(item) for key, item in tree.items() if not _dynamic(item)}


def _unwrapped(node: Any) -> Any:
    """Look through templates and tagged literals to the tree they hold."""
    while True:
        if isinstance(node, ValueTemplate):
            node = node.value
        elif isinstance(node, Binding) and not is_value(node):
            node = node.source
        else:
            return node


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
