"""Resolve and replay effective LLM configuration over the static call tree.

Resolution is a compile-layer operation: callers provide both the outermost authored
layer and the host's defaults, which may be empty; there is no built-in provider or
model.  An agent that no layer gives a provider and model fails, naming its static
path.  The resulting map is therefore complete and can be recorded and replayed
without consulting process settings again.

Static paths join local author node ids with ``/``.  The Compose parser reserves that
separator in node ids, so paths such as ``review/summarize/model`` are unambiguous.  A
child flow is deep-copied at every CALL/MAP/LOOP callsite before either resolution or
replay.  A root-level copy alone preserves aliases between shared descendant defs and
would let one parent's cascade overwrite another's.

Layer: compile — imports nodes and LLM configuration primitives; never runtime.
Defaults are injected by the host.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from pydantic import SecretStr

from yian.diagnostics import OriginRef

from yian.llm_clients.config import HostConnection, merge_llm_config
from yian.llm_clients.connection import (
    HOST_KEY_MARKER,
    ConnectionSelection,
    ConnectionSnapshot,
    record_selection,
    restore_selection,
)
from yian.llm_clients.layers import (
    require_concrete_selection,
    resolve_agent_config,
    validate_layer,
)
from yian.nodes.agent import AgentNode
from yian.nodes.agent.llm_values import (
    LLM_CONFIG_KEY,
    CallTimeLLM,
    Nullable,
    agent_label,
    check_resolved_layer,
    contract_nullable,
    contributing,
    field_where,
    keep_values,
    layer_refs,
    reads_only_inputs,
    resolve_layer,
    resolve_layers,
    types_nullable,
    value_leaves,
)
from yian.nodes.base import Node
from yian.nodes.binding import BindingError, bind_params
from yian.nodes.contracts import Binding, Parameter, ValuePath, ValueTemplate
from yian.typesys.pool import VariablePool


def resolve_llm_cascade(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
    provider_models: dict[str, str] | None = None,
    stray_model: bool = False,
) -> dict[str, dict[str, Any]]:
    """Bake and return every AGENT node's complete effective configuration.

    This is `resolve_run_llm` for a run given no inputs; its ``effective`` map is
    returned.

    Args:
        flow (`CompiledFlow`):
            The flow whose complete static call tree is resolved in place.
        parent_config (`dict[str, Any]`):
            The outermost gap-fill layer (the CLI/host layer).
        defaults (`dict[str, Any]`):
            Lowest-precedence host defaults, possibly empty.
        provider_models (`dict[str, str] | None`):
            Saved models by provider.
        stray_model (`bool`, *optional*, defaults to `False`):
            Whether ``YIAN_DEFAULT_MODEL`` was set without a default provider.

    Returns:
        `dict[str, dict[str, Any]]`:
            Complete effective configurations keyed by unambiguous static agent path.

    Raises:
        `ValueError`: As `resolve_run_llm` raises it.
    """
    return resolve_run_llm(
        flow,
        parent_config,
        defaults=defaults,
        provider_models=provider_models,
        stray_model=stray_model,
    ).effective


@dataclass(frozen=True)
class RunLLM:
    """The configurations a run start resolves for its agents.

    Attributes:
        effective (`dict[str, dict[str, Any]]`): Each agent resolved at run start, its
            complete effective configuration keyed by static agent path.
        call_time (`dict[str, dict[str, Any]]`): Each agent whose configuration is
            resolved only when it runs, keyed by static agent path: its base, the
            layers below its node and own-flow layers (`CallTimeLLM.base`).
    """

    effective: dict[str, dict[str, Any]]
    call_time: dict[str, dict[str, Any]]


def resolve_run_llm(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
    provider_models: dict[str, str] | None = None,
    stray_model: bool = False,
    run_inputs: Mapping[str, Any] | None = None,
) -> RunLLM:
    """Bake every AGENT node's complete effective configuration at run start.

    The normal precedence is node, current flow, enclosing flows, caller layer, then
    ``defaults``. An AGENT node with ``llm_inherit=False`` skips the flow/enclosing/caller
    layers but still receives the defaults. An unset model comes from
    ``provider_models`` for the agent's provider.

    Values (`${...}`) in the run's flow that read only its inputs are resolved here,
    in that flow's layer and in the layers of its agents. Each input such a value reads
    is checked first, as START checks it: present unless optional, its default when
    omitted, and its contract. A value resolving to null leaves its field unset. A
    flow layer passes down only its literal and resolved fields; Compose has refused
    any other value an inheriting agent could read (`find_inherited_llm_value`).
    Any other agent whose layers carry values resolves its configuration only when
    it runs; run start fixes its base: the enclosing flows' layers and the
    caller layer, merged, or nothing under ``inherit: false``. Its node and own-flow
    layers ride its reserved ``__llm_config`` binding.
    Every returned value and every node-owned value is an independent deep copy.

    Args:
        flow (`CompiledFlow`):
            The flow whose complete static call tree is resolved in place.
        parent_config (`dict[str, Any]`):
            The outermost gap-fill layer.  At the root this is the CLI/host layer; in a
            child it is the already-accumulated enclosing flow layer.
        defaults (`dict[str, Any]`):
            Lowest-precedence host defaults, possibly empty. ``model`` applies only
            to the default ``provider``. No environment is read.
        provider_models (`dict[str, str] | None`):
            Saved models by provider. An authored model still wins per field; a
            provider with no saved model needs an authored one.
        stray_model (`bool`, *optional*, defaults to `False`):
            Whether ``YIAN_DEFAULT_MODEL`` was set without a default provider; an
            agent missing its provider or model then says so.
        run_inputs (`Mapping[str, Any] | None`, *optional*, defaults to `None`):
            The run's supplied inputs, coerced at the run boundary; `None` is none.

    Returns:
        `RunLLM`: The resolved configurations.

    Raises:
        `ValueError`:
            If a provider or model selection is invalid, an agent has no provider or
            model, an input a value reads is missing or breaks its contract, a value
            fails to resolve, or a child flow's agent would inherit an `llm_config`
            value read only when its flow runs (`find_inherited_llm_value`).
    """
    inherited = find_inherited_llm_value(flow)
    if inherited is not None:
        where = (
            "the run's flow" if not inherited.owner
            else f"the flow node {_path_key(inherited.owner)!r} calls"
        )
        raise ValueError(
            f"node {_path_key(inherited.agent)!r} would inherit "
            f"{field_label(inherited.field)} from {where}, a value read only when that flow "
            f"runs; {inherited_value_fix(inherited.field)}"
        )
    validate_layer(parent_config)
    validate_layer(defaults)
    effective: dict[str, dict[str, Any]] = {}
    call_time: dict[str, dict[str, Any]] = {}
    _resolve_flow(
        flow,
        copy.deepcopy(parent_config),
        defaults=copy.deepcopy(defaults),
        provider_models=dict(provider_models or {}),
        stray_model=stray_model,
        path=(),
        effective=effective,
        call_time=call_time,
        inputs=_RunInputs(flow, run_inputs or {}),
    )
    return RunLLM(effective, call_time)


class _RunInputs:
    """The run's inputs that run-start values read, each checked once as START checks it."""

    def __init__(self, flow: Any, supplied: Mapping[str, Any]) -> None:
        """Hold ``flow``'s declared inputs and the supplied values, none checked yet."""
        self._start_id = flow.start_id
        self._params: dict[str, Parameter] = {
            parameter.name: parameter for parameter in flow.nodes[flow.start_id].params
        }
        self._supplied = dict(supplied)
        self._checked: dict[str, Any] = {}

    def pool_for(self, layer: Any, where: Callable[[ValuePath], str]) -> VariablePool:
        """Check each input the values in ``layer`` read; return a pool holding them.

        Args:
            layer (`Any`): A layer whose values read only ``input.*``.
            where (`Callable[[ValuePath], str]`): Names a value's field in a message.

        Returns:
            `VariablePool`: A pool whose START record holds every input checked so far;
                an omitted optional input without a default is null.

        Raises:
            `ValueError`: An input is missing or breaks its contract; the message names
                the field and the input.
        """
        for path, leaf in value_leaves(layer).items():
            for name in self._read_names(leaf):
                if name not in self._checked:
                    self._checked[name] = self._check(name, where(path))
        pool = VariablePool(start_id=self._start_id)
        pool.set(self._start_id, copy.deepcopy(self._checked))
        return pool

    def empty_pool(self) -> VariablePool:
        """Return a pool with no inputs, for a layer whose values are all dropped."""
        return VariablePool(start_id=self._start_id)

    def _read_names(self, leaf: Any) -> list[str]:
        """Return the names of the inputs a value reads; ``${input}`` reads them all."""
        names: list[str] = []
        for ref in layer_refs(leaf):
            parts = ref.split(".")
            names.extend(self._params if len(parts) == 1 else parts[1:2])
        return list(dict.fromkeys(names))

    def _check(self, name: str, where: str) -> Any:
        """Bind one input as START binds it, and return its value."""
        parameter = self._params.get(name)
        if parameter is None:
            raise ValueError(f"{where}: input {name!r} is not declared")
        present = name in self._supplied
        if not present and parameter.required:
            raise ValueError(f"{where}: input {name!r} is required and was not given")
        pool = VariablePool(start_id=self._start_id)
        pool.set(self._start_id, {name: self._supplied[name]} if present else {})
        wiring = {name: f"${{input.{name}}}"} if present else {}
        try:
            record = bind_params([parameter], wiring, pool)
        except BindingError as exc:
            raise ValueError(f"{where}: {exc}") from None
        return record.get(name)


@dataclass(frozen=True)
class InheritedValue:
    """An `llm_config` value in an enclosing flow's layer that a child flow's agent inherits.

    Only the run's flow layer may pass a value down, and only one that reads just the
    run's inputs, which are known at run start; any other value is read only when its
    flow runs, after the child's agents have their config.

    Attributes:
        owner (`tuple[str, ...]`): The static path of the callsite whose flow holds the
            layer; empty for the run's flow.
        callsite (`Node | None`): The CALL, MAP or LOOP node at ``owner``; `None` for
            the run's flow.
        field (`ValuePath`): The value's field path in its layer; empty for a whole
            record.
        origin (`OriginRef | None`): The value's authored site, when known.
        agent (`tuple[str, ...]`): The static path of the agent that inherits it.
    """

    owner: tuple[str, ...]
    callsite: Optional[Node]
    field: ValuePath
    origin: Optional[OriginRef]
    agent: tuple[str, ...]


def find_inherited_llm_value(flow: Any) -> Optional[InheritedValue]:
    """Return the first `llm_config` value a child flow's agent would inherit, if any.

    The walk follows the cascade: a child's agent inherits every enclosing flow's layer
    unless it has ``inherit: false``. A value reaches it unless a layer closer to the
    agent (its own, its flow's, a nearer enclosing flow's) always sets that field. A
    value answers from the static type Compose proved for it, else from its tag.

    Args:
        flow (`CompiledFlow`): The run's flow; its static call tree is walked, unchanged.

    Returns:
        `InheritedValue | None`: The first such value in static tree order, or `None`.
    """
    return _find_inherited(flow, (), None, [])


def field_label(field: ValuePath) -> str:
    """Name an `llm_config` field path in a message, such as ``llm_config.model``."""
    return "llm_config" + "".join(f".{part}" for part in field)


def inherited_value_fix(field: ValuePath) -> str:
    """Tell the author how to pass an inherited value to a child flow instead."""
    name = field[-1] if field else "llm_config"
    return (
        f"pass it to the child as an input and write `${{input.{name}}}` in the child's "
        "llm_config"
    )


@dataclass(frozen=True)
class _Enclosing:
    """One enclosing flow's layer on the way down to a child flow."""

    layer: Any
    nullable: Nullable
    owner: tuple[str, ...]
    callsite: Optional[Node]


def _flow_nullable(flow: Any, layer: Any) -> Nullable:
    """Answer from the static types Compose proved for a flow layer's values, else tags."""
    types = getattr(flow, "flow_llm_value_types", None)
    return types_nullable(layer, types) if types else contract_nullable


def _find_inherited(
    flow: Any,
    path: tuple[str, ...],
    callsite: Optional[Node],
    enclosing: list[_Enclosing],
) -> Optional[InheritedValue]:
    """Search ``flow``, reached at callsite ``path`` under ``enclosing``, and its children."""
    layer = getattr(flow, "flow_llm_config", None) or {}
    own = _Enclosing(layer, _flow_nullable(flow, layer), path, callsite)
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode) and node.llm_inherit and enclosing:
            found = _inherited_by(node, own, enclosing, node_path)
            if found is not None:
                return found
        child = getattr(node, "child", None)
        if child is not None:
            found = _find_inherited(child, node_path, node, [*enclosing, own])
            if found is not None:
                return found
    return None


def _inherited_by(
    node: AgentNode, own: _Enclosing, enclosing: list[_Enclosing], agent: tuple[str, ...]
) -> Optional[InheritedValue]:
    """Return the first enclosing value that can reach ``node``, innermost layer first."""
    higher: list[tuple[Any, Nullable]] = [
        (node.own_llm_config, node.llm_nullable()),
        (own.layer, own.nullable),
    ]
    for outer in reversed(enclosing):
        remaining = outer.layer
        for layer, nullable in higher:
            remaining = contributing(remaining, layer, nullable)
        for field, leaf in value_leaves(remaining).items():
            if not outer.owner and reads_only_inputs(leaf):
                continue  # the run's inputs are known at run start
            return InheritedValue(
                owner=outer.owner,
                callsite=outer.callsite,
                field=field,
                origin=_value_origin(outer.layer, field, leaf),
                agent=agent,
            )
        higher.append((outer.layer, outer.nullable))
    return None


def _value_origin(layer: Any, field: ValuePath, leaf: Any) -> Optional[OriginRef]:
    """Return the authored site of the value at ``field`` in ``layer``."""
    if isinstance(layer, Binding):
        return layer.origin
    if isinstance(leaf, Binding) and leaf.origin is not None:
        return leaf.origin
    return layer.origin_at(field) if isinstance(layer, ValueTemplate) else None


def apply_effective_llm_map(
    flow: Any,
    effective: dict[str, dict[str, Any]],
    call_time: Mapping[str, Any] | None = None,
) -> None:
    """Apply one recorded complete effective map, and call-time agents, to a static flow tree.

    The effective map and the call-time ``agents`` together must name exactly the
    flow's static agent paths, each once. Validation finishes before mutation.
    Applying deep-copies each config onto its agent's ``llm_config``, gives each
    call-time agent a `CallTimeLLM` with its base and the restored selection (its
    ``llm_config`` stays unset until it runs), and de-aliases every static child
    callsite, without importing settings or reading the environment.

    Args:
        flow (`CompiledFlow`):
            A clean compiled flow to mutate in place.
        effective (`dict[str, dict[str, Any]]`):
            Recorded complete configurations keyed by static agent path.
        call_time (`Mapping[str, Any] | None`, *optional*, defaults to `None`):
            The recorded ``RunStarted.call_time_llm``: ``agents``, the base of each
            agent whose config resolves only when it runs, keyed by static agent path,
            and ``selection``, the `record_selection` record they bind from. `None`
            when there are none.

    Raises:
        `ValueError`:
            If paths are missing, extra or in both maps, a config lacks provider/model,
            a call-time path names an agent with no ``llm_config`` value, a base is not
            a valid layer, or the selection is invalid.
        `TypeError`:
            If a map or one of its entries is not a dictionary.
    """
    if not isinstance(effective, dict):
        raise TypeError("effective LLM map must be a dict keyed by static agent path")
    selection = None
    if call_time is None:
        call_time = {}
    else:
        if not isinstance(call_time, Mapping):
            raise TypeError("call-time LLM record must be a dict of selection and agents")
        if set(call_time) != {"selection", "agents"}:
            raise ValueError("call-time LLM record must hold exactly selection and agents")
        selection = restore_selection(call_time["selection"])
        call_time = call_time["agents"]
        if not isinstance(call_time, dict):
            raise TypeError("call-time LLM agents must be a dict keyed by static agent path")

    expected = set(_static_agent_paths(flow))
    supplied = set(effective) | set(call_time)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    both = sorted(set(effective) & set(call_time))
    if missing or extra or both:
        details = []
        if missing:
            details.append(f"missing paths: {missing}")
        if extra:
            details.append(f"extra paths: {extra}")
        if both:
            details.append(f"paths in both maps: {both}")
        raise ValueError("effective LLM map path mismatch; " + "; ".join(details))

    labels = static_agent_labels(flow)
    for path in sorted(effective):
        config = effective[path]
        if not isinstance(config, dict):
            subject = labels.get(path) or repr(path)
            raise TypeError(f"effective LLM config for {subject} must be a dict")
        require_concrete_selection(config, path=path, label=labels.get(path))
    carrying = _carrying_agent_paths(flow)
    for path in sorted(call_time):
        subject = labels.get(path) or repr(path)
        if path not in carrying:
            raise ValueError(
                f"call-time LLM base for {subject}: the agent has no llm_config value"
            )
        base = call_time[path]
        if not isinstance(base, dict):
            raise TypeError(f"call-time LLM base for {subject} must be a dict")
        try:
            validate_layer(base)
        except ValueError as exc:
            raise ValueError(f"call-time LLM base for {subject}: {exc}") from None

    _apply_flow(flow, effective, call_time, selection, path=())


def _resolve_flow(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
    provider_models: dict[str, str],
    stray_model: bool,
    path: tuple[str, ...],
    effective: dict[str, dict[str, Any]],
    call_time: dict[str, dict[str, Any]],
    inputs: _RunInputs,
) -> None:
    """Resolve one flow scope recursively into ``effective`` and ``call_time``."""
    flow_layer = merge_llm_config(
        _run_start_flow_layer(flow, root=not path, inputs=inputs),
        parent_config,
    )

    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            path_key = _path_key(node_path)
            label = agent_label(node.llm_config_label, _path_key(path))
            carried = flow.wiring.get(node_id, {}).get(LLM_CONFIG_KEY)
            if carried is not None and (path or not reads_only_inputs(carried)):
                # Its own layers resolve when it runs; only the layers below them are
                # known now.
                call_time[path_key] = copy.deepcopy(parent_config) if node.llm_inherit else {}
                node.llm_config = None
                continue
            if carried is None:
                own = _literal_layer(node.own_llm_config, inputs)
                validate_layer(own)
                base = {} if not node.llm_inherit else flow_layer
                authored = merge_llm_config(own, base)
            else:
                authored = _resolve_carried(
                    carried, node, parent_config, path=path_key, label=label, inputs=inputs,
                )
            config = resolve_agent_config(
                authored,
                defaults=defaults,
                provider_models=provider_models,
                stray_model=stray_model,
                path=path_key,
                label=label,
            )
            node.llm_config = copy.deepcopy(config)
            effective[path_key] = copy.deepcopy(config)

        child = getattr(node, "child", None)
        if child is not None:
            # Deep-copy at EACH callsite.  A deepcopy of only the root preserves shared
            # descendant aliases and lets differently configured parents interfere.
            node.child = copy.deepcopy(child)
            _resolve_flow(
                node.child,
                flow_layer,
                defaults=defaults,
                provider_models=provider_models,
                stray_model=stray_model,
                path=node_path,
                effective=effective,
                call_time=call_time,
                inputs=inputs,
            )


_FLOW_SUBJECT = "flow llm_config"


def _run_start_flow_layer(flow: Any, *, root: bool, inputs: _RunInputs) -> dict[str, Any]:
    """Return a flow's layer as its agents inherit it at run start.

    The run's flow keeps its values that read only its inputs, resolved; every other
    value is dropped with its field, since no agent that inherits the layer statically
    can read it.
    """
    authored = getattr(flow, "flow_llm_config", None) or {}
    if isinstance(authored, dict):
        layer = copy.deepcopy(authored)
    else:
        known = keep_values(authored, reads_only_inputs if root else lambda _leaf: False)
        pool = (
            inputs.pool_for(known, lambda field: field_where(_FLOW_SUBJECT, field))
            if root else inputs.empty_pool()
        )
        layer = _resolved(lambda: resolve_layer(known, pool), _FLOW_SUBJECT)
        if root:
            check_resolved_layer(layer, _FLOW_SUBJECT, written=known)
    validate_layer(layer)
    return layer


def _literal_layer(layer: Any, inputs: _RunInputs) -> dict[str, Any]:
    """Return an agent's own layer that holds no value as a plain dict."""
    if isinstance(layer, dict):
        return copy.deepcopy(layer)
    return resolve_layer(layer, inputs.empty_pool())


def _resolve_carried(
    carried: Binding,
    node: AgentNode,
    parent_config: dict[str, Any],
    *,
    path: str,
    label: Optional[str],
    inputs: _RunInputs,
) -> dict[str, Any]:
    """Resolve an agent's reserved ``__llm_config`` layers and merge them over the caller's.

    The agent is in the run's flow, and its values read just the run's inputs.
    """
    subject = label or f"node {path!r} llm_config"

    def where(field: ValuePath) -> str:
        part, *rest = field
        return field_where(subject if part == "node" else _FLOW_SUBJECT, tuple(rest))

    pool = inputs.pool_for(carried.source, where)
    node_layer, flow_layer = _resolved(lambda: resolve_layers(carried, pool), subject)
    written = carried.source.value
    check_resolved_layer(node_layer, subject, written=written.get("node"))
    check_resolved_layer(flow_layer, _FLOW_SUBJECT, written=written.get("flow"))
    if not node.llm_inherit:
        return node_layer
    return merge_llm_config(node_layer, merge_llm_config(flow_layer, parent_config))


def known_provider(
    node: AgentNode, flow: Any, base: dict[str, Any], defaults: dict[str, Any]
) -> Optional[str]:
    """Return a call-time agent's provider when run start already knows it.

    It is known when a literal sets it and no value in a higher layer can override
    it: the first provider set walking down the node layer, the own-flow layer (unless
    ``inherit: false``), ``base`` and ``defaults``, with no provider or whole-record
    value above it.

    Args:
        node (`AgentNode`): The agent.
        flow (`CompiledFlow`): The flow ``node`` is in, whose layer is its own-flow layer.
        base (`dict[str, Any]`): The agent's call-time base.
        defaults (`dict[str, Any]`): The host defaults.

    Returns:
        `str | None`: The provider, or `None` when a value may set it or nothing does.
    """
    for layer in _own_layers(node, flow):
        if any(path[:1] in ((), ("provider",)) for path in value_leaves(layer)):
            return None
        literal = _literal_fields(layer, flow)
        if literal.get("provider") is not None:
            return literal["provider"]
    return base.get("provider") or defaults.get("provider") or None


def known_base_url(node: AgentNode, flow: Any, base: dict[str, Any]) -> Optional[str]:
    """Return the ``base_url`` a call-time agent's layers set, when one does.

    ``base_url`` is never a value, so the first layer setting it, walking down the
    node layer, the own-flow layer (unless ``inherit: false``) and ``base``, gives it.

    Args:
        node (`AgentNode`): The agent.
        flow (`CompiledFlow`): The flow ``node`` is in, whose layer is its own-flow layer.
        base (`dict[str, Any]`): The agent's call-time base.

    Returns:
        `str | None`: The endpoint, or `None` when no layer sets one.
    """
    for layer in _own_layers(node, flow):
        literal = _literal_fields(layer, flow)
        if literal.get("base_url") is not None:
            return literal["base_url"]
    return base.get("base_url")


def _own_layers(node: AgentNode, flow: Any) -> list[Any]:
    """Return a call-time agent's node layer and, unless it opts out, its own-flow layer."""
    layers = [node.own_llm_config]
    if node.llm_inherit:
        layers.append(getattr(flow, "flow_llm_config", None) or {})
    return layers


def _literal_fields(layer: Any, flow: Any) -> dict[str, Any]:
    """Return a layer's literal fields, dropping every value in it."""
    if isinstance(layer, dict):
        return layer
    pool = VariablePool(start_id=flow.start_id)
    return resolve_layer(keep_values(layer, lambda _leaf: False), pool)


def known_call_time_providers(
    flow: Any, call_time: Mapping[str, dict[str, Any]], defaults: dict[str, Any]
) -> dict[str, Optional[str]]:
    """Return each call-time agent's `known_provider`, keyed by static agent path.

    Args:
        flow (`CompiledFlow`): The run's flow; its static call tree is walked, unchanged.
        call_time (`Mapping[str, dict[str, Any]]`): Each call-time agent's base, keyed by
            static agent path.
        defaults (`dict[str, Any]`): The host defaults.

    Returns:
        `dict[str, str | None]`: The provider, or `None` when not known, per path in
            ``call_time``.
    """
    return {
        path: known_provider(node, owner, call_time[path], defaults)
        for path, node, owner in _static_agents(flow)
        if path in call_time
    }


def known_call_time_base_urls(
    flow: Any, call_time: Mapping[str, dict[str, Any]]
) -> dict[str, str]:
    """Return the `known_base_url` of each call-time agent that has one.

    Args:
        flow (`CompiledFlow`): The run's flow; its static call tree is walked, unchanged.
        call_time (`Mapping[str, dict[str, Any]]`): Each call-time agent's base, keyed by
            static agent path.

    Returns:
        `dict[str, str]`: The endpoint, keyed by static agent path, for each agent in
            ``call_time`` whose layers set one.
    """
    urls: dict[str, str] = {}
    for path, node, owner in _static_agents(flow):
        if path in call_time:
            url = known_base_url(node, owner, call_time[path])
            if url is not None:
                urls[path] = url
    return urls


def in_flight_llm_configs(
    nodes: Mapping[str, Any], steps: Mapping[str, tuple[dict[str, Any], ...]],
) -> dict[str, dict[str, Any]]:
    """Return the config each started call-time agent recorded in its steps.

    An agent that has not started records nothing; its config is checked when it
    resolves.

    Args:
        nodes (`Mapping[str, Any]`): The run's live nodes, keyed by node id.
        steps (`Mapping[str, tuple[dict[str, Any], ...]]`): The steps of each unsettled
            node, keyed by node id.

    Returns:
        `dict[str, dict[str, Any]]`: Each recorded config, with no key or header value,
            keyed by node id.
    """
    configs: dict[str, dict[str, Any]] = {}
    for node_id, node_steps in steps.items():
        node = nodes.get(node_id)
        if isinstance(node, AgentNode) and node.call_time is not None:
            recorded = node.recorded_llm_config(node_steps)
            if recorded is not None:
                configs[node_id] = recorded
    return configs


def call_time_selection(
    snapshot: ConnectionSnapshot,
    *,
    known: Mapping[str, Optional[str]],
    host_key_provider: str | None = None,
    host_base_url: str | None = None,
    host_settings: HostConnection | None = None,
) -> dict[str, Any]:
    """Record the connection selection the call-time agents may need.

    When every call-time agent's provider is known (`known_provider`), only those
    providers' connections are recorded; otherwise every connection is.

    Args:
        snapshot (`ConnectionSnapshot`): The run's selection snapshot.
        known (`Mapping[str, str | None]`): Each call-time agent's
            `known_call_time_providers` entry, keyed by static agent path.
        host_key_provider (`str | None`, *optional*, defaults to `None`): The host key's
            provider, if the host gave one.
        host_base_url (`str | None`, *optional*, defaults to `None`): Where the host key
            is sent, if the host gave it.
        host_settings (`HostConnection | None`, *optional*, defaults to `None`): The
            host's connection; only its nonsecret settings are recorded.

    Returns:
        `dict[str, Any]`: The `record_selection` record.
    """
    providers = None if None in known.values() else {p for p in known.values() if p}
    return record_selection(
        snapshot,
        providers=providers,
        host_key_provider=host_key_provider,
        host_base_url=host_base_url,
        host_settings=host_settings,
    )


def value_config_subjects(flow: Any) -> dict[str, str]:
    """Name the ``llm_config`` of each run-start agent whose config takes values.

    A run-start check of such an agent's effective config names the field at fault
    under this subject. An agent of the run's flow whose layers carry values is named
    by its own layer. When the run's flow layer holds values, every other agent that
    inherits it, at any depth, is named with ``(flow llm_config)``.

    Args:
        flow (`CompiledFlow`): The run's flow.

    Returns:
        `dict[str, str]`: The subject, such as ``node 'chat' llm_config``,
            ``Agent 'c/a' (flow llm_config)`` or a synthesized agent's label, keyed by
            static agent path.
    """
    flow_values = bool(value_leaves(getattr(flow, "flow_llm_config", None)))
    subjects: dict[str, str] = {}
    for path, node, owner in _static_agents(flow):
        *parent, node_id = path.split("/")
        if LLM_CONFIG_KEY in owner.wiring.get(node_id, {}):
            if owner is flow:
                label = agent_label(node.llm_config_label, "")
                subjects[path] = label or f"node {node_id!r} llm_config"
        elif flow_values and node.llm_inherit:
            agent = agent_label(node.llm_config_label, "/".join(parent)) or f"Agent {path!r}"
            subjects[path] = f"{agent} ({_FLOW_SUBJECT})"
    return subjects


def _resolved(resolve: Callable[[], Any], subject: str) -> Any:
    """Run one resolution, reporting a binding failure as a `ValueError` naming ``subject``."""
    try:
        return resolve()
    except BindingError as exc:
        raise ValueError(f"{subject}: {exc}") from None


def _static_agent_paths(
    flow: Any,
    path: tuple[str, ...] = (),
) -> list[str]:
    """Return all agent paths without mutating or de-aliasing the graph."""
    paths: list[str] = []
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            paths.append(_path_key(node_path))
        child = getattr(node, "child", None)
        if child is not None:
            paths.extend(_static_agent_paths(child, node_path))
    return paths


def _static_agents(flow: Any, path: tuple[str, ...] = ()) -> list[tuple[str, AgentNode, Any]]:
    """Return each static agent's path, node and the flow it is in, without mutating."""
    agents: list[tuple[str, AgentNode, Any]] = []
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            agents.append((_path_key(node_path), node, flow))
        child = getattr(node, "child", None)
        if child is not None:
            agents.extend(_static_agents(child, node_path))
    return agents


def _carrying_agent_paths(flow: Any) -> set[str]:
    """Return the static paths of the agents whose layers ride ``__llm_config``."""
    return {
        path
        for path, _node, owner in _static_agents(flow)
        if LLM_CONFIG_KEY in owner.wiring.get(path.rsplit("/", 1)[-1], {})
    }


def static_agent_labels(flow: Any) -> dict[str, str]:
    """Map each synthesized agent's static path to the authored site of its ``llm_config``.

    Walks the static tree like the effective map does, without mutating it. An
    authored agent has no label and is absent, so its messages keep naming its path.

    Args:
        flow (`CompiledFlow`):
            The flow whose static agents to label.

    Returns:
        `dict[str, str]`: The label of each labelled agent, keyed by static agent path.
    """
    labels: dict[str, str] = {}
    _collect_agent_labels(flow, (), labels)
    return labels


def _collect_agent_labels(flow: Any, path: tuple[str, ...], labels: dict[str, str]) -> None:
    """Add the labels of ``flow``'s static agents, at callsite ``path``, to ``labels``."""
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            label = agent_label(node.llm_config_label, _path_key(path))
            if label is not None:
                labels[_path_key(node_path)] = label
        child = getattr(node, "child", None)
        if child is not None:
            _collect_agent_labels(child, node_path, labels)


def _apply_flow(
    flow: Any,
    effective: dict[str, dict[str, Any]],
    call_time: dict[str, dict[str, Any]],
    selection: Optional[ConnectionSelection],
    *,
    path: tuple[str, ...],
) -> None:
    """Apply already-validated configs and bases, and de-alias children recursively."""
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            path_key = _path_key(node_path)
            if path_key in call_time:
                if selection is None:
                    raise ValueError(
                        f"call-time agent {path_key!r} has no recorded connection selection"
                    )
                node.llm_config = None
                node.call_time = CallTimeLLM(
                    path_key, copy.deepcopy(call_time[path_key]), copy.deepcopy(selection)
                )
            else:
                node.llm_config = copy.deepcopy(effective[path_key])
                node.call_time = None

        child = getattr(node, "child", None)
        if child is not None:
            node.child = copy.deepcopy(child)
            _apply_flow(node.child, effective, call_time, selection, path=node_path)


def bind_host_credentials(
    flow: Any, api_key: SecretStr, headers: dict[str, SecretStr] | None = None,
) -> None:
    """Give every static AGENT marked for a host key that key and its header values.

    `ConnectionSnapshot.bind` marks the agents of the key's provider in the
    effective map. The key and header values go on each marked node's run-local
    ``llm_config`` only, never into the map that ``RunStarted`` records. A call-time
    agent whose recorded selection names the key's provider gets them as its
    run-local ``host_credentials``, never merged into a config; it uses them only if
    it resolves to that provider. Call it after `apply_effective_llm_map`, which has
    already de-aliased every child callsite.

    Args:
        flow (`CompiledFlow`):
            A flow whose static agents already carry their effective configs.
        api_key (`SecretStr`):
            The host-supplied key.
        headers (`dict[str, SecretStr] | None`, *optional*, defaults to `None`):
            The header values sent with that key; `None` or empty binds none.
    """
    for node in flow.nodes.values():
        if isinstance(node, AgentNode) and node.call_time is not None:
            if node.call_time.selection.host_key_provider is not None:
                node.host_credentials = HostConnection(
                    api_key=api_key, headers=dict(headers) if headers else None
                )
        elif isinstance(node, AgentNode) and node.llm_config.get(HOST_KEY_MARKER) is True:
            node.llm_config["api_key"] = api_key
            if headers:
                node.llm_config["headers"] = dict(headers)
        child = getattr(node, "child", None)
        if child is not None:
            bind_host_credentials(child, api_key, headers)


def host_key_providers(effective: dict[str, dict[str, Any]]) -> set[str]:
    """Return the providers whose agents a recorded effective map marks for a host key."""
    return {
        config["provider"] for config in effective.values() if config.get(HOST_KEY_MARKER) is True
    }


def _path_key(path: tuple[str, ...]) -> str:
    """Encode one non-empty static author path."""
    return "/".join(path)
