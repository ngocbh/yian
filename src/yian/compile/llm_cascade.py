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
from typing import Any, Optional

from pydantic import SecretStr

from yian.diagnostics import OriginRef

from yian.llm_clients.config import merge_llm_config
from yian.llm_clients.connection import HOST_KEY_MARKER
from yian.llm_clients.layers import (
    require_concrete_selection,
    resolve_agent_config,
    validate_layer,
)
from yian.nodes.agent import AgentNode
from yian.nodes.agent.llm_values import (
    Nullable,
    contract_nullable,
    contributing,
    reads_only_inputs,
    value_leaves,
)
from yian.nodes.base import Node
from yian.nodes.contracts import Binding, ValuePath, ValueTemplate


def resolve_llm_cascade(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
    provider_models: dict[str, str] | None = None,
    stray_model: bool = False,
) -> dict[str, dict[str, Any]]:
    """Bake and return every AGENT node's complete effective configuration.

    The normal precedence is node, current flow, enclosing flows, caller layer, then
    ``defaults``. An AGENT node with ``llm_inherit=False`` skips the flow/enclosing/caller
    layers but still receives the defaults. An unset model comes from
    ``provider_models`` for the agent's provider.
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

    Returns:
        `dict[str, dict[str, Any]]`:
            Complete effective configurations keyed by unambiguous static agent path.

    Raises:
        `ValueError`:
            If a provider or model selection is invalid, an agent has no provider or
            model, or a child flow's agent would inherit an `llm_config` value read
            only when its flow runs (`find_inherited_llm_value`).
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
    _resolve_flow(
        flow,
        copy.deepcopy(parent_config),
        defaults=copy.deepcopy(defaults),
        provider_models=dict(provider_models or {}),
        stray_model=stray_model,
        path=(),
        effective=effective,
    )
    return effective


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
    agent (its own, its flow's, a nearer enclosing flow's) always sets that field; the
    agent's own values answer from their static types, others from their tags.

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
    owner: tuple[str, ...]
    callsite: Optional[Node]


def _find_inherited(
    flow: Any,
    path: tuple[str, ...],
    callsite: Optional[Node],
    enclosing: list[_Enclosing],
) -> Optional[InheritedValue]:
    """Search ``flow``, reached at callsite ``path`` under ``enclosing``, and its children."""
    own = getattr(flow, "flow_llm_config", None) or {}
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode) and node.llm_inherit and enclosing:
            found = _inherited_by(node, own, enclosing, node_path)
            if found is not None:
                return found
        child = getattr(node, "child", None)
        if child is not None:
            found = _find_inherited(
                child, node_path, node, [*enclosing, _Enclosing(own, path, callsite)]
            )
            if found is not None:
                return found
    return None


def _inherited_by(
    node: AgentNode, own: Any, enclosing: list[_Enclosing], agent: tuple[str, ...]
) -> Optional[InheritedValue]:
    """Return the first enclosing value that can reach ``node``, innermost layer first."""
    higher: list[tuple[Any, Nullable]] = [
        (node.own_llm_config, node.llm_nullable()),
        (own, contract_nullable),
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
        higher.append((outer.layer, contract_nullable))
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
) -> None:
    """Apply one recorded complete effective map to a static flow tree.

    The map must name exactly the flow's static agent paths.  Validation finishes
    before mutation.  Applying the map deep-copies each config and de-aliases every
    static child callsite, without importing settings or reading the environment.

    Args:
        flow (`CompiledFlow`):
            A clean compiled flow to mutate in place.
        effective (`dict[str, dict[str, Any]]`):
            Recorded complete configurations keyed by static agent path.

    Raises:
        `ValueError`:
            If paths are missing/extra or a config lacks provider/model.
        `TypeError`:
            If the map or one of its configurations is not a dictionary.
    """
    if not isinstance(effective, dict):
        raise TypeError("effective LLM map must be a dict keyed by static agent path")

    expected = set(_static_agent_paths(flow))
    supplied = set(effective)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing paths: {missing}")
        if extra:
            details.append(f"extra paths: {extra}")
        raise ValueError("effective LLM map path mismatch; " + "; ".join(details))

    labels = static_agent_labels(flow)
    for path in sorted(expected):
        config = effective[path]
        if not isinstance(config, dict):
            subject = labels.get(path) or repr(path)
            raise TypeError(f"effective LLM config for {subject} must be a dict")
        require_concrete_selection(config, path=path, label=labels.get(path))

    _apply_flow(flow, effective, path=())


def _resolve_flow(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
    provider_models: dict[str, str],
    stray_model: bool,
    path: tuple[str, ...],
    effective: dict[str, dict[str, Any]],
) -> None:
    """Resolve one flow scope recursively into ``effective``."""
    authored_flow = copy.deepcopy(getattr(flow, "flow_llm_config", {}))
    validate_layer(authored_flow)
    flow_layer = merge_llm_config(
        authored_flow,
        parent_config,
    )

    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            validate_layer(node.own_llm_config)
            base = {} if not node.llm_inherit else flow_layer
            authored = merge_llm_config(copy.deepcopy(node.own_llm_config), base)
            path_key = _path_key(node_path)
            config = resolve_agent_config(
                authored,
                defaults=defaults,
                provider_models=provider_models,
                stray_model=stray_model,
                path=path_key,
                label=_agent_label(node, path),
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
            )


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
            label = _agent_label(node, path)
            if label is not None:
                labels[_path_key(node_path)] = label
        child = getattr(node, "child", None)
        if child is not None:
            _collect_agent_labels(child, node_path, labels)


def _agent_label(node: AgentNode, parent: tuple[str, ...]) -> str | None:
    """Return a synthesized agent's ``llm_config`` label, with its callsite when nested."""
    if node.llm_config_label is None:
        return None
    if not parent:
        return node.llm_config_label
    return f"{node.llm_config_label} in {_path_key(parent)!r}"


def _apply_flow(
    flow: Any,
    effective: dict[str, dict[str, Any]],
    *,
    path: tuple[str, ...],
) -> None:
    """Apply already-validated configs and de-alias children recursively."""
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            node.llm_config = copy.deepcopy(effective[_path_key(node_path)])

        child = getattr(node, "child", None)
        if child is not None:
            node.child = copy.deepcopy(child)
            _apply_flow(node.child, effective, path=node_path)


def bind_host_credentials(
    flow: Any, api_key: SecretStr, headers: dict[str, SecretStr] | None = None,
) -> None:
    """Give every static AGENT marked for a host key that key and its header values.

    `ConnectionSnapshot.bind` marks the agents of the key's provider in the
    effective map. The key and header values go on each marked node's run-local
    ``llm_config`` only, never into the map that ``RunStarted`` records. Call it
    after `apply_effective_llm_map`, which has already de-aliased every child
    callsite.

    Args:
        flow (`CompiledFlow`):
            A flow whose static agents already carry their effective configs.
        api_key (`SecretStr`):
            The host-supplied key.
        headers (`dict[str, SecretStr] | None`, *optional*, defaults to `None`):
            The header values sent with that key; `None` or empty binds none.
    """
    for node in flow.nodes.values():
        if isinstance(node, AgentNode) and node.llm_config.get(HOST_KEY_MARKER) is True:
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
