"""Resolve and replay effective LLM configuration over the static call tree.

Resolution is a compile-layer operation: callers provide both the outermost authored
layer and explicit process defaults.  The resulting map is therefore complete and can
be recorded and replayed without consulting process settings again.

Static paths join local author node ids with ``/``.  The Compose parser reserves that
separator in node ids, so paths such as ``review/summarize/model`` are unambiguous.  A
child flow is deep-copied at every CALL/MAP/LOOP callsite before either resolution or
replay.  A root-level copy alone preserves aliases between shared descendant defs and
would let one parent's cascade overwrite another's.

Layer: compile — imports nodes and LLM configuration primitives; never runtime or
``_settings``. Defaults are injected by the host.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import SecretStr

from yian.llm_clients.config import (
    LLM_CONFIG_KEYS,
    RETIRED_LLM_CONFIG_KEYS,
    LLMConfig,
    host_only_field_error,
    merge_llm_config,
    retired_key_error,
    validate_base_url,
)
from yian.llm_clients.connection import HOST_KEY_MARKER, validate_connection
from yian.nodes.agent import AgentNode

_REQUIRED_SELECTION_FIELDS = ("provider", "model")


def resolve_llm_cascade(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
    provider_models: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Bake and return every AGENT node's complete effective configuration.

    The normal precedence is node, current flow, enclosing flows, caller layer, then
    ``defaults``. An AGENT node with ``llm_inherit=False`` skips the flow/enclosing/caller
    layers but still receives explicit defaults so its provider and model are concrete.
    Every returned value and every node-owned value is an independent deep copy.

    Args:
        flow (`CompiledFlow`):
            The flow whose complete static call tree is resolved in place.
        parent_config (`dict[str, Any]`):
            The outermost gap-fill layer.  At the root this is the CLI/host layer; in a
            child it is the already-accumulated enclosing flow layer.
        defaults (`dict[str, Any]`):
            Explicit lowest-precedence process defaults. Must supply ``provider``;
            ``model`` is a fallback only for that provider. No environment is read.
        provider_models (`dict[str, str] | None`):
            Matching fallback models for other providers. An authored model still
            wins per field; a provider with no matching fallback needs an explicit model.

    Returns:
        `dict[str, dict[str, Any]]`:
            Complete effective configurations keyed by unambiguous static agent path.

    Raises:
        `ValueError`:
            If a provider or model selection is invalid.
    """
    _validate_layer(parent_config)
    _validate_layer(defaults)
    if not isinstance(defaults.get("provider"), str) or not defaults["provider"]:
        raise ValueError("defaults lacks concrete provider")
    models = dict(provider_models or {})
    if defaults.get("model"):
        models.setdefault(defaults["provider"], defaults["model"])
    effective: dict[str, dict[str, Any]] = {}
    _resolve_flow(
        flow,
        copy.deepcopy(parent_config),
        defaults=copy.deepcopy(defaults),
        provider_models=models,
        path=(),
        effective=effective,
    )
    return effective


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
        _require_concrete_selection(config, path=path, label=labels.get(path))

    _apply_flow(flow, effective, path=())


def _resolve_flow(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
    provider_models: dict[str, str],
    path: tuple[str, ...],
    effective: dict[str, dict[str, Any]],
) -> None:
    """Resolve one flow scope recursively into ``effective``."""
    authored_flow = copy.deepcopy(getattr(flow, "flow_llm_config", {}))
    _validate_layer(authored_flow)
    flow_layer = merge_llm_config(
        authored_flow,
        parent_config,
    )

    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            _validate_layer(node.own_llm_config)
            base = {} if not node.llm_inherit else flow_layer
            authored = merge_llm_config(copy.deepcopy(node.own_llm_config), base)
            config = merge_llm_config(authored, {key: value for key, value in defaults.items() if key != "model"})
            if not authored.get("model"):
                config["model"] = provider_models.get(config["provider"])
                if not config["model"]:
                    raise ValueError(f"Provider {config['provider']!r} needs an explicit model; set llm_config.model or --model")
            path_key = _path_key(node_path)
            _require_concrete_selection(config, path=path_key, label=_agent_label(node, path))
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


def bind_api_key(flow: Any, api_key: SecretStr) -> None:
    """Give every static AGENT marked for a host key that key.

    `ConnectionSnapshot.bind` marks the agents of the key's provider in the
    effective map. The key goes on each marked node's run-local ``llm_config``
    only, never into the map that ``RunStarted`` records. Call it after
    `apply_effective_llm_map`, which has already de-aliased every child callsite.

    Args:
        flow (`CompiledFlow`):
            A flow whose static agents already carry their effective configs.
        api_key (`SecretStr`):
            The host-supplied key.
    """
    for node in flow.nodes.values():
        if isinstance(node, AgentNode) and node.llm_config.get(HOST_KEY_MARKER) is True:
            node.llm_config["api_key"] = api_key
        child = getattr(node, "child", None)
        if child is not None:
            bind_api_key(child, api_key)


def host_key_providers(effective: dict[str, dict[str, Any]]) -> set[str]:
    """Return the providers whose agents a recorded effective map marks for a host key."""
    return {
        config["provider"] for config in effective.values() if config.get(HOST_KEY_MARKER) is True
    }


def _path_key(path: tuple[str, ...]) -> str:
    """Encode one non-empty static author path."""
    return "/".join(path)


def _require_concrete_selection(
    config: dict[str, Any], *, path: str, label: str | None = None,
) -> None:
    """Reject a config that could fall through to ambient provider/model settings.

    Args:
        config (`dict[str, Any]`):
            One agent's effective configuration.
        path (`str`):
            The agent's static path, named by every message unless ``label`` is given.
        label (`str`, *optional*, defaults to `None`):
            The authored site of a synthesized agent's ``llm_config``, named instead of
            ``path``.

    Raises:
        `ValueError`:
            If the config has an invalid marker or base_url, a bad field, or no
            concrete provider/model.
    """
    subject = label or repr(path)
    if config.get(HOST_KEY_MARKER, True) is not True:
        raise ValueError(f"effective LLM config for {subject} has an invalid host key marker")
    if "base_url" in config:
        # An Azure base_url is its resource endpoint; any other serves only a host key.
        if HOST_KEY_MARKER not in config and config.get("provider") != "azure":
            raise ValueError(
                f"effective LLM config for {subject} has a base_url without a host key; "
                "only provider azure takes a base_url from a flow file"
            )
        # A recorded map is replayed on resume; check its URL before any node runs.
        try:
            validate_base_url(config["base_url"])
        except ValueError as exc:
            raise ValueError(f"effective LLM config for {subject}: {exc}") from None
    # Only a resolved map carries the marker and the host key's base_url;
    # authored layers may not (except an Azure endpoint, checked above). A retired
    # key can only come from a map recorded before it retired; such a finished run
    # still replays, and resuming an unfinished one refuses it per agent.
    _validate_layer({
        key: value
        for key, value in config.items()
        if key not in (HOST_KEY_MARKER, "base_url") and key not in RETIRED_LLM_CONFIG_KEYS
    })
    missing = [
        field
        for field in _REQUIRED_SELECTION_FIELDS
        if not isinstance(config.get(field), str) or not config[field]
    ]
    if missing:
        raise ValueError(
            f"effective LLM config for {subject} lacks concrete fields: {missing}"
        )


def _validate_layer(config: dict[str, Any]) -> None:
    """Reject unknown fields, retired options, and malformed host metadata before composition."""
    host_only = host_only_field_error(config)
    if host_only is not None:
        # Every layer here ends up in the recorded effective map, and a lower
        # layer's base_url would redirect the host key.
        raise ValueError(host_only)
    if HOST_KEY_MARKER in config:
        raise ValueError(f"llm_config.{HOST_KEY_MARKER} is internal to recorded runs")
    if str(config.get("provider", "")).lower() == "codex" or any(key in config for key in ("sandbox", "cwd")):
        LLMConfig(**config)
    retired = retired_key_error(config)
    if retired is not None:
        raise ValueError(retired)
    # The loader checks authored values; a host layer is checked here only for
    # unknown keys, which would otherwise be recorded and silently ignored. Its
    # values, including block contents, are checked against the effective config
    # before the run starts (`check_model_config`) and when the model client is built.
    unknown = sorted(set(config) - LLM_CONFIG_KEYS - {"_connection"})
    for key in unknown:
        if isinstance(key, str) and key.replace("_", "-") in LLM_CONFIG_KEYS:
            raise ValueError(f"`{key}` is not an llm_config key; write `{key.replace('_', '-')}`")
    if unknown:
        raise ValueError(
            f"llm_config has unknown field(s) {unknown} "
            f"(allowed: {', '.join(sorted(LLM_CONFIG_KEYS))})"
        )
    if "_connection" in config:
        if config.get("provider") != "openai":
            raise ValueError("Internal connection metadata requires provider openai")
        validate_connection(config["_connection"])
