"""Resolve one agent's `llm_config` layers into its complete effective configuration.

The compile-time cascade (`yian.compile.llm_cascade`) merges each agent's authored
layers over the static call tree and calls `resolve_agent_config` per agent;
`validate_layer` checks every layer before it is merged and `require_concrete_selection`
checks a complete or recorded config before a run uses it. Nothing here reads the
environment or saved settings: the host passes its defaults and saved models in.
"""

from __future__ import annotations

import copy
from typing import Any

from yian.llm_clients.config import (
    LLM_CONFIG_KEYS,
    RETIRED_LLM_CONFIG_KEYS,
    LLMConfig,
    host_only_field_error,
    merge_llm_config,
    retired_key_error,
    validate_base_url,
)
from yian.llm_clients.connection import (
    HEADER_NAMES_KEY,
    HOST_KEY_MARKER,
    missing_model_message,
    validate_connection,
    validate_header_names,
)
from yian.llm_clients.connection_settings import validate_organization

_REQUIRED_SELECTION_FIELDS = ("provider", "model")
# Recorded only on agents marked for a host key: the nonsecret settings sent with it.
_HOST_SETTING_KEYS = ("organization", "project", HEADER_NAMES_KEY)


def resolve_agent_config(
    authored: dict[str, Any],
    *,
    defaults: dict[str, Any],
    provider_models: dict[str, str],
    stray_model: bool,
    path: str,
    label: str | None,
) -> dict[str, Any]:
    """Return one agent's complete effective configuration from its merged authored layers.

    ``defaults`` other than ``model`` fill the gaps under ``authored``. An unset model
    comes from ``provider_models`` for the resolved provider, or from ``defaults`` when
    that provider is the default one. The result is then checked as a concrete selection.

    Args:
        authored (`dict[str, Any]`):
            The agent's node layer already merged over its flow, enclosing and host
            layers (or over nothing under ``inherit: false``). Not mutated.
        defaults (`dict[str, Any]`):
            Lowest-precedence host defaults, possibly empty.
        provider_models (`dict[str, str]`):
            Saved models by provider.
        stray_model (`bool`):
            Whether ``YIAN_DEFAULT_MODEL`` was set without a default provider.
        path (`str`):
            The agent's static path, named by every message unless ``label`` is given.
        label (`str | None`):
            The authored site of a synthesized agent's ``llm_config``, named instead of
            ``path``.

    Returns:
        `dict[str, Any]`: A new configuration that shares no value with the inputs.

    Raises:
        `ValueError`:
            If ``authored`` holds a field internal to recorded runs, the agent has no
            provider or no model, or the result is not a concrete, valid selection.
    """
    # Only resolution adds the marker and header names; an input layer never has them.
    _reject_internal_keys(authored)
    authored = copy.deepcopy(authored)
    config = merge_llm_config(
        authored, {key: value for key, value in copy.deepcopy(defaults).items() if key != "model"}
    )
    provider = config.get("provider")
    if not provider:
        raise ValueError(missing_model_message(path, None, stray_model=stray_model, label=label))
    if not authored.get("model"):
        models = dict(provider_models)
        if defaults.get("provider") and defaults.get("model"):
            models.setdefault(defaults["provider"], defaults["model"])
        config["model"] = models.get(provider)
        if not config["model"]:
            raise ValueError(
                missing_model_message(path, provider, stray_model=stray_model, label=label)
            )
    require_concrete_selection(config, path=path, label=label)
    return config


def require_concrete_selection(
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
            If the config has an invalid marker, base_url or host setting, a bad
            field, or no concrete provider/model.
    """
    subject = label or repr(path)
    if config.get(HOST_KEY_MARKER, True) is not True:
        raise ValueError(f"effective LLM config for {subject} has an invalid host key marker")
    for key in _HOST_SETTING_KEYS:
        if key not in config:
            continue
        # Only the host's settings, sent with its key, are recorded beside the marker.
        if HOST_KEY_MARKER not in config:
            raise ValueError(f"effective LLM config for {subject} has {key} without a host key")
        if key != HEADER_NAMES_KEY and config.get("provider") != "openai":
            raise ValueError(
                f"effective LLM config for {subject}: {key} applies only to provider openai"
            )
        try:
            if key == HEADER_NAMES_KEY:
                validate_header_names(config[key], config.get("provider"))
            else:
                validate_organization(config[key], key)
        except ValueError as exc:
            raise ValueError(f"effective LLM config for {subject}: {exc}") from None
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
    # Only a resolved map carries the marker and the host key's base_url and
    # settings; authored layers may not (except an Azure endpoint, checked above). A
    # retired key can only come from a map recorded before it retired; such a finished
    # run still replays, and resuming an unfinished one refuses it per agent.
    validate_layer({
        key: value
        for key, value in config.items()
        if key not in (HOST_KEY_MARKER, "base_url", *_HOST_SETTING_KEYS)
        and key not in RETIRED_LLM_CONFIG_KEYS
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


def validate_layer(config: dict[str, Any]) -> None:
    """Reject unknown fields, retired options, and malformed host metadata before composition.

    Args:
        config (`dict[str, Any]`):
            One authored, host, or defaults layer.

    Raises:
        `ValueError`:
            If the layer sets a host-only or internal field, a retired or unknown key,
            or a malformed connection.
    """
    host_only = host_only_field_error(config)
    if host_only is not None:
        # Every layer here ends up in the recorded effective map, and a lower
        # layer's base_url would redirect the host key.
        raise ValueError(host_only)
    _reject_internal_keys(config)
    provider = config.get("provider")
    if (isinstance(provider, str) and provider.lower() == "codex") or any(
        key in config for key in ("sandbox", "cwd")
    ):
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
        validate_connection(config["_connection"], config.get("provider"))


def _reject_internal_keys(config: dict[str, Any]) -> None:
    """Reject the fields only a resolved, recorded config may hold."""
    for internal in (HOST_KEY_MARKER, HEADER_NAMES_KEY):
        if internal in config:
            raise ValueError(f"llm_config.{internal} is internal to recorded runs")
