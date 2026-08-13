"""Resolve and replay effective LLM configuration over the static call tree.

Resolution is a compile-layer operation: callers provide both the outermost authored
layer and explicit process defaults.  The resulting map is therefore complete and can
be recorded and replayed without consulting process settings again.

Static paths join local author node ids with ``/``.  The Compose parser reserves that
separator in node ids, so paths such as ``review/summarize/model`` are unambiguous.  A
child flow is deep-copied at every CALL/MAP/LOOP callsite before either resolution or
replay.  A root-level copy alone preserves aliases between shared descendant defs and
would let one parent's cascade overwrite another's.

Layer: compile — imports nodes and the LLM merge primitive only; never runtime or
``_settings``.
"""

from __future__ import annotations

import copy
from typing import Any

from agent_composer.llm_clients.config import merge_llm_config
from agent_composer.nodes.agent import AgentNode

_REQUIRED_SELECTION_FIELDS = ("provider", "model")


def resolve_llm_cascade(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bake and return every agent's complete effective configuration.

    The normal precedence is node, current flow, enclosing flows, caller layer, then
    ``defaults``.  An agent with ``llm_inherit=False`` skips the flow/enclosing/caller
    layers but still receives explicit defaults so its provider and model are concrete.
    Every returned value and every node-owned value is an independent deep copy.

    Args:
        flow (`CompiledFlow`):
            The flow whose complete static call tree is resolved in place.
        parent_config (`dict[str, Any]`):
            The outermost gap-fill layer.  At the root this is the CLI/host layer; in a
            child it is the already-accumulated enclosing flow layer.
        defaults (`dict[str, Any]`):
            Explicit lowest-precedence process defaults.  Must supply concrete
            ``provider`` and ``model`` values; this module never reads the environment.

    Returns:
        `dict[str, dict[str, Any]]`:
            Complete effective configurations keyed by unambiguous static agent path.

    Raises:
        `ValueError`:
            If an effective agent selection lacks a concrete provider or model.
    """
    _require_concrete_selection(defaults, path="defaults")
    effective: dict[str, dict[str, Any]] = {}
    _resolve_flow(
        flow,
        copy.deepcopy(parent_config),
        defaults=copy.deepcopy(defaults),
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

    for path in sorted(expected):
        config = effective[path]
        if not isinstance(config, dict):
            raise TypeError(f"effective LLM config for {path!r} must be a dict")
        _require_concrete_selection(config, path=path)

    _apply_flow(flow, effective, path=())


def _resolve_flow(
    flow: Any,
    parent_config: dict[str, Any],
    *,
    defaults: dict[str, Any],
    path: tuple[str, ...],
    effective: dict[str, dict[str, Any]],
) -> None:
    """Resolve one flow scope recursively into ``effective``."""
    flow_layer = merge_llm_config(
        copy.deepcopy(getattr(flow, "flow_llm_config", {})),
        parent_config,
    )
    inherited_layer = merge_llm_config(flow_layer, defaults)

    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        if isinstance(node, AgentNode):
            base = defaults if not node.llm_inherit else inherited_layer
            config = merge_llm_config(copy.deepcopy(node.own_llm_config), base)
            path_key = _path_key(node_path)
            _require_concrete_selection(config, path=path_key)
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


def _path_key(path: tuple[str, ...]) -> str:
    """Encode one non-empty static author path."""
    return "/".join(path)


def _require_concrete_selection(config: dict[str, Any], *, path: str) -> None:
    """Reject a config that could fall through to ambient provider/model settings."""
    missing = [
        field
        for field in _REQUIRED_SELECTION_FIELDS
        if not isinstance(config.get(field), str) or not config[field]
    ]
    if missing:
        raise ValueError(
            f"effective LLM config for {path!r} lacks concrete fields: {missing}"
        )
