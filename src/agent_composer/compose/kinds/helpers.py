"""Shared descriptor <-> runtime `Node` bits for the kind parse/build handles.

The kinds-package-internal shared layer: the machinery each `compose/kinds/<kind>.py`
`parse`/`build` needs but that is genuinely common across kinds. Two clusters:

- **Common node fields** (`check_node_keys`, `read_common_fields`, `_COMMON_FIELDS`,
  `_KEYWORD_FIELD_REMAP`) — the parse-side bits every kind's `parse` shares over the
  `NodeDescriptor` base (which lives in `base.py`): `check_node_keys` rejects unknown/missing
  keys (unioning in the common keys + `kind` so no kind repeats them); and `read_common_fields`
  lifts the four author-surface commons (`node_name`/`depends_on`/`runs_after`/`env`) into its
  kwargs. `_COMMON_FIELDS` (derived from the base's fields) and the surface->field keyword remap
  are private details of those helpers. All of this is consumed only by the kind `parse`s.
- **Sink helpers** (`_output_type`, `_sink_params`, `_sink_wiring`, `_validate_llm_config`) —
  the leaf kinds (agent/code/model/tool/human_input) call these to stamp a node's
  `output_type`/`params` and the flow-owned wiring (the split: the node holds only param
  NAMES, the flow owns the sources).

The child resolve-and-signature cluster (`ChildSignature`/`ChildResolver`, `child_signature`,
`_resolve_child`, `_output_value_type`) that `call`/`map`/`loop` and the graph-stages share
lives in the sibling leaf `resolver.py`, not here.

This module imports only the `NodeDescriptor` base (from the leaf `base.py`), no `kinds/__init__`
symbol and no other kind module, so it is itself a leaf of the kinds package: every kind's
`parse`/`build` imports from it at module top level without cycling. It deliberately does NOT
import `parser.py` (which re-exports the descriptors), so the common-field constants live here
rather than in `parser` — keeping the kind `parse` imports top-level.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Optional

from agent_composer.nodes.binding import ParamDecl
from agent_composer.llm_clients import LLMConfig
from agent_composer.typesys.values import Type
from agent_composer.typesys.types import TypeRegistry
from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.types import read_type
from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.events import SourceRange


# --------------------------------------------------------------------------- #
# Common node fields — the parse-side bits every kind's `parse` shares.
# --------------------------------------------------------------------------- #

# Common keys every node body accepts (besides `kind` + its own kind-specific fields), derived
# from the `NodeDescriptor` base's author-surface fields (`id` is the keyed-map key, not a body
# key; `range`/`field_ranges` are parse metadata, not body keys). Private to this module: kinds
# pass only their OWN allowed set to `check_node_keys`, which unions this in.
#
# `inputs`/`outputs` live on the base (so the sink helpers can read them uniformly) but are NOT
# common authoring keys: a signature-less kind (`case`/`wait`) must still reject them. Each
# value-shaped kind re-lists `inputs`/`outputs` in its OWN allowed set, so subtracting them here
# keeps `case`/`wait` rejecting them while every other kind still accepts them.
_COMMON_FIELDS = frozenset(f.name for f in fields(NodeDescriptor)) - {
    "id",
    "range",
    "field_ranges",
    "inputs",
    "outputs",
}

# Surface keys that collide with Python keywords map onto a trailing-underscore field on the
# descriptor: `else:`->`else_` (case), `while:`/`until:`->`while_`/`until_` (loop).
_KEYWORD_FIELD_REMAP = {"else": "else_", "while": "while_", "until": "until_"}


def check_node_keys(
    node_id: str,
    kind: str,
    reader: NodeReader,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> None:
    """Legality check for a node body's surface keys, shared by every kind's `parse`.

    `allowed` is the kind's OWN allowed-key set (surface spelling, e.g. `while`/`else`); the
    common node keys (`_COMMON_FIELDS`) and `kind` are unioned in here so no kind repeats them.
    Raises `LoadError` for any key outside that union, then for any missing `required` key —
    both naming `kind`. Reads nothing into the descriptor (that is `read_common_fields` + the
    kind's own field loop)."""
    allowed_keys = allowed | _COMMON_FIELDS | {"kind"}
    body = reader.value
    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind={kind}): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                range=reader.range_of(key),
            )
    for key in required:
        if key not in body:
            raise LoadError(
                f"node {node_id!r} (kind={kind}): missing required field {key!r}",
                range=reader._own_range(),
            )


def read_common_fields(
    node_id: str,
    kind: str,
    reader: NodeReader,
    kwargs: dict[str, Any],
    ranges: dict[str, SourceRange],
) -> None:
    """Lift the four common node fields into a descriptor's `parse` accumulators.

    Reads `node_name`/`depends_on`/`runs_after`/`env` from the node body (`reader.value`) into
    `kwargs`, recording each present field's `SourceRange` into `ranges`; both dicts are mutated
    IN PLACE (they are the accumulators each `parse` passes to its descriptor constructor).
    `env:` is validated to be a mapping with string keys — a bad shape raises `LoadError`
    naming `kind`. The other three are copied through untouched (their per-field shape is
    checked later by the loader edge pass)."""
    body = reader.value
    for key in ("node_name", "depends_on", "runs_after"):
        if key in body:
            kwargs[key] = body[key]
            rng = reader.range_of(key)
            if rng is not None:
                ranges[key] = rng
    if "env" in body:
        env = body["env"]
        if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
            raise LoadError(
                f"node {node_id!r} (kind={kind}): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng


# --------------------------------------------------------------------------- #
# Sink helpers — the node-side signature + flow-owned wiring split, shared by
# every leaf build and the call/map/loop cluster below.
# --------------------------------------------------------------------------- #


def _output_type(outputs: Any, registry: TypeRegistry) -> Optional[Type]:
    """A descriptor's declared `outputs:` as one `Type`; `None` when absent."""
    if outputs is None:
        return None
    return read_type(outputs, registry)


def _sink_params(mapping: dict[str, Any]) -> list[ParamDecl]:
    """A descriptor's `inputs:`/`args:` map -> untyped `ParamDecl`s (the node-side signature).

    The split: `params` carries only the declared NAMES (no source); the flow owns the
    sources in `CompiledFlow.wiring`. Untyped (`type_str`/`type` None) — the source carries the
    type; derived from the SAME map as `_sink_wiring` so names and keys are in lockstep."""
    return [ParamDecl(name=k) for k in (mapping or {})]


def _sink_wiring(mapping: dict[str, Any]) -> dict[str, Any]:
    """A descriptor's `inputs:`/`args:` map -> the flow-owned wiring (`param -> source`).
    The source half of the split; the same dict `_sink_params` reads the names from."""
    return dict(mapping or {})


def _validate_llm_config(
    cfg: dict[str, Any] | None, node_id: str
) -> tuple[dict[str, Any], bool]:
    """Validate a raw llm_config dict at LOAD time and split off the reserved `inherit` key.

    `inherit` (whole-node cascade opt-out, default True) is NOT an `LLMConfig` field, so it
    must be popped before the `extra="forbid"` round-trip that fires on typo'd keys (e.g.
    `temparature`). Returns `(config_without_inherit, inherit)`. Raises `LoadError` (not
    pydantic ValidationError) so the surface speaks one error type.
    """
    if not cfg:
        return {}, True
    cfg = dict(cfg)
    inherit = bool(cfg.pop("inherit", True))
    try:
        LLMConfig(**cfg)
    except Exception as exc:
        raise LoadError(f"node {node_id!r}: llm_config: {exc}") from exc
    return cfg, inherit

