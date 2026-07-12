"""kind=case — the branch-routing node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `case` node's frozen
`CaseDescriptor` and its located `parse(node_id, reader)` (which records each field's
`SourceRange`). Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch
routes here.

A `case` is DESUGARED (into a `CaseNode` + call targets) by the graph-stage pass in
`compose/cases.py`; it never becomes a leaf runtime node. So `build` here is unreachable
in the normal load path (the loader desugars case before `build_leaf_node`) and exists only
to satisfy the `KindSpec` contract — it raises the same "case is desugared" guard the old
`build_leaf_node` fall-through raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.compose.kinds.helpers import (
    _KEYWORD_FIELD_REMAP,
    check_node_keys,
    read_common_fields,
)
from agent_composer.events import SourceRange

# YAML 1.1 (PyYAML `safe_load`) coerces the bare key `on:` to the boolean `True` (likewise
# `off`/`yes`/`no`). The case-node surface key `on:` is the only such key in any node body, so
# this back-map + `_normalize_keys` (below) undo the coercion. Case-local: no other kind body
# carries a YAML-bool-coerced key.
_YAML_BOOL_KEYS = {True: "on", False: "off"}


def _normalize_keys(body: dict) -> dict:
    """Undo YAML 1.1 boolean-key coercion for the surface key `on:`."""
    if not any(k in _YAML_BOOL_KEYS for k in body):
        return body
    return {_YAML_BOOL_KEYS.get(k, k): v for k, v in body.items()}

# Case's own field surface (was the `case` row of parser._KIND_SPECS): `cases` is the one
# required flat field; `on`/`else` are the optional discriminator/fallback. These flat keys
# are allowed besides the shared `_COMMON_FIELDS` + `kind`. NOTE: `else` is a Python keyword,
# so it maps onto the descriptor's `else_` field (via helpers._KEYWORD_FIELD_REMAP).
CASE_REQUIRED: frozenset[str] = frozenset({"cases"})
CASE_ALLOWED: frozenset[str] = frozenset({"cases", "else", "on"})
# A `case` carries CONDITIONS (`on:` + each `when:`), not `${...}` sink bindings, so the
# kind-blind binding walk in `bindings.py` touches no field (the legacy ladder in
# `calls.py:map_binding_strings_in_descriptor` passes a case through unchanged).
CASE_BINDING_FIELDS: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class CaseDescriptor(NodeDescriptor):
    """kind=case — branch routing (searched `when:` or `on:`-value form).

    A `case` routes only — it carries NO `inputs:` (the desugar maps it onto the
    inputs-bearing `CaseNode`). `else_` is the `else:` fallback.

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`; only the case-specific
    fields are declared here.
    """

    cases: list[dict[str, Any]] = field(default_factory=list)
    else_: Optional[str] = None
    on: Optional[str] = None
    asserts: list[str] = field(default_factory=list)


def parse(node_id: str, reader: NodeReader) -> CaseDescriptor:
    """Read one `kind: case` node body (via a located `NodeReader`) into a `CaseDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` case path. The bare surface key `on:` is coerced to the boolean
    `True` by YAML 1.1 (SafeLoader) — undone via the module-local `_normalize_keys` for the
    body iteration; ranges still resolve because `NodeReader` matches the RAW key text (`"on"`).
    The `else:` surface key (a Python keyword) maps onto the `else_` field.
    """
    # Undo the `on:`->True coercion on the reader in place, so spanless presence checks
    # (`_pair` reads `reader.value`) find `on`; span-aware range lookups match raw key text.
    reader.value = _normalize_keys(reader.value)
    check_node_keys(node_id, "case", reader, CASE_ALLOWED, CASE_REQUIRED)
    body = reader.value

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in CASE_ALLOWED:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        # `else` -> `else_` (the descriptor field name); `cases`/`on` are unchanged.
        kwargs[_KEYWORD_FIELD_REMAP.get(key, key)] = body[key]

    read_common_fields(node_id, "case", reader, kwargs, ranges)

    return CaseDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(desc: CaseDescriptor, *, registry, resolver=None, **_loader_hints):
    """A `case` is not a leaf node — it is desugared (see the module docstring).

    Unreachable in the normal load path (the loader desugars case before dispatching to a
    kind's `build`). Present only to satisfy the `KindSpec` contract; raises the same guard
    the old `build_leaf_node` fall-through raised, so a stray call fails loud.
    """
    raise LoadError(
        f"node {desc.id!r}: kind case is not a leaf node "
        f"(case nodes are desugared; ref/map nodes expand at runtime)"
    )
