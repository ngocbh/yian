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
from agent_composer.events import SourceRange

# Case's own field surface (was the `case` row of parser._KIND_SPECS): `cases` is the one
# required flat field; `on`/`else` are the optional discriminator/fallback. These flat keys
# are allowed besides the shared `_COMMON_FIELDS` + `kind`. NOTE: `else` is a Python keyword,
# so it maps onto the descriptor's `else_` field (via parser._KEYWORD_FIELD_REMAP).
CASE_REQUIRED: frozenset[str] = frozenset({"cases"})
CASE_ALLOWED: frozenset[str] = frozenset({"cases", "else", "on"})
# A `case` carries CONDITIONS (`on:` + each `when:`), not `${...}` sink bindings, so the
# kind-blind binding walk in `bindings.py` touches no field (the legacy ladder in
# `calls.py:map_binding_strings_in_descriptor` passes a case through unchanged).
CASE_BINDING_FIELDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseDescriptor:
    """kind=case — branch routing (searched `when:` or `on:`-value form).

    A `case` routes only — it carries NO `inputs:` (the desugar maps it onto the
    inputs-bearing `CaseNode`). `else_` is the `else:` fallback.

    `range`/`field_ranges` are located-parse metadata (the node body's `SourceRange` and
    per-field `SourceRange`s); `compare=False` keeps them out of `==` so a descriptor still
    equals an identically-configured one built without ranges. Populated by `parse`.
    """

    id: str
    cases: list[dict[str, Any]] = field(default_factory=list)
    else_: Optional[str] = None
    on: Optional[str] = None
    asserts: list[str] = field(default_factory=list)
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)  # static per-node config; merged under flow env: at build
    # Located-parse metadata (see class docstring); excluded from equality.
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)


def parse(node_id: str, reader: NodeReader) -> CaseDescriptor:
    """Read one `kind: case` node body (via a located `NodeReader`) into a `CaseDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` case path. The bare surface key `on:` is coerced to the boolean
    `True` by YAML 1.1 (SafeLoader) — undone via `parser._normalize_keys` for the body
    iteration; ranges still resolve because `NodeReader` matches the RAW key text (`"on"`).
    The `else:` surface key (a Python keyword) maps onto the `else_` field.
    """
    from agent_composer.compose.parser import (
        _COMMON_FIELDS,
        _KEYWORD_FIELD_REMAP,
        _normalize_keys,
    )

    # Undo the `on:`->True coercion on the reader in place, so spanless presence checks
    # (`_pair` reads `reader.value`) find `on`; span-aware range lookups match raw key text.
    reader.value = _normalize_keys(reader.value)
    body = reader.value
    allowed_keys = CASE_ALLOWED | _COMMON_FIELDS | {"kind"}
    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind=case): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                range=reader.range_of(key),
            )
    for key in CASE_REQUIRED:
        if key not in body:
            raise LoadError(
                f"node {node_id!r} (kind=case): missing required field {key!r}",
                range=reader._own_range(),
            )

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
                f"node {node_id!r} (kind=case): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng

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


def register() -> None:
    """Register the `case` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["case"] = KindSpec(descriptor_cls=CaseDescriptor, parse=parse, build=build)
    BINDING_FIELDS["case"] = CASE_BINDING_FIELDS


register()
