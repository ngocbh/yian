"""kind=loop — the `('a -> 'a) -> 'a -> 'a` driver's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `loop` node's frozen
`LoopDescriptor` and its located `parse(node_id, reader)` (recording each field's `SourceRange`).
`build` drives `build.build_loop_node` (predicate-slice legality + resolve-and-bake). Registered
into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.events import SourceRange

# Loop's own field surface (was the `loop` row of parser._KIND_SPECS): `call` (the body subflow)
# is the one required flat field; `inputs` seeds the carried record; exactly one of
# `while`/`until`/`times` selects the driver (checked at build), and `max` is the runaway guard.
LOOP_REQUIRED: frozenset[str] = frozenset({"call"})
LOOP_ALLOWED: frozenset[str] = frozenset(
    {"call", "inputs", "while", "until", "times", "max", "asserts"}
)
# A `loop` carries no `${...}` sink bindings the kind-blind walk in `bindings.py` should touch:
# `LoopDescriptor` is absent from the legacy `calls.py:map_binding_strings_in_descriptor` ladder,
# so its `inputs:` are NOT inline-call-desugared today. Empty preserves that behavior.
LOOP_BINDING_FIELDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopDescriptor:
    """kind=loop — a `('a -> 'a) -> 'a -> 'a` driver over a carried record.

    `call` names the body subflow (defs-first, else external — same resolution as
    `call`/`map`). `inputs` seeds the carried record at turn 0. Exactly one of
    `while_`/`until_`/`times` carries the predicate; `max` is the runaway guard (required for
    while/until). `while`/`until` are Python keywords, so they land here as `while_`/`until_`
    (the parse remap, like `else`->`else_`).

    `range`/`field_ranges` are located-parse metadata (the node body's `SourceRange` and
    per-field `SourceRange`s); `compare=False` keeps them out of `==` so a descriptor still
    equals an identically-configured one built without ranges. Populated by `parse`.
    """

    id: str
    call: str
    inputs: dict[str, Any] = field(default_factory=dict)
    while_: Optional[str] = None
    until_: Optional[str] = None
    times: Optional[int] = None
    max: Optional[int] = None
    asserts: list[str] = field(default_factory=list)
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)  # static per-node config; merged under flow env: at build
    # Located-parse metadata (see class docstring); excluded from equality.
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)


def parse(node_id: str, reader: NodeReader) -> LoopDescriptor:
    """Read one `kind: loop` node body (via a located `NodeReader`) into a `LoopDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's `SourceRange`
    (`desc.field_ranges[field]`). Field legality mirrors the legacy `parser._parse_node` loop
    path. The surface keys `while:`/`until:` (Python keywords) map onto the `while_`/`until_`
    fields (via `parser._KEYWORD_FIELD_REMAP`).
    """
    from agent_composer.compose.parser import _COMMON_FIELDS, _KEYWORD_FIELD_REMAP

    body = reader.value
    allowed_keys = LOOP_ALLOWED | _COMMON_FIELDS | {"kind"}
    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind=loop): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                range=reader.range_of(key),
            )
    for key in LOOP_REQUIRED:
        if key not in body:
            raise LoadError(
                f"node {node_id!r} (kind=loop): missing required field {key!r}",
                range=reader._own_range(),
            )

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in LOOP_ALLOWED:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        # `while`/`until` -> `while_`/`until_` (the descriptor field names); the rest unchanged.
        kwargs[_KEYWORD_FIELD_REMAP.get(key, key)] = body[key]

    asserts = kwargs.get("asserts")
    if asserts is not None and not (
        isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
    ):
        raise LoadError(
            f"node {node_id!r} (kind=loop): `asserts:` must be a list of strings",
            range=ranges.get("asserts"),
        )

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
                f"node {node_id!r} (kind=loop): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng

    return LoopDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(desc: LoopDescriptor, *, registry=None, resolver, **_loader_hints):
    """Construct the runtime `LoopNode` (+ wiring) for one `LoopDescriptor`.

    Drives `build.build_loop_node` (predicate-slice legality + resolve-and-bake), then stamps
    `.range`/`.field_ranges` from the descriptor. `registry` is unused (loop resolves its output
    type from the body signature); `_loader_hints` absorbs the inline-CODE anchors only CODE reads.
    """
    from agent_composer.compose.build import build_loop_node

    node, wiring = build_loop_node(desc, resolver)
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring


def register() -> None:
    """Register the `loop` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["loop"] = KindSpec(descriptor_cls=LoopDescriptor, parse=parse, build=build)
    BINDING_FIELDS["loop"] = LOOP_BINDING_FIELDS


register()
