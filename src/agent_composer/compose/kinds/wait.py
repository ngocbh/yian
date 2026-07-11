"""kind=wait — the suspend-until-timestamp node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `wait` node's frozen
`WaitDescriptor`, its located `parse(node_id, reader)` (which records each field's
`SourceRange`), and its `build(desc, *, registry, resolver)` (which constructs the runtime
`WaitNode` + the reserved `until` wiring and stamps span metadata onto it). Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.events import SourceRange

# Wait's own field surface (was the `wait` row of parser._KIND_SPECS): `until` is the one
# required flat field and the only allowed one besides the shared `_COMMON_FIELDS` + `kind`.
WAIT_REQUIRED: frozenset[str] = frozenset({"until"})
WAIT_ALLOWED: frozenset[str] = frozenset({"until"})
# A WAIT carries no author sink-binding surface: its `until` source rides the reserved wiring
# key, not the kind-blind inline-call walk. So no binding fields.
WAIT_BINDING_FIELDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class WaitDescriptor:
    """kind=wait — suspend until a timestamp (until:); produces no value.

    `range`/`field_ranges` are located-parse metadata (the node body's `SourceRange` and
    per-field `SourceRange`s); `compare=False` keeps them out of `==` so a descriptor still
    equals an identically-configured one built without ranges. Populated by `parse`.
    """

    id: str
    until: str
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)  # static per-node config; merged under flow env: at build
    # Located-parse metadata (see class docstring); excluded from equality.
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)


def parse(node_id: str, reader: NodeReader) -> WaitDescriptor:
    """Read one `kind: wait` node body (via a located `NodeReader`) into a `WaitDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` wait path (allowed/required flat keys, `env:` shape).
    """
    from agent_composer.compose.parser import _COMMON_FIELDS

    body = reader.value
    allowed_keys = WAIT_ALLOWED | _COMMON_FIELDS | {"kind"}
    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind=wait): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                range=reader.range_of(key),
            )
    for key in WAIT_REQUIRED:
        if key not in body:
            raise LoadError(
                f"node {node_id!r} (kind=wait): missing required field {key!r}",
                range=reader._own_range(),
            )

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in WAIT_ALLOWED:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        kwargs[key] = body[key]

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
                f"node {node_id!r} (kind=wait): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng

    return WaitDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(desc: WaitDescriptor, *, registry, resolver=None, **_loader_hints):
    """Construct the runtime `WaitNode` (+ wiring) for one `WaitDescriptor`.

    Mirrors the legacy `build.build_leaf_node` wait path: a timed WAIT's `until` SOURCE rides
    the reserved wiring key (`{"until": desc.until}`), never a node-side param, and the node
    carries only the `is_timed` residual. Additionally stamps `.range`/`.field_ranges` from
    the descriptor. `resolver` is unused (leaf kind) and `_loader_hints` absorbs the
    inline-CODE traceback anchors that only the CODE build reads; both ride the shared
    `KindSpec.build` signature.
    """
    from agent_composer.nodes.wait import WaitNode

    node = WaitNode(desc.id, is_timed=desc.until is not None, title=desc.node_name)
    node.params = []
    wiring = {"until": desc.until} if desc.until is not None else {}
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring


def register() -> None:
    """Register the `wait` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["wait"] = KindSpec(descriptor_cls=WaitDescriptor, parse=parse, build=build)
    BINDING_FIELDS["wait"] = WAIT_BINDING_FIELDS


register()
