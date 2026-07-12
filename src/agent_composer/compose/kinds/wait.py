"""kind=wait — the suspend-until-timestamp node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `wait` node's frozen
`WaitDescriptor`, its located `parse(node_id, reader)` (which records each field's
`SourceRange`), and its `build(desc, *, registry, resolver)` (which constructs the runtime
`WaitNode` + the reserved `until` wiring and stamps span metadata onto it). Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_composer.compose.reader import NodeReader
from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.compose.kinds.helpers import (
    check_node_keys,
    read_common_fields,
)
from agent_composer.events import SourceRange
from agent_composer.nodes.wait import WaitNode

# Wait's own field surface (was the `wait` row of parser._KIND_SPECS): `until` is the one
# required flat field and the only allowed one besides the shared `_COMMON_FIELDS` + `kind`.
WAIT_REQUIRED: frozenset[str] = frozenset({"until"})
WAIT_ALLOWED: frozenset[str] = frozenset({"until"})
# A WAIT carries no author sink-binding surface: its `until` source rides the reserved wiring
# key, not the kind-blind inline-call walk. So no binding fields.
WAIT_BINDING_FIELDS: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class WaitDescriptor(NodeDescriptor):
    """kind=wait — suspend until a timestamp (until:); produces no value.

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`; only `until` is declared here.
    """

    until: str


def parse(node_id: str, reader: NodeReader) -> WaitDescriptor:
    """Read one `kind: wait` node body (via a located `NodeReader`) into a `WaitDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` wait path (allowed/required flat keys, `env:` shape).
    """
    check_node_keys(node_id, "wait", reader, WAIT_ALLOWED, WAIT_REQUIRED)
    body = reader.value

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in WAIT_ALLOWED:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        kwargs[key] = body[key]

    read_common_fields(node_id, "wait", reader, kwargs, ranges)

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
    node = WaitNode(desc.id, is_timed=desc.until is not None, title=desc.node_name)
    node.params = []
    wiring = {"until": desc.until} if desc.until is not None else {}
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring
