"""kind=map — mapped function application (`List.map`); reuses `call`'s descriptor + build (#70).

The second of the two REF/MAP kinds that SHARE one frozen `CallDescriptor` (owned by
`kinds/call.py`). A `map` is `kind: map` + `over: ${list}` — iteration with `${item}` in
element scope, built as a `MapNode` (`parallel:` overlaps the element runs). This module owns
only the map FIELD SURFACE (`over`/`parallel` join call's `call`/`inputs`) and reuses call's
shared located `parse_call` and its `build` (which drives `build.build_call_node`, branching on
`desc.kind == "map"`). Registered into `KINDS`/`BINDING_FIELDS` at import.
"""

from __future__ import annotations

from agent_composer.compose.kinds.call import CallDescriptor, build, parse_call
from agent_composer.compose.reader import NodeReader

# Map's field surface (was the `map` row of parser._KIND_SPECS): `call` (the callable) and
# `over` (the iteration source) are both required; `inputs`/`parallel`/`asserts` are optional.
MAP_REQUIRED: frozenset[str] = frozenset({"call", "over"})
MAP_ALLOWED: frozenset[str] = frozenset({"call", "inputs", "over", "parallel", "asserts"})
# Descriptor fields carrying `${...}` binding strings: `inputs` (per-element call-arg bindings)
# and `over` (the map iteration source — a `${list}` ref). Same walk as call (see kinds/call.py).
MAP_BINDING_FIELDS: tuple[str, ...] = ("inputs", "over")


def parse(node_id: str, reader: NodeReader) -> CallDescriptor:
    """Read one `kind: map` node body into a `CallDescriptor` (mapped application).

    Reuses call's shared `parse_call`, stamping `kind="map"` (the build discriminator) and the
    map field surface (`over`/`parallel` in addition to `call`/`inputs`).
    """
    return parse_call(node_id, reader, kind="map", required=MAP_REQUIRED, allowed=MAP_ALLOWED)


def register() -> None:
    """Register the `map` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["map"] = KindSpec(descriptor_cls=CallDescriptor, parse=parse, build=build)
    BINDING_FIELDS["map"] = MAP_BINDING_FIELDS


register()
