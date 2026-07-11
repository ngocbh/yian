"""kind=call|map — function application of a callable; descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules. `call` and `map` SHARE one frozen
`CallDescriptor` (the REF/MAP discriminator is `desc.kind`), so this module owns the
descriptor and the shared located parse; `kinds/map.py` reuses both with the map field
surface. `build(desc, *, registry, resolver)` drives `build.build_call_node` (which branches
on `desc.kind`: `"map"` -> `MapNode`, else `CallNode`). Registered into `KINDS`/`BINDING_FIELDS`
at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.events import SourceRange

# Call's own field surface (was the `call` row of parser._KIND_SPECS): `call` (the callable) is
# the one required flat field. `map` reuses CallDescriptor with a wider surface (see kinds/map.py).
CALL_REQUIRED: frozenset[str] = frozenset({"call"})
CALL_ALLOWED: frozenset[str] = frozenset({"call", "inputs", "asserts"})
# Descriptor fields carrying `${...}` binding strings (the kind-blind walk in bindings.py):
# `inputs` (the call-arg bindings) and `over` (the map iteration source — always None for a
# plain `call`, so the walk skips its non-string value). Mirrors the CallDescriptor branch of
# the legacy `calls.py:map_binding_strings_in_descriptor` ladder.
CALL_BINDING_FIELDS: tuple[str, ...] = ("inputs", "over")


@dataclass(frozen=True)
class CallDescriptor:
    """kind=call|map — function application of a callable (`call:` + call-arg bindings).

    `call` names the callable, resolved **defs-first, else an external flow id** (the
    loader composes that resolution). `kind: call` is a single application (built as a plain
    `CallNode`); `kind: map` + `over: ${list}` is `List.map` — iteration with `${item}` in
    element scope (built as a `MapNode`), and `parallel:` overlaps the element runs. `inputs:`
    are the (per-element) call-arg bindings.

    `kind` is the REF/MAP discriminator: `"call"` (REF, the default) or `"map"`. The parser sets
    it from the YAML kind; the two SYNTH paths (inline-call + case desugar) keep the REF default
    (`over=None`). `build` branches on it: a `map` builds a `MapNode`, a `call` a `CallNode`.

    `range`/`field_ranges` are located-parse metadata (the node body's `SourceRange` and
    per-field `SourceRange`s); `compare=False` keeps them out of `==` so a descriptor still
    equals an identically-configured one built without ranges. Populated by `parse`.
    """

    id: str
    call: str
    inputs: dict[str, Any] = field(default_factory=dict)
    over: Optional[str] = None
    parallel: bool = False
    kind: str = "call"
    asserts: list[str] = field(default_factory=list)
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)  # static per-node config; merged under flow env: at build
    # Located-parse metadata (see class docstring); excluded from equality.
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)


def parse_call(
    node_id: str,
    reader: NodeReader,
    *,
    kind: str,
    required: frozenset[str],
    allowed: frozenset[str],
) -> CallDescriptor:
    """Shared located parse for `call` and `map` (both build a `CallDescriptor`).

    `kind` (`"call"`/`"map"`) is stamped onto the descriptor (the REF/MAP build discriminator);
    `required`/`allowed` are the kind's flat-field surface. Records the node body's `SourceRange`
    (`desc.range`) and each present field's `SourceRange` (`desc.field_ranges[field]`). Field
    legality mirrors the legacy `parser._parse_node` call/map path.
    """
    from agent_composer.compose.parser import _COMMON_FIELDS

    body = reader.value
    allowed_keys = allowed | _COMMON_FIELDS | {"kind"}
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

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id, "kind": kind}
    for key in allowed:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        kwargs[key] = body[key]

    asserts = kwargs.get("asserts")
    if asserts is not None and not (
        isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
    ):
        raise LoadError(
            f"node {node_id!r} (kind={kind}): `asserts:` must be a list of strings",
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
                f"node {node_id!r} (kind={kind}): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng

    return CallDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def parse(node_id: str, reader: NodeReader) -> CallDescriptor:
    """Read one `kind: call` node body into a `CallDescriptor` (single application)."""
    return parse_call(node_id, reader, kind="call", required=CALL_REQUIRED, allowed=CALL_ALLOWED)


def build(desc: CallDescriptor, *, registry=None, resolver, **_loader_hints):
    """Construct the runtime `CallNode`/`MapNode` (+ wiring) for one `CallDescriptor`.

    Drives `build.build_call_node` (which branches on `desc.kind`), then stamps
    `.range`/`.field_ranges` from the descriptor. `registry` is unused (call/map resolve their
    output type from the child signature, not the flow typedefs); `_loader_hints` absorbs the
    inline-CODE traceback anchors that only the CODE build reads — both ride the shared
    `KindSpec.build` signature.
    """
    from agent_composer.compose.build import build_call_node

    node, wiring = build_call_node(desc, resolver)
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring


def register() -> None:
    """Register the `call` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["call"] = KindSpec(descriptor_cls=CallDescriptor, parse=parse, build=build)
    BINDING_FIELDS["call"] = CALL_BINDING_FIELDS


register()
