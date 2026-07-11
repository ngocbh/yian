"""kind=tool — the deterministic-tool node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `tool` node's frozen
`ToolDescriptor`, its located `parse(node_id, reader)` (which records each field's
`SourceRange`), and its `build(desc, *, registry, resolver)` (which constructs the runtime
`ToolNode` + wiring and stamps span metadata onto it). Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.events import SourceRange

# Tool's own field surface (was the `tool` row of parser._KIND_SPECS): `tool_id` is the one
# required flat field; these flat keys are allowed besides the shared `_COMMON_FIELDS` + `kind`.
TOOL_REQUIRED: frozenset[str] = frozenset({"tool_id"})
TOOL_ALLOWED: frozenset[str] = frozenset({"tool_id", "args", "asserts"})
# Descriptor fields carrying inline-call binding strings (the kind-blind walk in bindings.py):
# `args` — a tool's arg map is its sink-binding surface (the tool's `inputs` analogue).
TOOL_BINDING_FIELDS: tuple[str, ...] = ("args",)


@dataclass(frozen=True)
class ToolDescriptor:
    """kind=tool — a direct, deterministic call into TOOL_REGISTRY (untyped args).

    `range`/`field_ranges` are located-parse metadata (the node body's `SourceRange` and
    per-field `SourceRange`s); `compare=False` keeps them out of `==` so a descriptor still
    equals an identically-configured one built without ranges. Populated by `parse`.
    """

    id: str
    tool_id: str
    args: dict[str, Any] = field(default_factory=dict)
    asserts: list[str] = field(default_factory=list)
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)  # static per-node config; merged under flow env: at build
    # Located-parse metadata (see class docstring); excluded from equality.
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)


def parse(node_id: str, reader: NodeReader) -> ToolDescriptor:
    """Read one `kind: tool` node body (via a located `NodeReader`) into a `ToolDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` tool path (allowed/required flat keys, `asserts:`/`env:` shape).
    """
    from agent_composer.compose.parser import _COMMON_FIELDS

    body = reader.value
    allowed_keys = TOOL_ALLOWED | _COMMON_FIELDS | {"kind"}
    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind=tool): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                range=reader.range_of(key),
            )
    for key in TOOL_REQUIRED:
        if key not in body:
            raise LoadError(
                f"node {node_id!r} (kind=tool): missing required field {key!r}",
                range=reader._own_range(),
            )

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in TOOL_ALLOWED:
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
            f"node {node_id!r} (kind=tool): `asserts:` must be a list of strings",
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
                f"node {node_id!r} (kind=tool): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng

    return ToolDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(desc: ToolDescriptor, *, registry, resolver=None, **_loader_hints):
    """Construct the runtime `ToolNode` (+ wiring) for one `ToolDescriptor`.

    Mirrors the legacy `build.build_leaf_node` tool path (node-side `params` + flow-owned
    `args` wiring; a tool declares no `outputs:`, so `output_type` stays unset) and
    additionally stamps `.range`/`.field_ranges` from the descriptor. `resolver` is unused
    (leaf kind) and `_loader_hints` absorbs the inline-CODE traceback anchors that only the
    CODE build reads; both ride the shared `KindSpec.build` signature.
    """
    from agent_composer.compose.build import _sink_params, _sink_wiring
    from agent_composer.nodes.tool import ToolNode

    node = ToolNode(desc.id, tool_id=desc.tool_id, title=desc.node_name)
    node.params = _sink_params(desc.args)
    wiring = _sink_wiring(desc.args)
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring


def register() -> None:
    """Register the `tool` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["tool"] = KindSpec(descriptor_cls=ToolDescriptor, parse=parse, build=build)
    BINDING_FIELDS["tool"] = TOOL_BINDING_FIELDS


register()
