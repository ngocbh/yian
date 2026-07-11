"""kind=model — the ML-inference node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `model` node's frozen
`ModelDescriptor`, its located `parse(node_id, reader)` (which records each field's
`SourceRange`), and its `build(desc, *, registry, resolver)` (which constructs the runtime
`ModelNode` + wiring and stamps span metadata onto it). Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.events import SourceRange

# Model's own field surface (was the `model` row of parser._KIND_SPECS): `model_id` is the one
# required flat field; these flat keys are allowed besides the shared `_COMMON_FIELDS` + `kind`.
MODEL_REQUIRED: frozenset[str] = frozenset({"model_id"})
MODEL_ALLOWED: frozenset[str] = frozenset(
    {"model_id", "inputs", "outputs", "weights_uri", "runtime", "asserts"}
)
# Descriptor fields carrying inline-call binding strings (the kind-blind walk in bindings.py):
# only `inputs` — `model_id`/`weights_uri`/`runtime` are static refs, not `${...}` bindings.
MODEL_BINDING_FIELDS: tuple[str, ...] = ("inputs",)


@dataclass(frozen=True)
class ModelDescriptor:
    """kind=model — ML inference (a `model_id`, optional weights/runtime).

    `range`/`field_ranges` are located-parse metadata (the node body's `SourceRange` and
    per-field `SourceRange`s); `compare=False` keeps them out of `==` so a descriptor still
    equals an identically-configured one built without ranges. Populated by `parse`.
    """

    id: str
    model_id: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    weights_uri: Optional[str] = None
    runtime: Optional[str] = None
    asserts: list[str] = field(default_factory=list)
    node_name: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    runs_after: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)  # static per-node config; merged under flow env: at build
    # Located-parse metadata (see class docstring); excluded from equality.
    range: Optional[SourceRange] = field(default=None, compare=False)
    field_ranges: dict[str, SourceRange] = field(default_factory=dict, compare=False)


def parse(node_id: str, reader: NodeReader) -> ModelDescriptor:
    """Read one `kind: model` node body (via a located `NodeReader`) into a `ModelDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` model path (allowed/required flat keys, `asserts:`/`env:` shape).
    """
    from agent_composer.compose.parser import _COMMON_FIELDS

    body = reader.value
    allowed_keys = MODEL_ALLOWED | _COMMON_FIELDS | {"kind"}
    for key in body:
        if key not in allowed_keys:
            raise LoadError(
                f"node {node_id!r} (kind=model): field {key!r} is not allowed "
                f"(allowed: {', '.join(sorted(allowed_keys))})",
                range=reader.range_of(key),
            )
    for key in MODEL_REQUIRED:
        if key not in body:
            raise LoadError(
                f"node {node_id!r} (kind=model): missing required field {key!r}",
                range=reader._own_range(),
            )

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in MODEL_ALLOWED:
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
            f"node {node_id!r} (kind=model): `asserts:` must be a list of strings",
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
                f"node {node_id!r} (kind=model): `env:` must be a mapping with string keys",
                range=reader.range_of("env"),
            )
        kwargs["env"] = env
        rng = reader.range_of("env")
        if rng is not None:
            ranges["env"] = rng

    return ModelDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(desc: ModelDescriptor, *, registry, resolver=None, **_loader_hints):
    """Construct the runtime `ModelNode` (+ wiring) for one `ModelDescriptor`.

    Mirrors the legacy `build.build_leaf_node` model path (node-side `params` + flow-owned
    `inputs` wiring, `output_type` from `outputs:`) and additionally stamps
    `.range`/`.field_ranges` from the descriptor. `resolver` is unused (leaf kind) and
    `_loader_hints` absorbs the inline-CODE traceback anchors that only the CODE build reads;
    both ride the shared `KindSpec.build` signature.
    """
    from agent_composer.compose.build import _output_type, _sink_params, _sink_wiring
    from agent_composer.nodes.model import ModelNode

    node = ModelNode(
        desc.id,
        model_id=desc.model_id,
        weights_uri=desc.weights_uri,
        runtime=desc.runtime,
        title=desc.node_name,
    )
    node.output_type = _output_type(desc.outputs, registry)
    node.params = _sink_params(desc.inputs)
    wiring = _sink_wiring(desc.inputs)
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring


def register() -> None:
    """Register the `model` kind into the shared dispatch tables (idempotent)."""
    from agent_composer.compose.kinds import KindSpec, KINDS, BINDING_FIELDS

    KINDS["model"] = KindSpec(descriptor_cls=ModelDescriptor, parse=parse, build=build)
    BINDING_FIELDS["model"] = MODEL_BINDING_FIELDS


register()
