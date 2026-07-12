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
from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.compose.kinds.helpers import (
    _output_type,
    _sink_params,
    _sink_wiring,
    check_node_keys,
    read_common_fields,
)
from agent_composer.events import SourceRange
from agent_composer.nodes.model import ModelNode

# Model's own field surface (was the `model` row of parser._KIND_SPECS): `model_id` is the one
# required flat field; these flat keys are allowed besides the shared `_COMMON_FIELDS` + `kind`.
MODEL_REQUIRED: frozenset[str] = frozenset({"model_id"})
MODEL_ALLOWED: frozenset[str] = frozenset(
    {"model_id", "inputs", "outputs", "weights_uri", "runtime", "asserts"}
)
# Descriptor fields carrying inline-call binding strings (the kind-blind walk in bindings.py):
# only `inputs` — `model_id`/`weights_uri`/`runtime` are static refs, not `${...}` bindings.
MODEL_BINDING_FIELDS: tuple[str, ...] = ("inputs",)


@dataclass(frozen=True, kw_only=True)
class ModelDescriptor(NodeDescriptor):
    """kind=model — ML inference (a `model_id`, optional weights/runtime).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`; only the model-specific
    fields are declared here.
    """

    model_id: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    weights_uri: Optional[str] = None
    runtime: Optional[str] = None
    asserts: list[str] = field(default_factory=list)


def parse(node_id: str, reader: NodeReader) -> ModelDescriptor:
    """Read one `kind: model` node body (via a located `NodeReader`) into a `ModelDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` model path (allowed/required flat keys, `asserts:`/`env:` shape).
    """
    check_node_keys(node_id, "model", reader, MODEL_ALLOWED, MODEL_REQUIRED)
    body = reader.value

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

    read_common_fields(node_id, "model", reader, kwargs, ranges)

    return ModelDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(desc: ModelDescriptor, *, registry, resolver=None, **_loader_hints):
    """Construct the runtime `ModelNode` (+ wiring) for one `ModelDescriptor`.

    Mirrors the legacy `build.build_leaf_node` model path (node-side `params` + flow-owned
    `inputs` wiring, `output_type` from `outputs:`) and additionally stamps
    `.range`/`.field_ranges` from the descriptor. `resolver` is unused (leaf kind) and
    `_loader_hints` absorbs the inline-CODE traceback anchors that only the CODE build reads;
    both ride the shared `KindSpec.build` signature.
    """
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
