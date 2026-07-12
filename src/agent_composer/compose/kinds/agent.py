"""kind=agent — the LLM-reasoner node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `agent` node's
frozen `AgentDescriptor`, its located `parse(node_id, reader)` (which records each
field's `SourceRange` into the descriptor), and its `build(desc, *, registry, resolver)`
(which constructs the runtime `AgentNode` + wiring and stamps span metadata onto it).
Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
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
    _validate_llm_config,
    check_node_keys,
    read_common_fields,
)
from agent_composer.events import SourceRange
from agent_composer.nodes.agent import AgentNode
from agent_composer.typesys.values import Type

# Agent's own field surface (was the `agent` row of parser._KIND_SPECS): no required flat
# field; these flat keys are allowed besides the shared `_COMMON_FIELDS` + `kind`.
AGENT_REQUIRED: frozenset[str] = frozenset()
AGENT_ALLOWED: frozenset[str] = frozenset(
    {"inputs", "outputs", "prompt", "tools", "controls", "mode", "llm_config", "retries", "asserts"}
)
# Descriptor fields that carry inline-call binding strings (the kind-blind walk in
# bindings.py, replacing calls.map_binding_strings_in_descriptor). Only `inputs` — a
# `prompt:`'s `${...}` refs are handled by template scanning, not the inline-call desugar.
AGENT_BINDING_FIELDS: tuple[str, ...] = ("inputs",)


@dataclass(frozen=True, kw_only=True)
class AgentDescriptor(NodeDescriptor):
    """kind=agent — an LLM reasoner (prompt + mode + optional tools/controls/llm).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`; only the agent-specific
    fields are declared here.
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    prompt: Optional[str] = None
    tools: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    mode: str = "tool_calling"
    llm_config: dict[str, Any] = field(default_factory=dict)
    retries: int = 2
    asserts: list[str] = field(default_factory=list)
    # INTERNAL — never parsed from YAML; set only by the adaptive_questions desugar pass.
    # When present it overrides the type-string derived `output_type` for a synthesized
    # structured agent whose codomain is a code-built Type with no surface type-string.
    output_type_override: Optional[Type] = None


def parse(node_id: str, reader: NodeReader) -> AgentDescriptor:
    """Read one `kind: agent` node body (via a located `NodeReader`) into an `AgentDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` agent path (allowed/required flat keys, `asserts:`/`env:` shape).
    """
    check_node_keys(node_id, "agent", reader, AGENT_ALLOWED, AGENT_REQUIRED)
    body = reader.value

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in AGENT_ALLOWED:
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
            f"node {node_id!r} (kind=agent): `asserts:` must be a list of strings",
            range=ranges.get("asserts"),
        )

    read_common_fields(node_id, "agent", reader, kwargs, ranges)

    return AgentDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(desc: AgentDescriptor, *, registry, resolver=None, **_loader_hints):
    """Construct the runtime `AgentNode` (+ wiring) for one `AgentDescriptor`.

    Mirrors the legacy `build.build_leaf_node` agent path (LLM-config validation, node-side
    `params` + flow-owned `inputs` wiring, `output_type` from `outputs:`/override) and
    additionally stamps `.range`/`.field_ranges` from the descriptor. `resolver` is unused
    (leaf kind) and `_loader_hints` absorbs the inline-CODE traceback anchors
    (`source_file`/`code_line`/`code_field_line`) that only the CODE build reads; both ride
    the shared `KindSpec.build` signature.
    """
    own_cfg, inherit = _validate_llm_config(desc.llm_config, desc.id)
    try:
        node = AgentNode(
            desc.id,
            prompt=desc.prompt or "",
            tools=list(desc.tools),
            controls=list(desc.controls),
            llm_config=own_cfg,
            llm_inherit=inherit,
            mode=desc.mode,
            retries=desc.retries,
            title=desc.node_name,
        )
    except ValueError as exc:
        # AgentNode validates mode/controls against the registries; surface a rejection as
        # the loader's one error type (loud), not a bare ValueError.
        raise LoadError(f"node {desc.id!r}: {exc}") from exc

    override = desc.output_type_override
    node.output_type = override if override is not None else _output_type(desc.outputs, registry)
    node.params = _sink_params(desc.inputs)
    wiring = _sink_wiring(desc.inputs)
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring
