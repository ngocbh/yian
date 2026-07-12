"""Graph-stage: the TYPE-level half of a `loop`'s `'a -> 'a` contract.

A loader post-build pass (run after `check_ref_map_types`). `kinds.loop.build` checked only the
carried field-NAME sets — it lacks the resolved carried-record Types, which depend on the parent
producers/flow inputs assembled LATER. This pass resolves each carried field's Type from its
bound seed source (`${<id>.output}`/`${input.X}` over the parent producers + flow-input types,
via `_output_value_type`) and checks it is structurally compatible (`types_compatible`, C-EQUIV)
with the body OUTPUT's same-named field Type. It relocates `build.check_loop_type_contract`
verbatim, adding a `.range` anchor off the built loop node's `field_ranges["inputs"]` (the coarse
`input:` mapping span) — the transitional `node_lines` still feeds `.line`.

Imports flow down only: `compile.validation` (the type comparator), `compose.nodes.resolver`
(`_output_value_type`, the shared ref-to-Type resolver — `resolver` does not import
`compose.stages`, so this edge is acyclic), `typesys` (Type), `compose.errors`. Only the loader
(and the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.validation import types_compatible
from agent_composer.compose.nodes.resolver import _output_value_type
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange
from agent_composer.typesys.values import Type


def check_loop_type_contract(
    nodes: dict,
    producers: dict[str, Type],
    flow_input_types: dict[str, Type],
    flow_wiring: dict[str, dict[str, Any]],
    node_lines: Optional[dict] = None,
) -> None:
    """The TYPE-level half of a `loop`'s `'a -> 'a` contract (a loader post-build pass).

    `kinds.loop.build` checked only the field-NAME sets (it lacks the resolved carried-record
    Types — the seed sources' types depend on the parent producers/flow inputs assembled
    LATER). This pass, run after `check_ref_map_types`, resolves each carried field's Type
    from its bound seed source (`${<id>.output}`/`${input.X}` over the parent producers +
    flow-input types, exactly as `check_ref_map_types` via `_output_value_type`) and checks
    it is structurally compatible (`types_compatible`, C-EQUIV) with the body OUTPUT's
    same-named field Type (the body `output_type`). A literal/coalesce/opaque seed source,
    or an opaque body output field, stays lenient (skipped). Loud + located at the loop
    node's `input:` mapping range (`field_ranges["inputs"]`, else the node's own range) on a
    same-names/different-TYPES mismatch; `node_lines` still feeds the transitional `.line`."""
    lines = node_lines or {}
    for nid, node in nodes.items():
        if not node.is_loop:
            continue
        body_out = node.output_type
        # A non-record / absent body output was already caught by the field-name half in
        # kinds.loop.build; nothing typed to compare against here.
        if body_out is None or body_out.fields is None:
            continue
        for field_name, src in flow_wiring.get(nid, {}).items():
            body_field = body_out.fields.get(field_name)
            if body_field is None:
                continue  # name mismatch caught in build; nothing to type-check
            source = _output_value_type(src, producers, flow_input_types)
            if source is None:
                continue  # literal/coalesce/opaque seed source -> lenient
            if not types_compatible(source, body_field):
                rng: Optional[SourceRange] = node.field_ranges.get("inputs") or node.range
                raise LoadError(
                    f"loop node {nid!r}: carried field {field_name!r} — body output is "
                    f"{body_field.kind.value!r}, carried source is "
                    f"{source.kind.value!r} (the body must return the same 'a it reads)",
                    line=lines.get(nid),
                    range=rng,
                )
