"""Graph-stage: the cross-flow TYPE check for every `call`/`map` binding.

A loader post-build pass (run after the flow's leaves + child signatures are assembled).
Each binding's SOURCE `Type` — resolved from its `${<id>.output}`/`${input.X}` ref over the
PARENT producers + flow-input types, read from the flow-owned `wiring` — must be structurally
compatible with the CHILD input's declared `Type` (`types_compatible`, C-EQUIV). It relocates
`build.check_ref_map_types` verbatim, adding a `.range` anchor off the built call node's
`field_ranges["inputs"]` (the coarse `input:` mapping span) — the transitional `node_lines`
still feeds `.line`.

Imports flow down only: `compile.validation` (the type comparator), `expr` (the binding-null
analysis), `compose.kinds.resolver` (`_output_value_type`, the shared ref-to-Type resolver —
`resolver` does not import `compose.stages`, so this edge is acyclic), `typesys` (Type),
`compose.errors`. Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from agent_composer.compile.validation import types_compatible
from agent_composer.compose.kinds.resolver import _output_value_type
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange
from agent_composer.expr import binding_co_skips
from agent_composer.typesys.values import Type


def check_ref_map_types(
    nodes: dict,
    producers: dict[str, Type],
    flow_input_types: dict[str, Type],
    flow_wiring: dict[str, dict[str, Any]],
    node_lines: Optional[dict] = None,
) -> None:
    """Cross-flow type check for every `call` binding (a loader post-build pass).

    Each binding's SOURCE `Type` (resolved from its `${<id>.output}`/`${input.X}` ref over the
    PARENT producers + flow-input types — read from the flow-owned `wiring`) must be
    structurally compatible with the CHILD input's declared `Type` (`types_compatible`,
    C-EQUIV). A `${item}`/coalesce/literal/opaque source — or an opaque/absent child input
    (incl. the reserved `over` key) — stays lenient (skipped). Loud + located at the
    node's `input:` mapping range (`field_ranges["inputs"]`, else the node's own range);
    `node_lines` still feeds the transitional `.line`. Iterates the WIRING (not the params)
    so a mis-named binding still surfaces here when its child decl is absent rather than
    being silently skipped."""
    lines = node_lines or {}
    for nid, node in nodes.items():
        # CALL/MAP carry `child_inputs` to type-check against; a LOOP also carries a child but
        # its `'a -> 'a` contract is checked by check_loop_type_contract/carried_types instead.
        # A leaf has no `child_inputs` attr, and a zero-input child gives [] (falsy) -> skip
        # (behavior-identical to the old empty-child_decls loop, which ran zero checks).
        if node.is_loop or not getattr(node, "child_inputs", None):
            continue
        child_decls = {d.name: d for d in node.child_inputs}
        for param, src in flow_wiring.get(nid, {}).items():
            decl = child_decls.get(param)
            if decl is None or decl.type is None:
                continue  # `over`/name-caught-in-build/opaque child input -> lenient
            source = _output_value_type(src, producers, flow_input_types)
            if source is None:
                continue  # ${item}/coalesce/literal/opaque source -> lenient
            # A nullable source is fine for a non-nullable child input ONLY when the
            # binding GUARANTEES a non-null value — a non-null literal escape or a `:?`
            # required (both fire on null, present-null included). A ref-default `:-${y}`
            # may itself be null, and the child's own `default:` does NOT cover it: a
            # present-null source SHADOWS the child default (apply_defaults fills only
            # OMITTED inputs, never a bound null). `not binding_co_skips` is exactly that
            # non-null guarantee (a `|` coalesce of refs stays strict).
            if (
                source.nullable
                and not decl.type.nullable
                and not binding_co_skips(src)
            ):
                source = replace(source, nullable=False)
            if not types_compatible(source, decl.type):
                rng: Optional[SourceRange] = node.field_ranges.get("inputs") or node.range
                raise LoadError(
                    f"call node {nid!r}: binding {param!r} — child expects "
                    f"{decl.type.kind.value!r}, source is {source.kind.value!r}",
                    line=lines.get(nid),
                    range=rng,
                )
