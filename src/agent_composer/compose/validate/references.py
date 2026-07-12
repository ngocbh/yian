"""Graph-stage: reference-wiring validation over the built flow.

Cross-node assembly pass. The analogue of `compile.validation._collect_reference_errors`,
walking the BUILT flow (not a `FlowSpec`): it CALLS the leaf checkers (`_classify_path`)
directly — supplying the `valid_targets`/`flow_inputs` sets + the `producers` Type-map —
only reimplementing the per-node walk glue. Every located problem accumulates; one
`LoadError` is raised carrying them all.

Two subtleties mirror the legacy collector:

- **Prompt scope**: an AGENT prompt may interpolate ONLY this node's declared inputs as
  bare `${name}`; a `${<id>.output}`/`${input.X}`/`${system.X}` in a prompt is a located error.
- **Case node-local refs**: the desugared CASE `when:` strings use `${__rN}`/`${__on}`,
  which resolve against the bound input record — they are EXCLUDED here (only the
  `__rN`/`__on` SOURCES are classified, against the pool; mirrors the strict CASE check).

Located parse metadata: each error anchors on the referencing field's `SourceRange`, read
off the built node's `field_ranges` (the coarse per-field span — e.g. the node's `input:`
mapping for an input-binding ref, its `prompt:` for a prompt ref). The transitional
`node_lines`/`outputs_line` line-maps still feed `LoadError.line` for the line-based CLI
renderer; a later renderer collapse retires them for the range.

Imports flow down only: `compile.validation` (the leaf ref checker), `expr` (the binding
parser), `nodes` (Node/AgentNode/WaitNode discrimination), `typesys` (Type), `compose.helpers.errors`.
Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.validation import _classify_path
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange
from agent_composer.expr import (
    ExpressionError,
    expr_refs_of,
    prompt_refs,
)
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Node
from agent_composer.nodes.wait import WaitNode
from agent_composer.typesys.values import Type

# Heads forbidden in a strict prompt (always trigger the bespoke hint).
# `input`/`system`/`item` are the pool-style heads; a node-id head (e.g. `${other.output}`)
# is detected separately by `head in valid_targets`.
_PROMPT_FORBIDDEN_HEADS = ("input", "system", "item")


def _output_bindings(outputs: Any) -> list[Any]:
    """The flow `outputs:` section as a flat list of binding values (sites to check).

    A whole-string binding (`${a.output | b.output}`) is one value; a
    name -> binding map is each mapped value. Mirrors
    `build._output_bindings` (kept local — this pass does not import `build`).
    """
    if isinstance(outputs, dict):
        return list(outputs.values())
    if outputs is None:
        return []
    return [outputs]


def validate_references(
    nodes: dict[str, Node],
    flow_inputs: "set[str]",
    producers: dict[str, Type],
    outputs: Any,
    flow_wiring: dict[str, dict[str, Any]],
    *,
    node_lines: "dict[str, int] | None" = None,
    outputs_line: "int | None" = None,
) -> None:
    """Validate every `${...}` reference site of a built flow.

    `nodes` is the built node map — leaf runtime `Node`s AND desugared `CaseNode`s
    (their `.inputs` carry the `__rN`/`__on` SOURCES, the reconciled `case` data refs).
    `flow_inputs` is the `read_flow_inputs(...)` decl names (the `${input.X}` set);
    `producers` maps node id -> its `output_type` (drives the dotted-field walk;
    only resolvable/checked producers belong here — an opaque/None producer stays
    lenient). `outputs` is the raw flow `outputs:` section.

    Each error anchors on the referencing field's `SourceRange`, resolved off the built
    node's `field_ranges` (the node's `input:` mapping for an input ref, its `prompt:` for a
    prompt ref). The transitional `node_lines` (node id -> source line) + `outputs_line`
    (the `outputs:` section line) still feed `LoadError.line` for the line-based renderer.
    The aggregate `LoadError` carries the FIRST located error's line AND range (best-effort,
    since several sites may be wrong at once).

    Accumulates ALL located errors and raises a single `LoadError`; passes silently
    on a clean flow. Reuses `_classify_path` (head-resolution + dotted walk)
    verbatim — the per-node walk glue is the only loader-specific code.
    """
    lines = node_lines or {}
    valid_targets = set(nodes)
    # Only resolvable producers participate in the dotted walk (opaque -> lenient).
    producer_types = {nid: sh for nid, sh in producers.items() if sh is not None}
    # (message, line, range) triples — line/range locate the offending site where known.
    errors: list[tuple[str, "int | None", Optional[SourceRange]]] = []

    def scan(
        from_value: Any,
        where: str,
        line: "int | None",
        rng: Optional[SourceRange],
        extra_heads: tuple = (),
    ) -> None:
        """Name-check every reference a BINDING `from:` reads (coalesce + nested-default
        split by `expr_refs_of`); a non-string source / literal yields no refs.

        `extra_heads` is the per-site set of body-local scopes treated as lenient — e.g.
        a MAP per-element input's `${item}` (valid only inside `map.inputs`, not `over:`)."""
        if not isinstance(from_value, str):
            return
        try:
            paths = expr_refs_of(from_value)
        except ExpressionError as exc:
            errors.append((f"{where}: {exc}", line, rng))
            return
        for path in paths:
            err = _classify_path(
                path, valid_targets, flow_inputs, extra_heads, producer_types
            )
            if err is not None:
                errors.append((f"{where}: {err}", line, rng))

    for node_id, node in nodes.items():
        node_line = lines.get(node_id)
        franges = node.field_ranges
        # Prompt scope: a strict AGENT prompt may interpolate ONLY this node's
        # declared inputs as bare `${name}` (or as a `${name}`-wrapped builtin-call arg);
        # a pool ref in a prompt is loud, and an unknown builtin / malformed span is
        # rejected here (via `prompt_refs`, the renderer's compile-time companion).
        if isinstance(node, AgentNode) and node.prompt:
            prompt_rng = franges.get("prompt") or node.range
            declared = {p.name for p in (node.params or [])}
            try:
                refs = prompt_refs(node.prompt)
            except ExpressionError as exc:
                errors.append((f"node {node_id!r} prompt: {exc}", node_line, prompt_rng))
                refs = []
            for ref in refs:
                head = ref.split(".", 1)[0].strip()
                if head not in declared:
                    hint = (
                        " (a pool ref belongs in an input `from:`, not a strict AGENT prompt)"
                        if head in _PROMPT_FORBIDDEN_HEADS or head in valid_targets
                        else ""
                    )
                    errors.append((
                        f"node {node_id!r} prompt: ${{{ref}}} is not a declared input{hint}",
                        node_line,
                        prompt_rng,
                    ))
        # A per-item binder (MAP) reads a parent-pool `list[T]` from its reserved wiring key(s)
        # (`over` — no `${item}` there); each per-element input MAY use `${item}` (the body-local
        # element scope, lenient).
        if node.binds_per_item:
            nwiring = flow_wiring.get(node_id, {})
            reserved = node.reserved_wiring_keys()
            input_rng = franges.get("inputs") or node.range
            for key in reserved:
                over_rng = franges.get(key) or input_rng
                scan(nwiring.get(key), f"node {node_id!r} map {key}", node_line, over_rng)
            for param, src in nwiring.items():
                if param in reserved:
                    continue  # the iteration source, scanned above (not a per-element input)
                scan(
                    src,
                    f"node {node_id!r} map input {param!r} from",
                    node_line,
                    input_rng,
                    extra_heads=("item",),
                )
            continue
        if isinstance(node, WaitNode):
            continue  # a timed WAIT's `until` is out of ref-validation scope (as today)
        # Each node input `from:` source, read from the flow-owned wiring. For a
        # desugared CASE these are the `__rN`/`__on` SOURCES (the original
        # `${<id>.output}` refs) — validated against the pool. The rewritten node-local
        # `${__rN}`/`${__on}` live in `node.cases` `when:` and are EXCLUDED here
        # (they resolve against the record).
        input_rng = franges.get("inputs") or node.range
        for param, src in flow_wiring.get(node_id, {}).items():
            scan(src, f"node {node_id!r} input {param!r} from", node_line, input_rng)

    # Flow outputs (the codomain): each binding `from:` checked like a node input —
    # located at the `outputs:` section line (dangling-head/dotted-field errors point here).
    # No built node owns the flow `outputs:` block, so no range is available yet (a later
    # renderer collapse resolves it from the section span); the `outputs_line` still locates it.
    for value in _output_bindings(outputs):
        scan(value, "flow output from", outputs_line, None)

    if errors:
        first_line = next((ln for _, ln, _ in errors if ln is not None), None)
        first_range = next((r for _, _, r in errors if r is not None), None)
        raise LoadError(
            "flow has unresolved references:\n  "
            + "\n  ".join(msg for msg, _, _ in errors),
            line=first_line,
            range=first_range,
        )
