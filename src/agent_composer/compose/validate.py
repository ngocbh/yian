"""Graph- and reference-level validation for the loader.

Reuses the representation-neutral leaf checkers from `compile.validation` —
`_walk_record_fields` (dotted-field dispatch) — on the built nodes, re-raising as a
`LoadError` so the surface speaks one error type. The handle-alignment check ports
`_check_case_handles`'s rule to the desugared `CaseNode`s + their control edges (the
Compose path has no `FlowSpec`). Directed-cycle rejection lives in the peer `stages.cycles`
pass; reference-wiring validation in `stages.references`.

This module retains the human_input questions check and the node-local `asserts:`
classifier — the per-node contract validators that still walk the built node map.

Imports flow down only: `compile.model` (Edge), `compile.validation` (the leaf
field walker), `expr` (the condition parser), `nodes` (Node/CaseNode discrimination).
Nothing imports this back.
"""

from __future__ import annotations

from typing import Any

from agent_composer.compile.model import Edge
from agent_composer.compile.validation import _walk_record_fields
from agent_composer.expr import ExpressionError, condition_refs
from agent_composer.expr.template import flow_call_callees_in_spans
from agent_composer.compose.stages.inline_calls import desugar_call_directives
from agent_composer.nodes.base import Node
from agent_composer.nodes.case import DEFAULT_HANDLE, CaseNode
from agent_composer.typesys.values import ValueKind
from agent_composer.compose.helpers.errors import LoadError


def check_case_handles(nodes: dict[str, Node], edges: list[Edge]) -> None:
    """Every desugared `CaseNode`'s handles align with its outgoing control edges.

    Mirrors `compile.validation._check_case_handles`: each outgoing edge's handle
    (defaulting to `default`) must be a declared case handle or `default`, and every
    declared case must have an outgoing edge. The `case` desugar emits the
    `gate -> <then|else>` control edges and the `CaseNode.cases`; this pins them
    consistent. Raises `LoadError` on the first misalignment.
    """
    for node_id, node in nodes.items():
        if not isinstance(node, CaseNode):
            continue
        case_handles = {c.handle for c in node.cases}
        out_handles = {
            (e.source_handle or DEFAULT_HANDLE) for e in edges if e.from_ == node_id
        }
        for handle in out_handles:
            if handle != DEFAULT_HANDLE and handle not in case_handles:
                raise LoadError(
                    f"case node {node_id!r} has edge handle {handle!r} with no matching case"
                )
        for handle in case_handles:
            if handle not in out_handles:
                raise LoadError(
                    f"case node {node_id!r} case {handle!r} has no outgoing edge"
                )


# Pool heads forbidden in a node-local assert (use bare `${name}` / `${output}` instead).
# Post-alias-delete — only the new singular heads remain.
_POOL_HEADS = frozenset({"input", "system", "item"})


def validate_node_asserts(
    nodes: dict[str, Node],
    descriptors: dict,
    node_lines: "dict[str, int] | None" = None,
) -> None:
    """Validate + classify + stamp each node's `asserts:` — the per-node contract.

    A node assert is node-LOCAL: a bare `${name}` is one of the node's declared inputs;
    `${output}`/`${output.field}` is the node's own output (dotted-walked vs `output_type`).
    Rejected (located `LoadError`): a pool head (`${outputs/inputs/system/item.X}`), an inline
    `${call}`, `${output.field}` on a non-record output, a declared input named `output`, and
    any node assert on a mapped call (a `MapNode`, `kind: map` — the per-node hook has no `${item}` scope).
    Classifies each PRE (no `${output}`) or POST (reads `${output}`) and stamps
    `node.pre_asserts` / `node.post_asserts`. Runs in the shared `_assemble`, so a `defs:`
    subflow's internal nodes get this too. Accumulates all errors into one `LoadError`.
    """
    lines = node_lines or {}
    errors: list[tuple[str, "int | None"]] = []
    for nid, node in nodes.items():
        desc = descriptors.get(nid)
        asserts = list(getattr(desc, "asserts", None) or [])
        if not asserts:
            continue
        line = lines.get(nid)
        if node.binds_per_item:
            errors.append((
                f"node {nid!r}: `asserts:` is not allowed on a mapped call (`kind: map`) — assert "
                f"inside the def, or on a downstream node", line))
            continue
        declared = {p.name for p in (node.params or [])}
        if "output" in declared:
            errors.append((
                f"node {nid!r}: a declared input named 'output' collides with the node-assert "
                f"`${{output}}` keyword — rename the input", line))
            continue
        pre: list[str] = []
        post: list[str] = []
        for a in asserts:
            # node asserts are node-local — an inline call (a whole-value `call(...)` directive
            # or the retired `${flow(args)}` span) belongs in a flow-level assert, not here.
            try:
                _, directive_calls = desugar_call_directives(a, lambda: "__a")
                span_callees = flow_call_callees_in_spans(a)
            except ExpressionError:
                directive_calls, span_callees = ["(malformed call)"], []
            if directive_calls or span_callees:
                errors.append((
                    f"node {nid!r} assert {a!r}: an inline call is not allowed in a "
                    f"node-local assert (use a flow-level assert)", line))
                continue
            # parse-check + collect refs on the ONE unified condition walk (any spelling).
            try:
                refs = condition_refs(a)
            except ExpressionError as exc:
                errors.append((
                    f"node {nid!r} assert {a!r}: not a valid boolean expression ({exc})", line))
                continue
            reads_output = False
            for ref in refs:
                segs = ref.split(".")
                head = segs[0] if segs else ""
                if head == "output":
                    reads_output = True
                    if len(segs) > 1:
                        sh = node.output_type
                        if sh is not None and sh.kind != ValueKind.OBJECT:
                            errors.append((
                                f"node {nid!r} assert: ${{{ref}}} — the node output is not a "
                                f"record, so it has no field {'.'.join(segs[1:])!r}", line))
                        else:
                            err = _walk_record_fields(sh, segs[1:], ref)
                            if err is not None:
                                errors.append((f"node {nid!r} assert: {err}", line))
                elif head in declared:
                    continue
                elif head in _POOL_HEADS:
                    errors.append((
                        f"node {nid!r} assert: ${{{ref}}} — node asserts are node-local; use "
                        f"`${{name}}` (a declared input) or `${{output}}`", line))
                else:
                    errors.append((
                        f"node {nid!r} assert: ${{{ref}}} is not a declared input or `${{output}}`",
                        line))
            (post if reads_output else pre).append(a)
        node.pre_asserts = pre
        node.post_asserts = post
    if errors:
        first_line = next((ln for _, ln in errors if ln is not None), None)
        raise LoadError(
            "flow has invalid node asserts:\n  " + "\n  ".join(msg for msg, _ in errors),
            line=first_line,
        )
