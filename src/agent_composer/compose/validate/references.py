"""Graph-stage: reference-wiring validation over the built flow.

Cross-node assembly pass. The analogue of `compile.validation._collect_reference_errors`,
walking the BUILT flow (not a `FlowSpec`): it CALLS the leaf checkers (`_classify_path`)
directly — supplying the `valid_targets`/`flow_inputs` sets and a semantic-Type
projection of the producer contracts — only reimplementing the per-node walk glue.
Every located problem accumulates; one `LoadError` is raised carrying them all.

Two subtleties mirror the representation-neutral collector:

- **Prompt scope**: an AGENT prompt may interpolate ONLY this node's declared inputs as
  bare `${name}`; a `${<id>.output}`/`${input.X}`/`${system.X}` in a prompt is a located error.
- **Case node-local refs**: the desugared CASE `when:` strings use `${__rN}`/`${__on}`,
  which resolve against the bound input record — they are EXCLUDED here (only the
  `__rN`/`__on` SOURCES are classified, against the pool; mirrors the strict CASE check).

Located parse metadata: each error anchors on the referencing field's `SourceRange`, read
off the built node's `field_ranges` (the coarse per-field span — e.g. the node's `input:`
mapping for an input-binding ref, its `prompt:` for a prompt ref).

Imports flow down only: `compile.validation` (the leaf ref checker), `expr` (the binding
parser), `nodes` (Node/AgentNode/WaitNode discrimination and `ValueContract`),
`compose.helpers.errors`.
Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.validation import _classify_path
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.events import SourceRange
from agent_composer.expr import (
    ExpressionError,
    expr_refs_of,
    prompt_refs,
)
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import ValueContract, binding_source
from agent_composer.nodes.wait import WaitNode

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
    producers: dict[str, ValueContract],
    outputs: Any,
    flow_wiring: dict[str, dict[str, Any]],
    *,
    outputs_range: Optional[SourceRange] = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Validate every `${...}` reference site of a built flow.

    `nodes` is the built node map — leaf runtime `Node`s AND desugared `CaseNode`s
    (their `.inputs` carry the `__rN`/`__on` SOURCES, the reconciled `case` data refs).
    `flow_inputs` is the `read_flow_inputs(...)` decl names (the `${input.X}` set);
    `producers` maps node id -> its exact `output_contract`. Only producers with a
    resolvable contract belong here; an absent producer contract stays lenient. The
    existing structural reference walker receives only each contract's semantic `.type`,
    leaving its origins attached to the untouched carrier. `outputs` is the raw flow
    `outputs:` section.

    Each error anchors on the referencing site's `SourceRange`, resolved off the built node's
    `field_ranges` — the OFFENDING ENTRY where one exists (`("input", name)`, stamped per
    binding by the parse loop for every kind), else the enclosing block (`("field", "inputs")`
    / `("field", "prompt")`), else the node header. `outputs_range` is the authored output
    value/block range when the caller retains it. The aggregate carries the first retained range.

    Accumulates ALL located errors and raises a single `LoadError`; passes silently
    on a clean flow. Reuses `_classify_path` (head-resolution + dotted walk)
    verbatim — the per-node walk glue is the only loader-specific code.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    valid_targets = set(nodes)
    # The representation-neutral leaf checker speaks Type. Project only at that boundary;
    # the canonical ValueContract carriers (and their origins) stay untouched here.
    for node_id, contract in producers.items():
        if type(contract) is not ValueContract:
            raise TypeError(
                "producer contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {node_id!r}"
            )
    producer_types = {
        node_id: contract.type for node_id, contract in producers.items()
    }
    errors: list[tuple[str, Optional[SourceRange]]] = []

    def scan(
        from_value: Any,
        where: str,
        rng: Optional[SourceRange],
        extra_heads: tuple = (),
    ) -> None:
        """Name-check every reference a BINDING `from:` reads (coalesce + nested-default
        split by `expr_refs_of`); a non-string source / literal yields no refs.

        `extra_heads` is the per-site set of body-local scopes treated as lenient — e.g.
        a MAP per-element input's `${item}` (valid only inside `map.inputs`, not `over:`)."""
        from_value = binding_source(from_value)
        if not isinstance(from_value, str):
            return
        try:
            paths = expr_refs_of(from_value)
        except ExpressionError as exc:
            errors.append((f"{where}: {exc}", rng))
            return
        for path in paths:
            err = _classify_path(
                path, valid_targets, flow_inputs, extra_heads, producer_types
            )
            if err is not None:
                errors.append((f"{where}: {err}", rng))

    for node_id, node in nodes.items():
        franges = node.field_ranges
        # Prompt scope: a strict AGENT prompt may interpolate ONLY this node's
        # declared inputs as bare `${name}` (or as a `${name}`-wrapped builtin-call arg);
        # a pool ref in a prompt is loud, and an unknown builtin / malformed span is
        # rejected here (via `prompt_refs`, the renderer's compile-time companion).
        if isinstance(node, AgentNode) and node.prompt:
            prompt_rng = franges.get(("field", "prompt")) or node.range
            declared = {p.name for p in (node.params or [])}
            try:
                refs = prompt_refs(node.prompt)
            except ExpressionError as exc:
                errors.append((f"node {node_id!r} prompt: {exc}", prompt_rng))
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
                        prompt_rng,
                    ))
        # A per-item binder (MAP) reads a parent-pool `list[T]` from its reserved wiring key(s)
        # (`over` — no `${item}` there); each per-element input MAY use `${item}` (the body-local
        # element scope, lenient).
        if node.binds_per_item:
            nwiring = flow_wiring.get(node_id, {})
            reserved = node.reserved_wiring_keys()
            input_rng = franges.get(("field", "inputs")) or node.range
            for key in reserved:
                # A reserved wiring key (`over`) is a KIND FIELD, so its range is tagged
                # `("field", key)` — not the bare key, which would never match and would
                # silently anchor the error on the whole `input:` block instead.
                over_rng = franges.get(("field", key)) or input_rng
                scan(nwiring.get(key), f"node {node_id!r} map {key}", over_rng)
            for param, src in nwiring.items():
                if param in reserved:
                    continue  # the iteration source, scanned above (not a per-element input)
                scan(
                    src,
                    f"node {node_id!r} map input {param!r} from",
                    franges.get(("input", param)) or input_rng,
                    extra_heads=("item",),
                )
            continue
        if isinstance(node, WaitNode):
            continue  # WAIT's `until` is reserved wiring outside the ordinary param scan
        # Each node input `from:` source, read from the flow-owned wiring. For a
        # desugared CASE these are the `__rN`/`__on` SOURCES (the original
        # `${<id>.output}` refs) — validated against the pool. The rewritten node-local
        # `${__rN}`/`${__on}` live in `node.cases` `when:` and are EXCLUDED here
        # (they resolve against the record).
        # Anchor each error on the OFFENDING ENTRY, not the whole `input:` mapping: the parse
        # loop stamps a per-binding `("input", name)` range for every kind, and a desugar stamps
        # one for its synthetic params (a CASE `__rN` -> its `cases[N].when` line); the block range
        # is only the fallback (a spanless flow has no ranges at all). `wiring_ref_label` gives a
        # synthetic param the author-facing name it was hoisted from instead of leaking `__rN`.
        input_rng = franges.get(("field", "inputs")) or node.range
        for param, src in flow_wiring.get(node_id, {}).items():
            scan(src, f"node {node_id!r} {node.wiring_ref_label(param)} from",
                 franges.get(("input", param)) or input_rng)

    # Flow outputs (the codomain): each binding `from:` is checked like a node input and
    # located at its retained value or block range. No built node owns this source site;
    # source-free callers deliberately remain unlocated.
    for value in _output_bindings(outputs):
        scan(value, "flow output from", outputs_range)

    if errors:
        first_range = next(
            (error_range for _, error_range in errors if error_range is not None),
            None,
        )
        raise diagnostics.error(
            "flow has unresolved references:\n  "
            + "\n  ".join(msg for msg, _ in errors),
            range=first_range,
        )
