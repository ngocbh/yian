"""Graph-stage: reference-wiring validation over the built flow.

Cross-node assembly pass. The analogue of `compile.validation._collect_reference_errors`,
walking the BUILT flow (not a `FlowSpec`): it CALLS the leaf checkers (`_classify_path`)
directly — supplying the `valid_targets`/`flow_inputs` sets and a semantic-Type
projection of the producer contracts — only reimplementing the per-node walk glue.
Every located problem accumulates; one `LoadError` is raised carrying them all.

Two subtleties mirror the representation-neutral collector:

- **Prompt scope**: an AGENT `prompt:` or `system_prompt:` may interpolate this node's inputs
  and variables as bare `${name}`; a `${<id>.output}`/`${input.X}`/`${system.X}` in either is a
  located error. A HUMAN_INPUT `prompt:` and its question text (`question`, option
  `label`/`description`) follow the same rule, reading the gate's declared inputs.
- **Case node-local refs**: the desugared CASE `when:` strings use `__rN`/`__on`,
  which resolve against the bound input record — they are EXCLUDED here (only the
  `__rN`/`__on` SOURCES are classified, against the pool; mirrors the strict CASE check).

Located parse metadata: each error anchors on the referencing field's `SourceRange`, read
from the transient Compose `BuildResult` (the coarse per-field span — e.g. the node's
`input:` mapping for an input-binding ref, its `prompt:` or `system_prompt:` for a template ref).

Imports flow down only: `compile.validation` (the leaf namespace checker), `expr` (the
binding parser), `nodes` (Node/AgentNode/HumanInputNode behavior and the output-contract
sum), `typesys` (`Type`, for a list's element type), and `compose.helpers` (closed-path
projection, binding-contract resolution, and errors).
Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Any, Optional

from yian.compile.validation import _classify_path
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.output_contracts import (
    output_contract_path_error,
    resolve_output_binding_contract,
)
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.narrowing import Narrowing
from yian.diagnostics import SourceRange
from yian.expr import (
    ExpressionError,
    expr_refs_of,
    prompt_refs,
)
from yian.nodes.agent import AgentNode
from yian.nodes.contracts import (
    OutputContract,
    PartialRecordContract,
    ValueContract,
    binding_source,
)
from yian.nodes.human_input import HumanInputNode
from yian.nodes.human_input.node import site_label, template_sites
from yian.typesys import Type

# Heads forbidden in a strict prompt (always trigger the bespoke hint).
# `input`/`system`/`item` are the pool-style heads; a node-id head (e.g. `${other.output}`)
# is detected separately by `head in valid_targets`.
_PROMPT_FORBIDDEN_HEADS = ("input", "system", "item")


def _output_bindings(outputs: Any) -> list[Any]:
    """The flow `output:` section as a flat list of binding values (sites to check).

    A whole-string binding (`${a.output | b.output}`) is one value; a
    name -> binding map is each mapped value. Mirrors
    `build._output_bindings` (kept local — this pass does not import `build`).
    """
    if isinstance(outputs, dict):
        return list(outputs.values())
    if outputs is None:
        return []
    return [outputs]


def _item_contract(
    over: Any,
    producers: dict[str, OutputContract],
    input_contracts: dict[str, ValueContract],
) -> ValueContract | None:
    """The element contract of a MAP's `over:` binding when it is a proven list, else `None`.

    Args:
        over (`Any`):
            The MAP's raw `over:` binding.
        producers (`dict[str, OutputContract]`):
            Exact or partial node output contracts keyed by producer id.
        input_contracts (`dict[str, ValueContract]`):
            Exact flow input contracts keyed by input name.

    Returns:
        `ValueContract | None`:
            The exact contract of one element, or `None` when `over:` is not a proven list
            or its element type is unknown (`list[Any]`).

    Raises:
        `TypeError`:
            A referenced producer is not an `OutputContract`, or a flow-input contract is
            not an exact `ValueContract`.
    """
    contract = resolve_output_binding_contract(over, producers, input_contracts)
    if type(contract) is not ValueContract or not contract.type.kind.is_list():
        return None
    typ = contract.type
    if typ.element is not None:
        return ValueContract(typ.element)
    element_kind = typ.kind.element_kind
    if element_kind is None:
        return None
    return ValueContract(Type.scalar(element_kind))


def validate_references(
    build_results: dict[str, BuildResult],
    flow_inputs: "set[str]",
    producers: dict[str, OutputContract],
    outputs: Any,
    flow_wiring: dict[str, dict[str, Any]],
    *,
    input_contracts: dict[str, ValueContract] | None = None,
    outputs_range: Optional[SourceRange] = None,
    narrowing: Narrowing | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Validate every `${...}` reference site of a built flow.

    `build_results` is the transient Compose build map — its nodes include leaf runtime
    `Node`s and desugared `CaseNode`s (their wiring carries the `__rN`/`__on` sources), while
    its range fields retain YAML locations only for this validation phase.
    `flow_inputs` is the `read_flow_inputs(...)` decl names (the `${input.X}` set);
    `input_contracts` maps those names to their exact contracts, so `${input.X.f}` is
    field-checked too (`None` checks names only). `producers` maps node id -> its exact or
    partial `output_contract`. An absent contract stays lenient. The namespace checker
    validates the reference head, then the shared output-contract helper checks exact fields
    or the partial arm's closed root name set without inventing a type for opaque fields.
    A MAP's per-element `${item.f}` is field-checked against the element type of `over:`
    when that is a proven list. `outputs` is the raw flow `output:` section.

    Each error anchors on the referencing site's `SourceRange`, resolved from its
    `BuildResult.field_ranges` — the OFFENDING ENTRY where one exists (`("input", name)`, stamped per
    binding by the parse loop for every kind), else the enclosing block (`("field", "inputs")`
    / `("field", "prompt")`), else the node header. `outputs_range` is the authored output
    value/block range when the caller retains it. The aggregate carries the first retained range.

    With `narrowing`, each node's references (a CASE's own sources included) are checked
    against that node's view of the producer and flow-input contracts, so a branch can
    read the fields of the union members that reach it. A node a topological order could
    not reach skips the member-only field rule, so the later cycle check reports the cycle.
    The flow outputs are always checked unnarrowed.

    Accumulates ALL located errors and raises a single `LoadError`; passes silently
    on a clean flow. Reuses `_classify_path` (head-resolution + dotted walk)
    verbatim — the per-node walk glue is the only loader-specific code.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    input_contracts = input_contracts or {}
    valid_targets = set(build_results)
    # Keep exact and partial carriers intact; the Compose projection below checks
    # closed record paths without inventing a Type for opaque fields.
    for node_id, contract in producers.items():
        if type(contract) not in (ValueContract, PartialRecordContract):
            raise TypeError(
                "producer contracts must be ValueContract or PartialRecordContract values, "
                f"got {type(contract).__name__} for {node_id!r}"
            )
    errors: list[tuple[str, Optional[SourceRange]]] = []
    unordered = narrowing.unordered if narrowing is not None else frozenset()

    def scan(
        from_value: Any,
        where: str,
        rng: Optional[SourceRange],
        *,
        site_producers: dict[str, OutputContract],
        site_inputs: dict[str, ValueContract],
        item_contract: ValueContract | None = None,
        extra_heads: tuple = (),
        partial_union_fields: bool = False,
    ) -> None:
        """Name-check every reference a BINDING `from:` reads (coalesce + nested-default
        split by `expr_refs_of`); a non-string source / literal yields no refs.

        `site_producers` and `site_inputs` are the producer and flow-input contracts this
        site sees (a node's narrowed view, or the unnarrowed maps), used for field checks.
        `extra_heads` is the per-site set of body-local scopes treated as lenient — e.g.
        a MAP per-element input's `${item}` (valid only inside `map.inputs`, not `over:`);
        `item_contract` is that `${item}`'s element contract when known, for field checks.
        `partial_union_fields` lets a field only some union members have be read."""
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
                path, valid_targets, flow_inputs, extra_heads
            )
            parts = path.split(".")
            if (
                err is None
                and len(parts) >= 2
                and parts[1] == "output"
                and parts[0] in site_producers
            ):
                err = output_contract_path_error(
                    site_producers[parts[0]],
                    parts[2:],
                    path,
                    partial_union_fields=partial_union_fields,
                )
            elif (
                err is None
                and len(parts) >= 3
                and parts[0] == "input"
                and parts[1] in site_inputs
            ):
                err = output_contract_path_error(
                    site_inputs[parts[1]],
                    parts[2:],
                    path,
                    partial_union_fields=partial_union_fields,
                )
            elif (
                err is None
                and len(parts) >= 2
                and parts[0] == "item"
                and item_contract is not None
            ):
                err = output_contract_path_error(
                    item_contract,
                    parts[1:],
                    path,
                    partial_union_fields=partial_union_fields,
                )
            if err is not None:
                errors.append((f"{where}: {err}", rng))

    def check_template(
        where: str,
        template: str,
        declared: "set[str]",
        rng: Optional[SourceRange],
        *,
        noun: str,
        hint_tail: str,
    ) -> None:
        """Parse one rendered template and name-check each ref against `declared`.

        A malformed span, unknown builtin, or nested `${...}` is one error at `where`; each
        ref whose head is not in `declared` is another, with the pool-ref hint (ending in
        `hint_tail`) when the head is a pool-style head or a node id."""
        try:
            refs = prompt_refs(template)
        except ExpressionError as exc:
            errors.append((f"{where}: {exc}", rng))
            return
        for ref in refs:
            head = ref.split(".", 1)[0].strip()
            if head not in declared:
                hint = (
                    f" (a pool ref belongs in an input `from:`, not {hint_tail})"
                    if head in _PROMPT_FORBIDDEN_HEADS or head in valid_targets
                    else ""
                )
                errors.append((f"{where}: ${{{ref}}} is not a declared {noun}{hint}", rng))

    for node_id, result in build_results.items():
        node = result.node
        franges = result.field_ranges
        node_range = result.node_range
        node_producers, node_inputs = (
            narrowing.for_node(node_id)
            if narrowing is not None
            else (producers, input_contracts)
        )
        partial_union_fields = node_id in unordered
        # Prompt scope: a strict AGENT prompt or system prompt may interpolate ONLY
        # this node's declared inputs and variables as `${name}` (or a builtin-call
        # argument); a pool ref in either is loud, and an unknown builtin / malformed
        # span is rejected here (via `prompt_refs`, the renderer's compile-time companion).
        # A HUMAN_INPUT `prompt:` and its question text (`question`, option `label` /
        # `description`) follow the same rule, reading the gate's declared inputs.
        if isinstance(node, AgentNode):
            declared = {p.name for p in (node.params or [])}
            declared.update(variable.name for variable in node.variables)
            for field_name, template in (
                ("prompt", node.prompt),
                ("system_prompt", node.system_prompt),
            ):
                if not template:
                    continue
                check_template(
                    f"node {node_id!r} {field_name}",
                    template,
                    declared,
                    franges.get(("field", field_name)) or node_range,
                    noun="input or variable",
                    hint_tail="a strict AGENT prompt",
                )
        elif isinstance(node, HumanInputNode):
            declared = {p.name for p in node.params or []}
            for path, text in template_sites(node.prompt, node.questions):
                if not text:
                    continue
                check_template(
                    f"node {node_id!r} {site_label(path)}",
                    text,
                    declared,
                    franges.get(("field", path[0])) or node_range,
                    noun="input",
                    hint_tail="human_input text",
                )
        # Node-owned reserved wiring (`over`, `until`, ...) follows the same pool reference
        # grammar as ordinary inputs. It never receives a body-local scope: in particular,
        # MAP's `${item}` is valid only in each per-element input, not in `over:`.
        nwiring = flow_wiring.get(node_id, {})
        reserved = node.reserved_wiring_keys()
        input_rng = franges.get(("field", "inputs")) or node_range
        for key in sorted(reserved):
            # A reserved wiring key is a KIND FIELD, so its range is tagged
            # `("field", key)` — not the bare key, which would silently fall back to the
            # whole `input:` block or node header.
            reserved_rng = franges.get(("field", key)) or input_rng
            scan(
                nwiring.get(key),
                f"node {node_id!r} {node.kind.value} {key}",
                reserved_rng,
                site_producers=node_producers,
                site_inputs=node_inputs,
                partial_union_fields=partial_union_fields,
            )

        # A per-item binder (MAP) reads its iteration source above, then validates each
        # element input with the body-local `${item}` scope enabled.
        if node.binds_per_item:
            item_contract = _item_contract(nwiring.get("over"), node_producers, node_inputs)
            for param, src in nwiring.items():
                if param in reserved:
                    continue
                scan(
                    src,
                    f"node {node_id!r} map input {param!r} from",
                    franges.get(("input", param)) or input_rng,
                    site_producers=node_producers,
                    site_inputs=node_inputs,
                    item_contract=item_contract,
                    extra_heads=("item",),
                    partial_union_fields=partial_union_fields,
                )
            continue

        # Each node input `from:` source, read from the flow-owned wiring. For a
        # desugared CASE these are the `__rN`/`__on` SOURCES (the original
        # `${<id>.output}` refs) — validated against the pool. The rewritten node-local
        # `__rN`/`__on` live in `node.cases` `when:` and are EXCLUDED here
        # (they resolve against the record).
        # Anchor each error on the OFFENDING ENTRY, not the whole `input:` mapping: the parse
        # loop stamps a per-binding `("input", name)` range for every kind, and a desugar stamps
        # one for its synthetic params (a CASE `__rN` -> its `cases[N].when` line); the block range
        # is only the fallback (a spanless flow has no ranges at all). `wiring_ref_label` gives a
        # synthetic param the author-facing name it was hoisted from instead of leaking `__rN`.
        for param, src in nwiring.items():
            if param in reserved:
                continue
            label = node.wiring_ref_label(param)
            param_rng = franges.get(("input", param)) or input_rng
            param_producers, param_inputs = node_producers, node_inputs
            if narrowing is not None:
                # A searched CASE's param read only after an `isinstance` check has its own
                # view; one read guarded first and unguarded later reports at the later read.
                param_producers, param_inputs = narrowing.for_param(node_id, param)
                site = narrowing.param_site(node_id, param)
                if site is not None:
                    label, param_rng = site[0], site[1] or param_rng
            scan(
                src,
                f"node {node_id!r} {label} from",
                param_rng,
                site_producers=param_producers,
                site_inputs=param_inputs,
                partial_union_fields=partial_union_fields,
            )

    # Flow outputs (the codomain): each binding `from:` is checked like a node input and
    # located at its retained value or block range. No built node owns this source site;
    # source-free callers deliberately remain unlocated.
    for value in _output_bindings(outputs):
        scan(
            value,
            "flow output from",
            outputs_range,
            site_producers=producers,
            site_inputs=input_contracts,
        )

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
