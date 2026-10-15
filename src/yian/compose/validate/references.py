"""Graph-stage: reference-wiring validation over the built flow.

Cross-node assembly pass. The analogue of `compile.validation._collect_reference_errors`,
walking the BUILT flow (not a `FlowSpec`): it CALLS the leaf checkers (`_classify_path`)
directly — supplying the `valid_targets`/`flow_inputs` sets and a semantic-Type
projection of the producer contracts — only reimplementing the per-node walk glue.
Every located problem accumulates; one `LoadError` is raised carrying them all. Builtin
calls are checked at every binding and prompt site.

Two subtleties mirror the representation-neutral collector:

- **Prompt scope**: an AGENT `prompt:` or `system_prompt:` may interpolate this node's inputs
  and variables as bare `${name}`; a `${<id>.output}`/`${input.X}`/`${system.X}` in either is a
  located error. A HUMAN_INPUT `prompt:` and its question text (`question`, option
  `label`/`description`) follow the same rule, reading the gate's declared inputs. Each
  `${name.f}` read is field-checked against the name's proven contracts (its binding's,
  under the node's narrowing, or a typed variable's), and is an error only when all of
  them reject it.
- **Case node-local refs**: the desugared CASE `when:` strings use `__rN`/`__on`,
  which resolve against the bound input record — they are EXCLUDED here (only the
  `__rN`/`__on` SOURCES are classified, against the pool; mirrors the strict CASE check).

Located parse metadata: each error anchors on the referencing field's `SourceRange`, read
from the transient Compose `BuildResult` (the coarse per-field span — e.g. the node's
`input:` mapping for an input-binding ref, its `prompt:` or `system_prompt:` for a template ref).
A HUMAN_INPUT question-text error anchors on its own `("text", "questions[i]...")` string
range, else the `questions:` block. A field or input a desugar authored under another node is
named by its `BuildResult.field_labels` entry: an adaptive gate's brief is reported as the
gate's `adaptive_questions.prompt` and anchors on the brief, and a gate input its composer
copies is reported as the gate's input. A plain input binding whose site name, source, and
range were already scanned is skipped, so such a copied binding is reported once.

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
from yian.compose.helpers.isinstance_types import constructor_type_fn
from yian.compose.helpers.output_contracts import (
    TypeFn,
    local_field_error,
    output_contract_path_error,
    project_output_contract,
    reference_contract,
    resolve_output_binding_contract,
)
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.constructors import builtin_field_errors, constructor_errors
from yian.compose.validate.narrowing import Narrowing
from yian.compose.validate.typed_bindings import LocalInput, ResolveOrigin
from yian.diagnostics import SourceRange
from yian.expr import (
    ExpressionError,
    check_template_calls,
    condition_refs,
    expr_refs_of,
    prompt_refs,
    scan_template,
)
from yian.expr.template import Span
from yian.nodes.agent import AgentNode
from yian.nodes.case import CaseNode
from yian.nodes.contracts import (
    OutputContract,
    PartialRecordContract,
    ValueContract,
    ValueTemplate,
    binding_source,
    iter_source_strings,
    output_contract_type,
)
from yian.nodes.human_input import HumanInputNode
from yian.nodes.human_input.node import site_label, template_sites
from yian.typesys import Type
from yian.typesys.types import TypeRegistry

# Heads forbidden in a strict prompt (always trigger the bespoke hint).
# `input`/`system`/`item` are the pool-style heads; a node-id head (e.g. `${other.output}`)
# is detected separately by `head in valid_targets`.
_PROMPT_FORBIDDEN_HEADS = ("input", "system", "item")


def _output_bindings(outputs: Any) -> list[Any]:
    """The flow `output:` section as a flat list of binding values (sites to check).

    A whole-string binding (`${a.output | b.output}`) or a tagged single output's
    `Binding` carrier is one value; a name -> binding map is each mapped value. Mirrors
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
    type_fn: TypeFn | None = None,
) -> ValueContract | None:
    """The element contract of a MAP's `over:` binding when it is a proven list, else `None`.

    Args:
        over (`Any`):
            The MAP's raw `over:` binding.
        producers (`dict[str, OutputContract]`):
            Exact or partial node output contracts keyed by producer id.
        input_contracts (`dict[str, ValueContract]`):
            Exact flow input contracts keyed by input name.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`):
            Resolves a constructor name to its type.

    Returns:
        `ValueContract | None`:
            The exact contract of one element, or `None` when `over:` is not a list or
            is an unproven one (whose items are unknown, labelled `list(Optional(Any))`).

    Raises:
        `TypeError`:
            A referenced producer is not an `OutputContract`, or a flow-input contract is
            not an exact `ValueContract`.
    """
    contract = resolve_output_binding_contract(
        over, producers, input_contracts, type_fn=type_fn
    )
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
    local_contracts: dict[str, dict[str, LocalInput]] | None = None,
    registry: TypeRegistry | None = None,
    resolve_origin: ResolveOrigin | None = None,
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
    / `("field", "prompt")`), else the node header. A HUMAN_INPUT question-text error anchors on
    its `("text", "questions[i]...")` string range, else the `questions:` block. An AGENT prompt
    site and a plain input site are named by `BuildResult.site_name`: the `field_labels` entry
    when a desugar authored the field or binding under another node (an adaptive gate's composer
    brief reads `node '<gate>' adaptive_questions.prompt` and anchors on the brief; its copied
    input reads `node '<gate>' input '<p>'`), else `node '<id>' <site>`. A plain input binding
    whose site name, source text, and range match one already scanned is skipped, so a binding
    a desugar copied onto a synth node is reported once. `outputs_range` is the authored output
    value/block range when the caller retains it. A string nested in a value template is located
    at its own authored site instead, resolved through `resolve_origin`. The aggregate carries
    the first retained range.

    With `narrowing`, each node's references (a CASE's own sources included) are checked
    against that node's view of the producer and flow-input contracts, so a branch can
    read the fields of the union members that reach it. A node a topological order could
    not reach skips the member-only field rule, so the later cycle check reports the cycle.
    The flow outputs are always checked unnarrowed.

    With `local_contracts` (`local_input_contracts`, keyed by node id then param), each
    prompt read `${name.f}` of an AGENT or HUMAN_INPUT input is field-checked against
    that name's alternatives, and an AGENT read of a typed variable against its
    declared type; a name with no entry is not field-checked.

    With `registry`, every `${T(...)}` constructor call in a binding is checked too
    (`constructor_errors`), with each reference typed by the same contracts.

    Every builtin call in a binding or prompt must pass arguments the builtin takes.

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
    # `(site name, source, range)` of each plain input binding already scanned.
    scanned: set[tuple[str, str, Optional[SourceRange]]] = set()
    constructor_problems: list[tuple[str, Optional[SourceRange]]] = []
    unordered = narrowing.unordered if narrowing is not None else frozenset()
    local_contracts = local_contracts or {}
    type_fn = constructor_type_fn(registry) if registry is not None else None

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
        constructors: bool = True,
    ) -> None:
        """Name-check every reference a BINDING `from:` reads (coalesce + nested-default
        split by `expr_refs_of`); a non-string source / literal yields no refs.

        `site_producers` and `site_inputs` are the producer and flow-input contracts this
        site sees (a node's narrowed view, or the unnarrowed maps), used for field checks.
        `extra_heads` is the per-site set of body-local scopes treated as lenient — e.g.
        a MAP per-element input's `${item}` (valid only inside `map.inputs`, not `over:`);
        `item_contract` is that `${item}`'s element contract when known, for field checks.
        `partial_union_fields` lets a field only some union members have be read.
        `constructors` checks its `${T(...)}` calls; a reserved key, where they are not
        supported, leaves that error to the type-table pass.
        Each string nested in a `ValueTemplate` is scanned on its own and located at
        its authored site when that resolves."""
        source = binding_source(from_value)
        for value_path, text in iter_source_strings(from_value):
            text_rng = rng
            if isinstance(source, ValueTemplate) and resolve_origin is not None:
                origin = source.origin_at(value_path)
                frame = resolve_origin(origin) if origin is not None else None
                if frame is not None:
                    text_rng = frame.range
            scan_text(
                text,
                where,
                text_rng,
                site_producers=site_producers,
                site_inputs=site_inputs,
                item_contract=item_contract,
                extra_heads=extra_heads,
                partial_union_fields=partial_union_fields,
                constructors=constructors,
            )

    def scan_text(
        text: str,
        where: str,
        rng: Optional[SourceRange],
        *,
        site_producers: dict[str, OutputContract],
        site_inputs: dict[str, ValueContract],
        item_contract: ValueContract | None,
        extra_heads: tuple,
        partial_union_fields: bool,
        constructors: bool,
    ) -> None:
        """Name-check every reference one template string reads; see `scan`.

        A call to a builtin with arguments it doesn't take, and each field read on a
        builtin's result its type can't have, is one more error at `where`."""
        try:
            paths = expr_refs_of(text)
        except ExpressionError as exc:
            errors.append((f"{where}: {exc}", rng))
            return
        trees = [
            segment.tree for segment in scan_template(text) if isinstance(segment, Span)
        ]
        add_builtin_field_errors(trees, where, rng)
        if registry is not None and constructors:

            def ref_type(path: str) -> Type | None:
                """The proven type of one reference this site reads."""
                head, *fields = path.split(".")
                if head == "item":
                    contract = (
                        project_output_contract(item_contract, fields)
                        if item_contract is not None
                        else None
                    )
                else:
                    contract = reference_contract(path, site_producers, site_inputs)
                return output_contract_type(contract)

            for tree in trees:
                constructor_problems.extend(
                    (f"{where}: {problem}", rng)
                    for problem in constructor_errors(tree, registry, ref_type)
                )
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
        try:
            check_template_calls(text)
        except ExpressionError as exc:
            errors.append((f"{where}: {exc}", rng))

    def add_builtin_field_errors(
        trees: list, where: str, rng: Optional[SourceRange]
    ) -> None:
        """Add one error at `where` per field read on a builtin's result in `trees`."""
        for tree in trees:
            errors.extend(
                (f"{where}: {problem}", rng) for problem in builtin_field_errors(tree)
            )

    def check_template(
        where: str,
        template: str,
        declared: "set[str]",
        rng: Optional[SourceRange],
        *,
        noun: str,
        hint_tail: str,
        contracts: dict[str, LocalInput],
        partial_union_fields: bool,
    ) -> None:
        """Parse one rendered template; check each ref against `declared` and `contracts`.

        A malformed span, unknown builtin, builtin called with arguments it doesn't take, or
        nested `${...}` is one error at `where`, and so is each field read on a builtin's
        result its type can't have; each ref whose head is not in `declared` is
        another, with the pool-ref hint (ending in `hint_tail`) when the head is a pool-style
        head or a node id. Each distinct ref reading a field of a declared name with an
        entry in `contracts` is another when every alternative rejects the read
        (`partial_union_fields` allows a field only some union members have)."""
        try:
            refs = prompt_refs(template)
        except ExpressionError as exc:
            errors.append((f"{where}: {exc}", rng))
            return
        trees = [
            segment.tree for segment in scan_template(template) if isinstance(segment, Span)
        ]
        add_builtin_field_errors(trees, where, rng)
        field_checked: set[str] = set()
        for ref in refs:
            head = ref.split(".", 1)[0].strip()
            if head not in declared:
                hint = (
                    f" (a pool ref belongs in an input `from:`, not {hint_tail})"
                    if head in _PROMPT_FORBIDDEN_HEADS or head in valid_targets
                    else ""
                )
                errors.append((f"{where}: ${{{ref}}} is not a declared {noun}{hint}", rng))
                continue
            path = ref.strip()
            parts = path.split(".")
            local = contracts.get(head)
            if local is None or len(parts) < 2 or path in field_checked:
                continue
            field_checked.add(path)
            err = local_field_error(
                local.alternatives,
                parts[1:],
                path,
                source=local.source,
                partial_union_fields=partial_union_fields,
            )
            if err is not None:
                errors.append((f"{where}: {err}", rng))

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
        node_locals = local_contracts.get(node_id, {})
        if isinstance(node, AgentNode):
            declared = {p.name for p in (node.params or [])}
            declared.update(variable.name for variable in node.variables)
            prompt_contracts = dict(node_locals)
            prompt_contracts.update(
                (variable.name, LocalInput((variable.contract,)))
                for variable in node.variables
                if variable.contract is not None
            )
            for field_name, template in (
                ("prompt", node.prompt),
                ("system_prompt", node.system_prompt),
            ):
                if not template:
                    continue
                check_template(
                    result.site_name(node_id, field_name),
                    template,
                    declared,
                    franges.get(("field", field_name)) or node_range,
                    noun="input or variable",
                    hint_tail="a strict AGENT prompt",
                    contracts=prompt_contracts,
                    partial_union_fields=partial_union_fields,
                )
        elif isinstance(node, HumanInputNode):
            declared = {p.name for p in (node.params or [])}
            for path, text in template_sites(node.prompt, node.questions):
                if not text:
                    continue
                label = site_label(path)
                check_template(
                    f"node {node_id!r} {label}",
                    text,
                    declared,
                    franges.get(("text", label))
                    or franges.get(("field", path[0]))
                    or node_range,
                    noun="input",
                    hint_tail="human_input text",
                    contracts=node_locals,
                    partial_union_fields=partial_union_fields,
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
                constructors=False,
            )

        # A per-item binder (MAP) reads its iteration source above, then validates each
        # element input with the body-local `${item}` scope enabled.
        if node.binds_per_item:
            item_contract = _item_contract(
                nwiring.get("over"), node_producers, node_inputs, type_fn
            )
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
        case_reads = (
            {
                ref.split(".", 1)[0]
                for case in node.cases
                if case.when
                for ref in condition_refs(case.when)
            }
            if isinstance(node, CaseNode)
            else None
        )
        for param, src in nwiring.items():
            if param in reserved:
                continue
            if case_reads and param not in case_reads:
                # In a searched case whose `when:`s read something, a param they don't read
                # is a join input over a co-skipping escape's producers; each of its
                # references is already checked at the read that introduced it, so
                # scanning it would repeat that error. An `on:` case with no cases reads
                # nothing, so its `on:` value is still checked.
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
            where = f"{result.site_name(node_id, label)} from"
            source = binding_source(src)
            if isinstance(source, str):
                # A desugar may copy an authored binding to a synth node, so check the
                # authored site once.
                key = (where, source, param_rng)
                if key in scanned:
                    continue
                scanned.add(key)
            scan(
                src,
                where,
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

    for headline, problems in (
        ("flow has unresolved references", errors),
        ("flow has constructor calls that can never succeed", constructor_problems),
    ):
        if problems:
            first_range = next(
                (error_range for _, error_range in problems if error_range is not None),
                None,
            )
            raise diagnostics.error(
                f"{headline}:\n  " + "\n  ".join(msg for msg, _ in problems),
                range=first_range,
            )
