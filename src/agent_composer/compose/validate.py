"""Graph- and reference-level validation for the loader.

Reuses the representation-neutral leaf checkers from `compile.validation` —
`_classify_path` (head-resolution + dotted-field dispatch) — on the synthesized graph +
built nodes, re-raising as a `LoadError` so the surface speaks one error type. The
handle-alignment check ports `_check_case_handles`'s rule to the desugared `CaseNode`s +
their control edges (the Compose path has no `FlowSpec`). Directed-cycle rejection lives
in the peer `stages.cycles` pass.

Reference-wiring (`validate_references`) runs the leaf checkers over every
binding site of a built flow — each node input source (from `flow.wiring`), the flow
`outputs:` bindings, and (via the desugared case's `__rN`/`__on` wiring sources)
each `case`'s `on:`/`when:` data refs — accumulating ALL located errors. Two subtleties
mirror the legacy `_collect_reference_errors`:

- **Prompt scope**: an AGENT prompt may interpolate ONLY this node's declared
  inputs as bare `${name}`; a `${<id>.output}`/`${input.X}`/`${system.X}` in a prompt is a
  located error.
- **Case node-local refs**: the desugared CASE `when:` strings use `${__rN}`/`${__on}`,
  which resolve against the bound input record — they are EXCLUDED from `_classify_path`
  (only the `__rN`/`__on` SOURCES are classified, against the pool; mirrors the strict
  CASE declared-input check).

Imports flow down only: `compile.model` (Edge), `compile.validation` (the leaf cycle +
ref checkers), `expr` (the binding parser), `nodes` (Node/CaseNode/AgentNode discrimination).
Nothing imports this back.
"""

from __future__ import annotations

from typing import Any

from agent_composer.compile.model import Edge
from agent_composer.compile.validation import (
    _classify_path,
    _walk_record_fields,
)
from agent_composer.expr import (
    ExpressionError,
    condition_refs,
    expr_refs_of,
    prompt_refs,
)
from agent_composer.expr.template import flow_call_callees_in_spans
from agent_composer.compose.stages.inline_calls import desugar_call_directives
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Node
from agent_composer.nodes.case import DEFAULT_HANDLE, CaseNode
from agent_composer.nodes.wait import WaitNode
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.errors import LoadError

# Heads forbidden in a strict prompt (always trigger the bespoke hint).
# `input`/`system`/`item` are the pool-style heads; a node-id head (e.g. `${other.output}`)
# is detected separately by `head in valid_targets`.
_PROMPT_FORBIDDEN_HEADS = ("input", "system", "item")


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


# --------------------------------------------------------------------------- #
# Reference-wiring validation: dangling head + dotted-field + prompt-scope
#
# The analogue of `compile.validation._collect_reference_errors`, walking the BUILT
# flow instead of a `FlowSpec`: it CALLS the leaf checkers (`_classify_path`) directly
# (supplying the `valid_targets`/`flow_inputs` sets + the `producers` Type-map), only
# reimplementing the per-node walk glue. Every located problem accumulates; one
# `LoadError` is raised carrying them all.
# --------------------------------------------------------------------------- #


def _output_bindings(outputs: Any) -> list[Any]:
    """The flow `outputs:` section as a flat list of binding values (sites to check).

    A whole-string binding (`${a.output | b.output}`) is one value; a
    name -> binding map is each mapped value. Mirrors
    `build._output_bindings` (kept local — `validate` does not import `build`).
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

    `node_lines` (node id -> source line) + `outputs_line` (the `outputs:` section line)
    locate each error: a node-binding/prompt error at the node's line; a flow-output ref
    at the `outputs:` line. The aggregate `LoadError` carries the FIRST
    located error's line (best-effort, since several sites may be wrong at once).

    Accumulates ALL located errors and raises a single `LoadError`; passes silently
    on a clean flow. Reuses `_classify_path` (head-resolution + dotted walk)
    verbatim — the per-node walk glue is the only loader-specific code.
    """
    lines = node_lines or {}
    valid_targets = set(nodes)
    # Only resolvable producers participate in the dotted walk (opaque -> lenient).
    producer_types = {nid: sh for nid, sh in producers.items() if sh is not None}
    # (message, line) pairs — line locates the offending `.yaml` line where known.
    errors: list[tuple[str, "int | None"]] = []

    def scan(from_value: Any, where: str, line: "int | None", extra_heads: tuple = ()) -> None:
        """Name-check every reference a BINDING `from:` reads (coalesce + nested-default
        split by `expr_refs_of`); a non-string source / literal yields no refs.

        `extra_heads` is the per-site set of body-local scopes treated as lenient — e.g.
        a MAP per-element input's `${item}` (valid only inside `map.inputs`, not `over:`)."""
        if not isinstance(from_value, str):
            return
        try:
            paths = expr_refs_of(from_value)
        except ExpressionError as exc:
            errors.append((f"{where}: {exc}", line))
            return
        for path in paths:
            err = _classify_path(
                path, valid_targets, flow_inputs, extra_heads, producer_types
            )
            if err is not None:
                errors.append((f"{where}: {err}", line))

    for node_id, node in nodes.items():
        node_line = lines.get(node_id)
        # Prompt scope: a strict AGENT prompt may interpolate ONLY this node's
        # declared inputs as bare `${name}` (or as a `${name}`-wrapped builtin-call arg);
        # a pool ref in a prompt is loud, and an unknown builtin / malformed span is
        # rejected here (via `prompt_refs`, the renderer's compile-time companion).
        if isinstance(node, AgentNode) and node.prompt:
            declared = {p.name for p in (node.params or [])}
            try:
                refs = prompt_refs(node.prompt)
            except ExpressionError as exc:
                errors.append((f"node {node_id!r} prompt: {exc}", node_line))
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
                    ))
        # A per-item binder (MAP) reads a parent-pool `list[T]` from its reserved wiring key(s)
        # (`over` — no `${item}` there); each per-element input MAY use `${item}` (the body-local
        # element scope, lenient).
        if node.binds_per_item:
            nwiring = flow_wiring.get(node_id, {})
            reserved = node.reserved_wiring_keys()
            for key in reserved:
                scan(nwiring.get(key), f"node {node_id!r} map {key}", node_line)
            for param, src in nwiring.items():
                if param in reserved:
                    continue  # the iteration source, scanned above (not a per-element input)
                scan(
                    src,
                    f"node {node_id!r} map input {param!r} from",
                    node_line,
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
        for param, src in flow_wiring.get(node_id, {}).items():
            scan(src, f"node {node_id!r} input {param!r} from", node_line)

    # Flow outputs (the codomain): each binding `from:` checked like a node input —
    # located at the `outputs:` section line (dangling-head/dotted-field errors point here).
    for value in _output_bindings(outputs):
        scan(value, "flow output from", outputs_line)

    if errors:
        first_line = next((ln for _, ln in errors if ln is not None), None)
        raise LoadError(
            "flow has unresolved references:\n  "
            + "\n  ".join(msg for msg, _ in errors),
            line=first_line,
        )


def validate_human_questions(
    nodes: dict[str, Node],
    node_lines: "dict[str, int] | None" = None,
) -> None:
    """Validate each built human_input gate's questions surface at load time.

    - A LITERAL questions list -> parse_questions (rejects bad count/shape/dup headers),
      re-raised as a located LoadError.
    - A ref-form questions_input -> the name MUST be one of the node's declared inputs
      (mirrors the AGENT prompt-scope rule); otherwise LoadError with a hint.
    Legacy (no questions) gates pass through. Accumulates all errors into one LoadError.
    """
    from agent_composer.nodes.human_input.node import HumanInputNode
    from agent_composer.nodes.human_input.questions import (
        QuestionSpecError,
        parse_questions,
    )

    lines = node_lines or {}
    # (message, line) pairs — line locates the offending `.yaml` line where known.
    errors: list[tuple[str, "int | None"]] = []
    for node_id, node in nodes.items():
        if not isinstance(node, HumanInputNode):
            continue
        line = lines.get(node_id)
        if node.questions is not None:
            # A literal list baked into the node: structurally check it at LOAD,
            # so a bad count/shape/duplicate header fails here rather than at run.
            try:
                parse_questions(node.questions)
            except QuestionSpecError as exc:
                errors.append((f"node {node_id!r} (kind=human_input): {exc}", line))
        elif node.questions_input is not None:
            # A ref-form gate reads its questions from a declared input — the name
            # must be wired in via `input:` (mirrors the AGENT prompt-scope rule).
            declared = {p.name for p in (node.params or [])}
            if node.questions_input not in declared:
                errors.append((
                    f"node {node_id!r} (kind=human_input): questions reference "
                    f"${{{node.questions_input}}} is not a declared input — feed dynamic "
                    f"questions through `input:` wiring and reference the declared name "
                    f"(questions: ${{name}}), or use adaptive_questions:",
                    line,
                ))
    if errors:
        first_line = next((ln for _, ln in errors if ln is not None), None)
        raise LoadError(
            "flow has invalid human_input questions:\n  "
            + "\n  ".join(msg for msg, _ in errors),
            line=first_line,
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
