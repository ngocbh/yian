"""`asserts:` compile-validate + classify — flow-level AND node-local.

Two halves of the `asserts:` contract, both in the validate phase:

FLOW-LEVEL (`AssertSet` + `classify_asserts`). A flow's `asserts:` is a list of
boolean checks in the `when:`/`asserts:` grammar (comparisons + `and`/`or`/`not`
+ arithmetic + `in`). This is the COMPILE-TIME half: it parses each assert,
validates every `${...}` reference it reads (reusing the leaf checker
`compile.validation._classify_path` — a dangling node/input or a bad record
field is a loud `LoadError`), and CLASSIFIES each assert by which side it
constrains:

- a `${input.X}` / `${system.X}`-only assert -> **boundary**: depends only on the
  run arguments + ambients, so it can fire BEFORE any node runs (fail-fast).
- an assert that reads ANY `${<id>.output[.X]}` -> **post-terminal**: it depends
  on a produced value, so it can only fire AFTER the run reaches a terminal.

This module does NOT run the asserts — `run_flow` evaluates each split against the
appropriate pool via `expr.evaluate_when`. Here we just parse, validate, and split.

NODE-LOCAL (`validate_node_asserts` + `_POOL_HEADS`). A node assert is scoped to
one node: a bare `${name}` is a declared input, `${output}`/`${output.field}` is
the node's own output. It validates + classifies each PRE/POST and stamps
`node.pre_asserts` / `node.post_asserts`.

Imports flow DOWN only within the engine core (`compile.validation`, `expr`,
`typesys.values`) plus two acknowledged validate-phase edges: `nodes` (Node/
CaseNode discrimination) and `desugar.inline_calls.desugar_call_directives`
(inline-call rejection). Nothing in helpers/nodes/parse/desugar/edges imports
this back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_composer.compile.validation import _classify_path, _walk_record_fields
from agent_composer.expr import ExpressionError, condition_refs
from agent_composer.expr.template import flow_call_callees_in_spans
from agent_composer.compose.desugar.inline_calls import desugar_call_directives
from agent_composer.nodes.base import Node
from agent_composer.typesys.values import Type, ValueKind
from agent_composer.compose.helpers.errors import LoadError


@dataclass(frozen=True)
class AssertSet:
    """The compile-time split of a flow's `asserts:`, for `run_flow` to enforce in two phases.

    `boundary` fires pre-run against the `${input}`/`${system}` pool; `post` fires after the
    run reaches a terminal, against the full pool (it reads `${<id>.output[.X]}`). Each list
    holds the raw assert strings (run via `expr.evaluate_when`), order-preserving.
    """

    boundary: list[str] = field(default_factory=list)
    post: list[str] = field(default_factory=list)


def classify_asserts(
    assert_list: list[str],
    flow_inputs: "set[str]",
    valid_targets: "set[str]",
    producers: "dict[str, Type]",
) -> AssertSet:
    """Parse, validate, and split a flow's `asserts:` into boundary vs post-terminal.

    `flow_inputs` is the declared flow-input names (the `${input.X}` set); `valid_targets`
    is the node ids (the `${<id>.output}` set); `producers` maps node id -> its
    `output_type` (drives the dotted-field walk, e.g. `${synth.output.confidence}`
    against the `View` record). Each assert is parse-checked in the boolean grammar (a
    malformed expression is loud) and each `${...}` ref is name-checked via
    `_classify_path` (a dangling ref / bad field is loud). An assert is
    **post-terminal** iff ANY of its refs has a head OTHER than `input`/`system`;
    otherwise **boundary**.

    Raises `LoadError` on the first malformed assert or bad ref.
    """
    result = AssertSet()
    # Only resolvable producers participate in the dotted walk (an opaque/None producer
    # stays lenient) — mirrors validate.validate_references.
    producer_types = {nid: sh for nid, sh in producers.items() if sh is not None}

    for expr in assert_list:
        # 1. parse-check + collect refs on the ONE unified condition walk. `condition_refs`
        # parses the WHOLE value as an expression (all three brace spellings) and returns its
        # reference leaves — so a whole-span `${a <= 10}` and a bare `a <= 10` extract the same
        # refs a mixed `${a} <= 10` does (a flat `${...}` regex could not).
        try:
            refs = condition_refs(expr)
        except ExpressionError as exc:
            raise LoadError(
                f"assert {expr!r} is not a valid boolean expression: {exc}"
            ) from exc

        # 2. validate each ref + 3. detect any outputs-head ref (-> post-terminal).
        is_post = False
        for path in refs:
            err = _classify_path(path, valid_targets, flow_inputs, (), producer_types)
            if err is not None:
                raise LoadError(f"assert {expr!r}: {err}")
            # post-terminal iff a ref has a head other than input/system.
            # The `item` head is body-local (not valid at flow-level), so a stray
            # `${item.X}` falls into "head not in input/system" → POST and raises later
            # at resolve time. Legacy `inputs`/`outputs` heads are already rejected by
            # _classify_path with the typo hint so they never reach here.
            head = path.split(".")[0]
            if head not in ("input", "system"):
                is_post = True

        (result.post if is_post else result.boundary).append(expr)

    return result


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
