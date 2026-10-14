"""`asserts:` compile-validate + lower — flow-level AND node-local.

Two halves of the `asserts:` contract, both in the validate phase:

FLOW-LEVEL (`LoweredAsserts` + `lower_flow_asserts`). A flow's `asserts:` is a list of
boolean checks in the `when:`/`asserts:` grammar (comparisons + `and`/`or`/`not`
+ arithmetic + `in`). This is the COMPILE-TIME half: it parses each assert,
validates every `${...}` reference it reads (reusing the leaf checker
`compile.validation._classify_path` — a dangling node/input or a bad record
field is a loud `LoadError`), and LOWERS each assert onto the synthesized
START/END boundary nodes so it fires node-locally through `eval_node`:

- a `${input.X}`-only assert -> **pre (boundary)**: depends only on the run
  arguments, so it rides START's ordinary `post_asserts` and fires against START's
  BOUND output record (fail-fast, before any body node runs), with each `${input.X}`
  rewritten to `${output.X}` (START's own bound output).
- an assert that reads ANY `${<id>.output[.X]}` -> **post-terminal**: it depends
  on a produced value, so it rides the synthesized `EndNode.post_asserts` and
  fires AFTER the run reaches a terminal, with each producer ref rewritten to a
  synthesized END input edge (so the dependency is STRUCTURAL and survives
  child-cloning/namespacing under `kind: call`).

`${system.X}` is forbidden in a flow assert (ambient — no producer to draw an edge
from). This module does NOT run the asserts — the boundary nodes fire them. Lowering
replaces only each carrier's executable expression; START/END receive the same
display/origin-bearing carriers.

An exact one-segment flow-input name is shorthand for `input.<name>` in this flow-level
scope. It therefore wins over a same-named node; node results remain explicit as
`node.output`. Dotted paths are not shorthand.

NODE-LOCAL (`validate_node_asserts` + `_POOL_HEADS`). A node assert is scoped to
one node: a bare `${name}` is a declared input, `${output}`/`${output.field}` is
the node's own output. It validates + classifies each PRE/POST and stamps
`node.pre_asserts` / `node.post_asserts`. The build conductor optionally supplies
the origin and validation range together from one retained item cursor, keyed by
node id and authored list index; only the origin crosses into the semantic assertion.

Imports flow DOWN only within the engine core (`compile.validation`, `expr`,
`typesys.values`) plus two acknowledged validate-phase edges: `nodes` (Node/
CaseNode discrimination) and `desugar.inline_calls.desugar_call_directives`
(inline-call rejection). Nothing in helpers/nodes/parse/desugar/edges imports
this back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from yian.compile.validation import _classify_path
from yian.diagnostics import OriginRef, SourceRange
from yian.expr import ExpressionError, condition_refs
from yian.expr.expressions import rewrite_condition_refs
from yian.expr.template import flow_call_callees_in_spans
from yian.compose.desugar.inline_calls import (
    SecretArgumentError,
    desugar_call_directives,
    holds_secret_keyword,
)
from yian.compose.helpers.isinstance_types import (
    collect_constructor_names,
    collect_isinstance_names,
)
from yian.compose.helpers.output_contracts import output_contract_path_error
from yian.nodes.base import Node, NodeKind
from yian.nodes.contracts import (
    Assertion,
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
)
from yian.typesys.values import ValueKind
from yian.compose.helpers.errors import LoadDiagnostics


# Reserved prefix for the END params synthesized from a lowered post-assert. Authors
# cannot write `__`-prefixed ids, so a synth name never collides with an author id.
_SYNTH_PREFIX = "__assert_"

# Kinds whose node asserts move onto the spawned child flow's END. MAP is absent because
# every MAP node assert is already rejected.
_SPAWNER_KINDS = frozenset({NodeKind.CALL, NodeKind.LOOP})


@dataclass(frozen=True)
class LoweredAsserts:
    """A flow's `asserts:` lowered into node-local asserts on the synthesized START/END.

    Flow-level assertions become node-local assertions on the synthesized boundary nodes,
    evaluated uniformly by the `Node.run` wrapper. That makes a cross-node post-assert's
    dependency structural — each
    `${producer.output}` ref becomes a real END input edge — so it survives child-cloning
    and namespacing under `kind: call` instead of silently failing.

    - `start_pre`: pre (boundary) asserts, `${input.X}` rewritten to `${output.X}` (they ride
      START's ordinary `post_asserts`, firing against START's BOUND output record).
    - `end_post`: post asserts, each producer/input ref rewritten to its synth-param ref.
    - `synth_params`: one untyped `Parameter(name=<synth>)` per synthesized END input.
    - `synth_wiring`: `dict[synth_name -> ${...} source string wired into END]`. A
      producer ref uses `${pid.output:-null}` (the `:-null` keeps the edge optional so a
      skipped producer does not co-skip END); an input ref uses `${input.X}` (always present).
    Each ordered `Assertion` owns both the lowered executable expression and stable
    author-facing display text. Rewrites preserve its opaque origin.
    """

    start_pre: list[Assertion] = field(default_factory=list)
    end_post: list[Assertion] = field(default_factory=list)
    synth_params: list[Parameter] = field(default_factory=list)
    synth_wiring: dict = field(default_factory=dict)


def lower_flow_asserts(
    assert_list: list[Assertion],
    flow_inputs: "set[str]",
    valid_targets: "set[str]",
    producers: "dict[str, OutputContract]",
    assert_ranges: "list[SourceRange | None] | None" = None,
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> LoweredAsserts:
    """Lower a flow's `asserts:` into node-local asserts for the START/END boundary nodes.

    Validation parses each assert + walks its `${...}` refs via `_classify_path` (a dangling
    ref / bad field is a loud LoadError). An exact unqualified name matching `flow_inputs`
    is first treated as `input.<name>` in this scope. The lowering then REWRITES each assert
    so it can fire node-locally, replacing only each assertion carrier's executable expression:

    - a PRE (boundary) assert rewrites each `${input.X}` ref to `${output.X}`, since START now
      fires these from its ordinary `post_asserts` against its own bound output record; and
    - a POST assert turns every non-`output` ref into a synthesized END param (`${producer
      .output}` -> `${__assert_<pid>}`, `${input.X}` -> `${__assert_input_<X>}`), with a
      matching `synth_wiring` edge, so the producer value is present in END's record when the
      assert fires. Multiple refs to the same producer / input SHARE one synth param.

    `${system.X}` is forbidden anywhere in an assert: it is ambient (no producer to draw an
    edge from), so a `system` ref raises a located `LoadError` telling the author to move the
    check into a body node.

    Raises `LoadError` on the first malformed assert, bad ref, or `${system}` ref.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    result = LoweredAsserts()
    # Retain exact and partial carriers so a closed partial root can reject an
    # unknown field without inventing a semantic Type for opaque siblings.
    for node_id, contract in producers.items():
        if type(contract) not in (ValueContract, PartialRecordContract):
            raise TypeError(
                "producer contracts must be ValueContract or PartialRecordContract values, "
                f"got {type(contract).__name__} for {node_id!r}"
            )
    # Positional flow-assert ranges, aligned 1:1 with `assert_list` (an entry is None where no span
    # exists). Indexed by the loop position below — never keyed by assert text — so two identical
    # flow asserts get distinct carets.
    ranges = assert_ranges or []

    # <producer pid or input X> -> allocated synth name, shared across all asserts so
    # repeated refs to the same source reuse the one END edge.
    synth_for_producer: dict[str, str] = {}
    synth_for_input: dict[str, str] = {}
    # Names already handed out, to guard the (near-impossible) collision case.
    allocated: set[str] = set()

    def _parseable(name: str) -> bool:
        # A synth name must be a legal ref path (defensive; the `__assert_` prefix + author id
        # components are always parseable in practice).
        try:
            condition_refs("${" + name + "}")
            return True
        except ExpressionError:
            return False

    def _alloc(base: str) -> str:
        """Allocate a collision-proof synth name from `base`, verified parseable as a ref."""
        name = base
        i = 1
        while (
            name in flow_inputs
            or name in valid_targets
            or name in allocated
            or not _parseable(name)
        ):
            name = f"{base}_{i}"
            i += 1
        allocated.add(name)
        return name

    def _qualify_flow_input(path: str) -> str:
        """Resolve one unqualified flow-input name into the explicit input namespace."""
        if "." not in path and path in flow_inputs:
            return f"input.{path}"
        return path

    for i, assertion in enumerate(assert_list):
        expr = assertion.expression
        rng = ranges[i] if i < len(ranges) else None
        # 1. parse-check + collect refs on the ONE unified condition walk (any brace spelling).
        try:
            refs = [_qualify_flow_input(path) for path in condition_refs(expr)]
        except ExpressionError as exc:
            raise diagnostics.error(
                f"assert {assertion.display!r} is not a valid boolean expression: {exc}",
                range=rng,
            ) from exc

        # 2. validate each ref + forbid `${system}` + detect any outputs-head ref (-> post).
        is_post = False
        for path in refs:
            err = _classify_path(path, valid_targets, flow_inputs, ())
            parts = path.split(".")
            if (
                err is None
                and len(parts) >= 2
                and parts[1] == "output"
                and parts[0] in producers
            ):
                err = output_contract_path_error(
                    producers[parts[0]], parts[2:], path
                )
            if err is not None:
                raise diagnostics.error(
                    f"assert {assertion.display!r}: {err}", range=rng
                )
            head = path.split(".")[0]
            if head == "system":
                raise diagnostics.error(
                    f"assert {assertion.display!r}: the `system` namespace is not allowed in asserts "
                    f"(it is ambient — there is no producer to draw an edge from; move the "
                    f"check into a body node)",
                    range=rng,
                )
            if head not in ("input", "system"):
                is_post = True

        if not is_post:
            # 3. PRE (boundary): rewrite each `${input.X}` ref to `${output.X}`. START's boundary
            # asserts now ride its ORDINARY `post_asserts`, fired by the wrapper/eval_node against
            # `{**inputs, "output": <START's bound record>}`. Reading `${output.X}` sees the
            # validated, defaulted value START binds. Unqualified flow-input names first
            # acquire the `input.` prefix; that prefix then maps to `output.`.
            def _pre_rename(path: str) -> "str | None":
                path = _qualify_flow_input(path)
                if path == "input":
                    return "output"  # defensive; a bare `${input}` head shouldn't appear
                if path.startswith("input."):
                    return "output." + path[len("input."):]
                return None

            lowered = rewrite_condition_refs(expr, _pre_rename)
            result.start_pre.append(replace(assertion, expression=lowered))
            continue

        # 4. POST: turn each non-`output` ref into a synth END param, wired as an edge.
        def _post_rename(path: str) -> "str | None":
            path = _qualify_flow_input(path)
            segs = path.split(".")
            head = segs[0]
            if head == "output":
                return None  # the node's own output — keep unchanged (defensive)
            if head == "input" and len(segs) >= 2:
                x = segs[1]
                synth = synth_for_input.get(x)
                if synth is None:
                    synth = _alloc(f"{_SYNTH_PREFIX}input_{x}")
                    synth_for_input[x] = synth
                    result.synth_params.append(Parameter(name=synth))
                    result.synth_wiring[synth] = f"${{input.{x}}}"
                return ".".join([synth, *segs[2:]])
            # producer-output ref: `pid.output[.f.g...]`, pid in valid_targets, seg[1] == output.
            if head in valid_targets and len(segs) >= 2 and segs[1] == "output":
                synth = synth_for_producer.get(head)
                if synth is None:
                    synth = _alloc(f"{_SYNTH_PREFIX}{head}")
                    synth_for_producer[head] = synth
                    result.synth_params.append(Parameter(name=synth))
                    # `:-null` makes the END edge optional: a conditionally-skipped
                    # producer binds None instead of co-skipping END to dead.
                    result.synth_wiring[synth] = f"${{{head}.output:-null}}"
                return ".".join([synth, *segs[2:]])
            return None  # anything else (e.g. `input` head w/o field) — keep

        lowered = rewrite_condition_refs(expr, _post_rename)
        result.end_post.append(replace(assertion, expression=lowered))

    return result


# Pool heads forbidden in a node-local assert (use bare `${name}` / `${output}` instead).
# Post-alias-delete — only the new singular heads remain.
_POOL_HEADS = frozenset({"input", "system", "item"})


def validate_node_asserts(
    nodes: dict[str, Node],
    descriptors: dict,
    *,
    assertion_site_at: Callable[
        [str, int],
        tuple[OriginRef | None, SourceRange | None] | None,
    ] | None = None,
    type_names: frozenset[str] = frozenset(),
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Validate + classify + stamp each node's `asserts:` — the per-node contract.

    A node assert is node-LOCAL: a bare `${name}` is one of the node's declared inputs;
    `${output}`/`${output.field}` is the node's own output. Exact contracts walk their
    semantic type; partial records enforce their closed root field set and walk any
    independently proven field contract.
    Rejected (located `LoadError`): a pool head (`${outputs/inputs/system/item.X}`), an inline
    `${call}`, `${output.field}` on a non-record output, a declared input named `output`,
    `isinstance` or a constructor on a CALL/MAP/LOOP node (its asserts run in the child
    flow), and any node assert on a mapped call (a `MapNode`, `kind: map` — the per-node
    hook has no `${item}` scope). A call inside `${...}` to a name in ``type_names`` (the
    flow's `typedefs:` names) is a constructor, not an inline call.
    Classifies each PRE (no `${output}`) or POST (reads `${output}`) and stamps
    `node.pre_asserts` / `node.post_asserts`. ``assertion_site_at`` optionally supplies each
    carrier's opaque origin and validation range from the same authored occurrence, keyed by
    node id and original authored index. Runs in the shared `_assemble`, so a `defs:`
    child flow's internal nodes get this too. Accumulates all errors into one `LoadError`, located
    at the first available exact assertion, offending field, or node range.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    errors: list[tuple[str, SourceRange | None]] = []
    for nid, node in nodes.items():
        desc = descriptors.get(nid)
        asserts = list(getattr(desc, "asserts", None) or [])
        aranges = list(getattr(desc, "assert_ranges", None) or [])
        if not asserts:
            continue
        descriptor_ranges = getattr(desc, "field_ranges", None)
        if not isinstance(descriptor_ranges, dict):
            descriptor_ranges = {}

        def field_range(kind: str, key: str) -> SourceRange | None:
            """Return one descriptor-owned field range."""
            return descriptor_ranges.get((kind, key))

        node_range = getattr(desc, "range", None)
        asserts_range = field_range("field", "asserts") or node_range
        if node.binds_per_item:
            errors.append(
                (
                    f"node {nid!r}: `asserts:` is not allowed on a mapped call "
                    f"(`kind: map`) — assert inside the def, or on a downstream node",
                    asserts_range,
                )
            )
            continue
        declared = {p.name for p in (node.params or [])}
        if "output" in declared:
            errors.append(
                (
                    f"node {nid!r}: a declared input named 'output' collides with the "
                    f"node-assert `${{output}}` keyword — rename the input",
                    field_range("input", "output") or asserts_range,
                )
            )
            continue
        pre: list[Assertion] = []
        post: list[Assertion] = []
        for i, a in enumerate(asserts):
            site = assertion_site_at(nid, i) if assertion_site_at is not None else None
            if site is None:
                origin = None
                rng = aranges[i] if i < len(aranges) else None
            else:
                origin, rng = site
            error_range = rng or asserts_range
            # node asserts are node-local — an inline call (a whole-value `call(...)` directive
            # or the retired `${flow(args)}` span) belongs in a flow-level assert, not here.
            try:
                _, directive_calls = desugar_call_directives(a, lambda: "__a")
                span_callees = flow_call_callees_in_spans(a, type_names)
            except SecretArgumentError as exc:
                errors.append((exc.message_for(f"node {nid!r} assert"), error_range))
                continue
            except ExpressionError:
                directive_calls, span_callees = ["(malformed call)"], []
            if directive_calls or span_callees:
                # A malformed call may hold a credential the parse never reached.
                shown = "" if holds_secret_keyword(a) else f" {a!r}"
                errors.append(
                    (
                        f"node {nid!r} assert{shown}: an inline call is not allowed in "
                        f"a node-local assert (use a flow-level assert)",
                        error_range,
                    )
                )
                continue
            # A spawner's asserts are lowered onto the child's END, which resolves
            # `isinstance` and constructor types in the child flow, not here.
            kind = getattr(node, "kind", None)
            if kind in _SPAWNER_KINDS:
                try:
                    used = (
                        "isinstance" if collect_isinstance_names(a, condition=True) else None
                    )
                    constructors = collect_constructor_names(a, condition=True)
                    if used is None and constructors:
                        used = f"{constructors[0]}(...)"
                except ExpressionError:
                    used = None
                if used is not None:
                    errors.append(
                        (
                            f"node {nid!r} assert {a!r}: {used} in a {kind.value} "
                            f"node's asserts is not supported; assert inside the child flow",
                            error_range,
                        )
                    )
                    continue
            # parse-check + collect refs on the ONE unified condition walk (any spelling).
            try:
                refs = condition_refs(a)
            except ExpressionError as exc:
                errors.append(
                    (
                        f"node {nid!r} assert {a!r}: not a valid boolean expression ({exc})",
                        error_range,
                    )
                )
                continue
            reads_output = False
            for ref in refs:
                segs = ref.split(".")
                head = segs[0] if segs else ""
                if head == "output":
                    reads_output = True
                    if len(segs) > 1:
                        contract = node.output_contract
                        if contract is not None and type(contract) not in (
                            ValueContract,
                            PartialRecordContract,
                        ):
                            raise TypeError(
                                f"node {nid!r} output_contract must be an exact "
                                "ValueContract, PartialRecordContract, or None, "
                                f"got {type(contract).__name__}"
                            )
                        if (
                            type(contract) is ValueContract
                            and contract.type.kind != ValueKind.OBJECT
                        ):
                            errors.append(
                                (
                                    f"node {nid!r} assert: ${{{ref}}} — the node output is "
                                    f"not a record, so it has no field "
                                    f"{'.'.join(segs[1:])!r}",
                                    error_range,
                                )
                            )
                        elif contract is not None:
                            err = output_contract_path_error(contract, segs[1:], ref)
                            if err is not None:
                                errors.append(
                                    (
                                        f"node {nid!r} assert: {err}",
                                        error_range,
                                    )
                                )
                elif head in declared:
                    continue
                elif head in _POOL_HEADS:
                    errors.append(
                        (
                            f"node {nid!r} assert: ${{{ref}}} — node asserts are node-local; "
                            f"use `${{name}}` (a declared input) or `${{output}}`",
                            error_range,
                        )
                    )
                else:
                    errors.append(
                        (
                            f"node {nid!r} assert: ${{{ref}}} is not a declared input or "
                            f"`${{output}}`",
                            error_range,
                        )
                    )
            if reads_output:
                post.append(Assertion(expression=a, display=a, origin=origin))
            else:
                pre.append(Assertion(expression=a, display=a, origin=origin))
        node.pre_asserts = pre
        node.post_asserts = post
    if errors:
        first_range = next(
            (error_range for _, error_range in errors if error_range is not None),
            None,
        )
        raise diagnostics.error(
            "flow has invalid node asserts:\n  "
            + "\n  ".join(msg for msg, _ in errors),
            range=first_range,
        )
