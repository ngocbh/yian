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
from agent_composer.expr.expressions import rewrite_condition_refs
from agent_composer.expr.template import flow_call_callees_in_spans
from agent_composer.compose.desugar.inline_calls import desugar_call_directives
from agent_composer.nodes.base import Node, ParamDecl
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


# Reserved prefix for the END params synthesized from a lowered post-assert. Authors
# cannot write `__`-prefixed ids, so a synth name never collides with an author id.
_SYNTH_PREFIX = "__assert_"


@dataclass(frozen=True)
class LoweredAsserts:
    """A flow's `asserts:` lowered into node-local asserts on the synthesized START/END.

    The refactor replaces flow-level assert enforcement (which resolves refs against the
    pool) with node-local asserts stamped on the boundary nodes and fired uniformly by
    `eval_node`. That makes a cross-node post-assert's dependency STRUCTURAL — each
    `${producer.output}` ref becomes a real END input edge — so it survives child-cloning
    and namespacing under `kind: call` instead of silently failing.

    - `start_pre`: pre (boundary) asserts, `${input.X}`/`input.X` rewritten to bare
      `${X}`/`X` (START's params ARE the input names, resolved record-scoped).
    - `end_post`: post asserts, each producer/input ref rewritten to its synth-param ref.
    - `synth_params`: one untyped `ParamDecl(name=<synth>)` per synthesized END input.
    - `synth_wiring`: `dict[synth_name -> ${...} source string wired into END]`. A
      producer ref uses `${pid.output:-null}` (the `:-null` keeps the edge optional so a
      skipped producer does not co-skip END); an input ref uses `${input.X}` (always present).
    - `pre_ranges` / `post_ranges`: `dict[lowered-string -> SourceRange]`, render-only for
      error carets; empty when `assert_ranges` was not supplied.
    - `pre_texts` / `post_texts`: `dict[lowered-string -> original author text]`. The node fires
      the LOWERED form, but a failure message must show the author's ORIGINAL text (`calc.output
      > 100`, not the synth form `${__assert_calc} > 100`) — eval_node / StartNode look the display
      text up here, falling back to the lowered string when a mapping is absent.
    """

    start_pre: list[str] = field(default_factory=list)
    end_post: list[str] = field(default_factory=list)
    synth_params: list[ParamDecl] = field(default_factory=list)
    synth_wiring: dict = field(default_factory=dict)
    pre_ranges: dict = field(default_factory=dict)
    post_ranges: dict = field(default_factory=dict)
    pre_texts: dict = field(default_factory=dict)
    post_texts: dict = field(default_factory=dict)


def lower_flow_asserts(
    assert_list: list[str],
    flow_inputs: "set[str]",
    valid_targets: "set[str]",
    producers: "dict[str, Type]",
    assert_ranges: "dict | None" = None,
) -> LoweredAsserts:
    """Lower a flow's `asserts:` into node-local asserts for the START/END boundary nodes.

    Validation mirrors `classify_asserts` exactly (same parse + `_classify_path` loop, same
    LoadError shapes); the difference is that instead of returning the raw assert strings
    split into two lists, each assert is REWRITTEN so it can fire node-locally:

    - a PRE (boundary) assert drops the `input.` prefix from every ref, since START's own
      params carry the input names; and
    - a POST assert turns every non-`output` ref into a synthesized END param (`${producer
      .output}` -> `${__assert_<pid>}`, `${input.X}` -> `${__assert_input_<X>}`), with a
      matching `synth_wiring` edge, so the producer value is present in END's record when the
      assert fires. Multiple refs to the same producer / input SHARE one synth param.

    `${system.X}` is forbidden anywhere in an assert: it is ambient (no producer to draw an
    edge from), so a `system` ref raises a located `LoadError` telling the author to move the
    check into a body node.

    Raises `LoadError` on the first malformed assert, bad ref, or `${system}` ref.
    """
    result = LoweredAsserts()
    # Only resolvable producers participate in the dotted walk (mirrors classify_asserts).
    producer_types = {nid: sh for nid, sh in producers.items() if sh is not None}
    ranges = assert_ranges or {}

    # <producer pid or input X> -> allocated synth name, shared across all asserts so
    # repeated refs to the same source reuse the one END edge.
    synth_for_producer: dict[str, str] = {}
    synth_for_input: dict[str, str] = {}
    # Names already handed out, to guard the (near-impossible) collision case.
    allocated: set[str] = set()

    def _alloc(base: str) -> str:
        """Allocate a collision-proof synth name from `base`, verified parseable as a ref."""
        name = base
        i = 1
        while (
            name in flow_inputs
            or name in valid_targets
            or name in allocated
        ):
            name = f"{base}_{i}"
            i += 1
        # A synth name must be a legal ref path — verify empirically (defensive; the
        # `__assert_` prefix + author id components are always parseable in practice).
        try:
            condition_refs("${" + name + "}")
        except ExpressionError:
            name = f"{base}_{i}"
        allocated.add(name)
        return name

    for expr in assert_list:
        # 1. parse-check + collect refs on the ONE unified condition walk (any brace spelling).
        try:
            refs = condition_refs(expr)
        except ExpressionError as exc:
            raise LoadError(
                f"assert {expr!r} is not a valid boolean expression: {exc}"
            ) from exc

        rng = ranges.get(expr)

        # 2. validate each ref + forbid `${system}` + detect any outputs-head ref (-> post).
        is_post = False
        for path in refs:
            err = _classify_path(path, valid_targets, flow_inputs, (), producer_types)
            if err is not None:
                raise LoadError(f"assert {expr!r}: {err}", range=rng)
            head = path.split(".")[0]
            if head == "system":
                raise LoadError(
                    f"assert {expr!r}: the `system` namespace is not allowed in asserts "
                    f"(it is ambient — there is no producer to draw an edge from; move the "
                    f"check into a body node)",
                    range=rng,
                )
            if head not in ("input", "system"):
                is_post = True

        if not is_post:
            # 3. PRE (boundary): drop the `input.` prefix so the ref names a START param.
            def _pre_rename(path: str) -> "str | None":
                if path == "input":
                    return "input"  # defensive; a bare `${input}` head shouldn't appear
                if path.startswith("input."):
                    return path[len("input."):]
                return None

            lowered = rewrite_condition_refs(expr, _pre_rename)
            result.start_pre.append(lowered)
            result.pre_texts[lowered] = expr
            if rng is not None:
                result.pre_ranges[lowered] = rng
            continue

        # 4. POST: turn each non-`output` ref into a synth END param, wired as an edge.
        def _post_rename(path: str) -> "str | None":
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
                    result.synth_params.append(ParamDecl(name=synth))
                    result.synth_wiring[synth] = f"${{input.{x}}}"
                return ".".join([synth, *segs[2:]])
            # producer-output ref: `pid.output[.f.g...]`, pid in valid_targets, seg[1] == output.
            if head in valid_targets and len(segs) >= 2 and segs[1] == "output":
                synth = synth_for_producer.get(head)
                if synth is None:
                    synth = _alloc(f"{_SYNTH_PREFIX}{head}")
                    synth_for_producer[head] = synth
                    result.synth_params.append(ParamDecl(name=synth))
                    # `:-null` makes the END edge optional: a conditionally-skipped
                    # producer binds None instead of co-skipping END to dead.
                    result.synth_wiring[synth] = f"${{{head}.output:-null}}"
                return ".".join([synth, *segs[2:]])
            return None  # anything else (e.g. `input` head w/o field) — keep

        lowered = rewrite_condition_refs(expr, _post_rename)
        result.end_post.append(lowered)
        result.post_texts[lowered] = expr
        if rng is not None:
            result.post_ranges[lowered] = rng

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
