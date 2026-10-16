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

Both halves check an assert's field reads narrowed by the assert's own checks: `and`,
`or` and `not` short-circuit, so an `isinstance` or union tag comparison before a read
decides which union members it sees (`isinstance(v, Buy) and v.qty > 0`), and a check
that can never be true is a load error (`helpers.condition_guards`,
`helpers.narrowing_facts`).

Imports flow DOWN only within the engine core (`compile.validation`, `expr`,
`typesys`) and the `helpers` peers (`condition_guards`, `isinstance_types`,
`narrowing_facts`, `output_contracts`, `errors`), plus two acknowledged validate-phase
edges: `nodes` (Node/CaseNode discrimination) and
`desugar.inline_calls.desugar_call_directives` (inline-call rejection). Within the
validate phase it reads `LocalInput` from `typed_bindings`. Nothing in helpers/nodes/parse/desugar/edges imports this back.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Callable, Collection, Mapping, NamedTuple, Optional

from lark import Token, Tree

from yian.compile.validation import _classify_path
from yian.diagnostics import OriginRef, SourceRange
from yian.expr import ExpressionError, condition_refs, parse_condition
from yian.expr.expressions import rewrite_condition_refs
from yian.expr.template import flow_call_callees_in_spans
from yian.compose.desugar.inline_calls import desugar_call_directives
from yian.compose.helpers.condition_guards import Guard, guarded_refs
from yian.compose.helpers.isinstance_types import (
    collect_constructor_names,
    collect_isinstance_names,
)
from yian.compose.helpers.narrowing_facts import (
    NOT_NARROWABLE,
    BaseTypeOf,
    ImpossibleCheck,
    NotNarrowable,
    checked_refs,
    guarded_read_types,
    impossible_check_message,
    impossible_checks,
    narrowed_read_error,
    unresolved_type,
)
from yian.compose.helpers.output_contracts import (
    output_contract_path_error,
    project_output_contract,
)
from yian.compose.validate.typed_bindings import LocalInput
from yian.nodes.base import Node, NodeKind
from yian.nodes.contracts import (
    Assertion,
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
)
from yian.typesys.types import TypeRegistry
from yian.typesys.values import IntTooLongError, Type, ValueKind
from yian.compose.helpers.errors import LoadDiagnostics


# Reserved prefix for the END params synthesized from a lowered post-assert. Authors
# cannot write `__`-prefixed ids, so a synth name never collides with an author id.
_SYNTH_PREFIX = "__assert_"

# Kinds whose node asserts move onto the spawned child flow's END. MAP is absent because
# every MAP node assert is already rejected.
_SPAWNER_KINDS = frozenset({NodeKind.CALL, NodeKind.LOOP})


def _root_of(path: str, roots: Collection[str]) -> str | None:
    """The root `path` is at or below, else `None`."""
    for root in roots:
        if path == root or path.startswith(root + "."):
            return root
    return None


def _root_base_type(choice: Mapping[str, OutputContract]) -> BaseTypeOf:
    """The declared type of a base path below one of `choice`'s roots, under that
    root's chosen contract; other bases are not narrowed."""

    def base_type_of(base: str) -> Type | None | NotNarrowable:
        root = _root_of(base, choice)
        if root is None:
            return NOT_NARROWABLE
        rest = base[len(root) + 1 :].split(".") if base != root else []
        projected = project_output_contract(choice[root], rest)
        return projected.type if type(projected) is ValueContract else None

    return base_type_of


class _ChoiceView(NamedTuple):
    """What a condition proves under one choice of root contracts."""

    choice: dict[str, OutputContract]
    base_type_of: BaseTypeOf
    impossible: list[ImpossibleCheck]
    checked: list[tuple[str, Optional[Guard]]]


def _condition_errors(
    tree: Tree | Token,
    roots: Mapping[str, tuple[OutputContract, ...]],
    registry: TypeRegistry,
    *,
    partial_union_roots: Collection[str] = frozenset(),
) -> list[str]:
    """Return an assert's impossible checks and field-read errors, each message once.

    `roots` maps each path whose field reads are checked (a node-local name, `output`,
    `input.X`, `<id>.output`) to the contracts its value may have. `and`, `or` and `not`
    short-circuit, so the checks before a read (`isinstance` and union tag comparisons)
    narrow the values it can see (`guarded_read_types`): a read is fine when any narrowed
    type allows it, and otherwise the first one's error is reported, naming the narrowed
    type when it lacks the field. A read nothing narrows is walked on the root's contract.
    A read after a reported impossible check, or guarded by an `isinstance` whose type
    name doesn't resolve, is not checked.

    A root with several contracts (a node-local name's alternatives, which carry the
    branch that routes the node) is checked under each: a read is an error only when
    every choice rejects it, and a check is impossible only when it is impossible under
    every choice. Each distinct (read, guard) pair is checked, since one path can be read
    both narrowed and not.
    """
    refs = guarded_refs(tree)
    used = [root for root in roots if any(_root_of(ref, (root,)) for ref, _ in refs)]
    if not used:
        return []
    choices = [
        dict(zip(used, picked)) for picked in itertools.product(*(roots[r] for r in used))
    ]
    views: list[_ChoiceView] = []
    for choice in choices:
        base_type_of = _root_base_type(choice)
        views.append(
            _ChoiceView(
                choice,
                base_type_of,
                impossible_checks(tree, base_type_of, registry),
                checked_refs(tree, base_type_of, registry),
            )
        )
    messages = [
        impossible_check_message(check)
        for check in views[0].impossible
        if all(any(seen.text == check.text for seen in view.impossible) for view in views[1:])
    ]
    seen_reads: set[tuple[str, Guard]] = set()
    for index, (ref, guard) in enumerate(refs):
        root = _root_of(ref, used)
        if root is None or ref == root or (ref, guard) in seen_reads:
            continue
        seen_reads.add((ref, guard))
        if unresolved_type(guard, registry):
            # The type-table pass reports the unknown name.
            continue
        message = _read_error(
            ref,
            root,
            index,
            views,
            registry,
            partial_union_fields=root in partial_union_roots,
        )
        if message is not None:
            messages.append(message)
    return list(dict.fromkeys(messages))


def _read_error(
    ref: str,
    root: str,
    index: int,
    views: list[_ChoiceView],
    registry: TypeRegistry,
    *,
    partial_union_fields: bool,
) -> str | None:
    """Return why the `index`-th read `ref` below `root` can't work under any choice of
    root contracts, else `None`; a narrowed error is preferred over an unnarrowed one."""
    errors: list[tuple[bool, str]] = []
    for choice, base_type_of, _, checked in views:
        guard = checked[index][1]
        if guard is None:
            # A read after a reported check is reached by no value.
            return None
        candidates = guarded_read_types(ref, guard, base_type_of, registry)
        if not candidates:
            error = output_contract_path_error(
                choice[root],
                ref[len(root) + 1 :].split("."),
                ref,
                partial_union_fields=partial_union_fields,
                isinstance_hint=True,
            )
            if error is None:
                return None
            errors.append((False, error))
            continue
        first: str | None = None
        for candidate in candidates:
            typ, fields, _ = candidate
            error = output_contract_path_error(
                ValueContract(typ),
                list(fields),
                ref,
                partial_union_fields=partial_union_fields,
                isinstance_hint=True,
            )
            if error is None:
                return None
            first = first or narrowed_read_error(ref, candidate, error, registry)
        errors.append((True, first))
    return next((error for narrowed, error in errors if narrowed), errors[0][1])


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
    input_contracts: dict[str, ValueContract] | None = None,
    registry: TypeRegistry | None = None,
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

    A `${<id>.output.f}` read is field-checked against the producer's contract, and an
    `${input.X.f}` read against `input_contracts[X]` when the flow input has one. The
    assert's own checks narrow its reads: `and`, `or` and `not` short-circuit, so an
    `isinstance` or union tag comparison before a read (`isinstance(input.o, Buy) and
    input.o.qty > 0`, `not isinstance(input.o, Buy) or input.o.qty > 0`) decides which
    members it sees. A shorthand check base (`isinstance(o, Buy)`) is qualified like a
    read. A check that can never be true is a load error, and the `isinstance` types
    resolve in ``registry`` (the flow's typedefs).

    Raises `LoadError` on the first malformed assert, bad ref, impossible check, bad
    field read, or `${system}` ref.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    input_contracts = input_contracts or {}
    registry = registry or {}
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

    def _qualified_or_none(path: str) -> str | None:
        """The qualified path when `_qualify_flow_input` changes it, else `None`."""
        qualified = _qualify_flow_input(path)
        return qualified if qualified != path else None

    # The values whose field reads are checked, with the one contract each has.
    roots: dict[str, tuple[OutputContract, ...]] = {
        **{f"input.{name}": (contract,) for name, contract in input_contracts.items()},
        **{f"{node_id}.output": (contract,) for node_id, contract in producers.items()},
    }

    for i, assertion in enumerate(assert_list):
        expr = assertion.expression
        rng = ranges[i] if i < len(ranges) else None
        # 1. parse-check + collect refs on the ONE unified condition walk (any brace spelling).
        try:
            refs = [_qualify_flow_input(path) for path in condition_refs(expr)]
        except ExpressionError as exc:
            # A long int's text would repeat every digit; the range still points at it.
            long_int = isinstance(exc.__cause__, IntTooLongError)
            shown = "" if long_int else f" {assertion.display!r}"
            raise diagnostics.error(
                f"assert{shown} is not a valid boolean expression: {exc}",
                range=rng,
            ) from exc

        # 2. validate each ref + forbid `${system}` + detect any outputs-head ref (-> post).
        is_post = False
        for path in dict.fromkeys(refs):
            err = _classify_path(path, valid_targets, flow_inputs, ())
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

        # 3. impossible checks and field reads, narrowed by the assert's own checks.
        qualified = parse_condition(rewrite_condition_refs(expr, _qualified_or_none))
        errors = _condition_errors(qualified, roots, registry)
        if errors:
            raise diagnostics.error(
                f"assert {assertion.display!r}: {errors[0]}", range=rng
            )

        if not is_post:
            # 4. PRE (boundary): rewrite each `${input.X}` ref to `${output.X}`. START's boundary
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

        # 5. POST: turn each non-`output` ref into a synth END param, wired as an edge.
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
    local_contracts: dict[str, dict[str, LocalInput]] | None = None,
    partial_union_nodes: frozenset[str] = frozenset(),
    registry: TypeRegistry | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Validate + classify + stamp each node's `asserts:` — the per-node contract.

    A node assert is node-LOCAL: a bare `${name}` is one of the node's declared inputs;
    `${output}`/`${output.field}` is the node's own output. Exact contracts walk their
    semantic type; partial records enforce their closed root field set and walk any
    independently proven field contract.
    Rejected (located `LoadError`): a pool head (`${outputs/inputs/system/item.X}`), an
    inline `${call}`, `${output.field}` on an output that is neither a record nor a `file`
    (whose `uri`, `mime` and `name` it can read), a `${name.field}` read of a declared
    input when every alternative of ``local_contracts[nid][name]`` (a `LocalInput` from
    `local_input_contracts`) rejects it (a field only some union members have is readable
    on a node in ``partial_union_nodes``; a name with no entry is not field-checked), a
    declared input named `output`, `isinstance` or a constructor on a CALL/MAP/LOOP node
    (its asserts run in the child flow), and any node assert on a mapped call (a
    `MapNode`, `kind: map` — the per-node hook has no `${item}` scope). A call inside
    `${...}` to a name in ``type_names`` (the flow's `typedefs:` names) is a constructor,
    not an inline call.
    Classifies each PRE (no `${output}`) or POST (reads `${output}`) and stamps
    `node.pre_asserts` / `node.post_asserts`. An assert's own checks narrow its reads:
    `and`, `or` and `not` short-circuit, so an `isinstance` or union tag comparison
    before a read (`isinstance(v, Buy) and v.qty > 0`, `v.action != 'buy' or v.qty > 0`)
    decides which members it sees, with the `isinstance` types resolved in ``registry``
    (the flow's typedefs). A local name's alternatives carry the branch that routes the
    node, so a check is a load error when it can never be true under every alternative.
    ``assertion_site_at`` optionally supplies
    each carrier's opaque origin and validation range from the same authored occurrence,
    keyed by node id and original authored index. Runs in the shared `_assemble`, so a
    `defs:` child flow's internal nodes get this too. Accumulates all errors into one
    `LoadError`, located at the first available exact assertion, offending field, or node
    range.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    local_contracts = local_contracts or {}
    registry = registry or {}
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
        node_locals = local_contracts.get(nid, {})
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
            except ExpressionError:
                directive_calls, span_callees = ["(malformed call)"], []
            if directive_calls or span_callees:
                errors.append(
                    (
                        f"node {nid!r} assert {a!r}: an inline call is not allowed in a "
                        f"node-local assert (use a flow-level assert)",
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
                tree = parse_condition(a)
            except ExpressionError as exc:
                # A long int's text would repeat every digit; the range still points at it.
                shown = "" if isinstance(exc.__cause__, IntTooLongError) else f" {a!r}"
                errors.append(
                    (
                        f"node {nid!r} assert{shown}: not a valid boolean expression ({exc})",
                        error_range,
                    )
                )
                continue
            reads_output = False
            # The values whose field reads are checked, with the contracts each may have.
            roots: dict[str, tuple[OutputContract, ...]] = {}
            for ref in dict.fromkeys(refs):
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
                        # A file output reaches the shared walk, which types its
                        # `uri`, `mime` and `name` and rejects any other field.
                        if type(contract) is ValueContract and contract.type.kind not in (
                            ValueKind.OBJECT,
                            ValueKind.ANY,
                            ValueKind.FILE,
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
                            roots["output"] = (contract,)
                elif head in declared:
                    local = node_locals.get(head)
                    if local is not None and local.alternatives:
                        roots[head] = local.alternatives
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
            # A spawner assert with `isinstance` was rejected above, so its impossible
            # checks are never reported on top of that error.
            for message in _condition_errors(
                tree,
                roots,
                registry,
                partial_union_roots=(
                    frozenset(roots) - {"output"}
                    if nid in partial_union_nodes
                    else frozenset()
                ),
            ):
                errors.append((f"node {nid!r} assert: {message}", error_range))
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
