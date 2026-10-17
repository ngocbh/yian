"""kind=loop — the `('a -> 'a) -> 'a -> 'a` driver's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `loop` node's frozen
`LoopDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding the resolve-and-bake `build` (termination configuration +
the `'a -> 'a` field-name contract: a body returns a subset of the carried record, and each
omitted field carries over, so the loop's output contract merges the returned body fields with
the seeds of the omitted ones). `_finish` keeps concrete ranges on the Compose-owned
`BuildResult` and stamps only the descriptor's opaque origin onto the runtime node. Registered
into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here. It also holds the
loop-entry helpers the load checks use: `while_non_null_paths` finds the carried paths a
`while:` null check proves, `nullable_at` relaxes a type along such a path,
`loop_entry_inputs` gives the body's inputs read that way, and `carried_field_types` gives the
carried record's proven field types.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from lark import Token, Tree

from yian.diagnostics import SourceRange
from yian.expr import ExpressionError, expr_refs, parse_condition
from yian.nodes.contracts import (
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
    output_contract_field_error,
)
from yian.nodes.loop import LoopNode
from yian.typesys.types import TypeRegistry
from yian.typesys.values import Type, ValueKind
from yian.compose.helpers.condition_guards import AllOf, Guard, Holds, NotNullCheck, proves
from yian.compose.helpers.narrowing_facts import (
    BaseTypeOf,
    checked_refs,
    guarded_read_types,
    impossible_check_message,
    impossible_checks,
    narrowed_read_error,
    unresolved_type,
    wider_type,
)
from yian.compose.helpers.output_contracts import (
    combine_output_contracts,
    project_output_contract,
)
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor, reject_non_text
from yian.compose.nodes.resolver import (
    _callable_display,
    _resolve_child,
    child_signature,
)


@dataclass(frozen=True, kw_only=True)
class LoopDescriptor(NodeDescriptor):
    """kind=loop — a `('a -> 'a) -> 'a -> 'a` driver over a carried record.

    `call` names the body child flow (defs-first, else external — same resolution as
    `call`/`map`). The authored `input:` mapping seeds the carried record at turn 0. The body
    returns a subset of the carried record; each omitted field keeps its previous value.
    Exactly one of `while_`/`until_`/`times` carries the predicate; `max` optionally limits
    while/until iterations. `while`/`until` are Python keywords, so they land here as
    `while_`/`until_`.

    Inherits the common descriptor fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    Compose-only `range`/`field_ranges` parse metadata) and the template `parse` from
    `NodeDescriptor`. Its `_validate_fields` hook rejects a non-text `call:` and a non-string
    `while:`/`until:` and runs the base checks; all its deeper legality lives in
    `build`. Only the loop-specific fields are declared here; the normalized `inputs` field comes
    from the base, and a loop has no authored `output:` field.
    `child_label` is compiler-owned diagnostic metadata for an anonymous inline child. It is
    deliberately absent from `allowed`, never authorable, and excluded from descriptor equality.

    Class attrs: `kind="loop"`; `allowed` adds `call`, normalized `inputs`, `while`/`until`/`times`/`max`, and
    `asserts` to the commons; `required` is `{"call"}` (the body child flow);
    `binding_fields=("inputs",)` — a loop's carried-record seed bindings participate in
    inline-call and case-output desugaring.
    `keyword_remap` is loop-owned: it maps the surface Python keywords `while:`/`until:` onto the
    `while_`/`until_` descriptor fields so the inherited template `parse` reads them correctly.
    """

    kind = "loop"
    allowed = NodeDescriptor.allow(
        "call", "inputs", "while", "until", "times", "max", "asserts"
    )
    required = NodeDescriptor.require("call")
    binding_fields = ("inputs",)
    # Loop owns this remap (the base's is empty): `while`/`until` are Python keywords, so the
    # template `parse` reads the surface `while:`/`until:` into the `while_`/`until_` fields.
    keyword_remap = {"while": "while_", "until": "until_"}

    call: str
    child_label: str | None = field(default=None, compare=False)
    while_: Optional[str] = None
    until_: Optional[str] = None
    times: Optional[int] = None
    max: Optional[int] = None
    asserts: list[str] = field(default_factory=list)

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
        own_range: Optional[SourceRange],
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Reject a non-text `call:` or `while:`/`until:`, then run the base checks.

        A YAML bool, number, or list would otherwise crash the callable resolution or the
        predicate ref check, or, when falsy (`while: false`), skip it and fail only at runtime."""
        reject_non_text(
            cls.kind,
            node_id,
            "call",
            kwargs.get("call"),
            ranges.get(("field", "call")) or own_range,
            diagnostics,
            required=True,
        )
        for key, example in (("while", "while: not done"), ("until", "until: done")):
            value = kwargs.get(cls.keyword_remap[key])
            if value is not None and not isinstance(value, str):
                raise diagnostics.error(
                    f"loop node {node_id!r}: `{key}:` must be an expression string over the "
                    f"carried record, such as `{example}` (got {value!r})",
                    range=ranges.get(("field", key)) or own_range,
                )
        super()._validate_fields(node_id, kwargs, ranges, own_range, diagnostics)

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `LoopNode` (+ wiring) for this `LoopDescriptor` — a
        `('a -> 'a) -> 'a -> 'a` driver — resolve-and-bake, then retain Compose-only ranges
        via `_finish`.

        `loop` is a higher-order driver: the body child flow (`call:`) reads a subset of the
        carried record (`input:`) and returns a subset of it; each omitted field carries over
        from the previous iteration. Returns a `BuildResult` (the split): the node carries
        `params`; the flow owns the carried-record seed sources in `wiring`. Resolves the
        callable through `ctx.resolver` (required — a loop is higher-order), derives its
        `ChildSignature`, bakes the compiled child + parameters, and enforces the FIELD-NAME
        half of the contract plus termination configuration legality, including that a
        `while:`/`until:` loop without `max:` whose condition reads carried fields reads at
        least one field the body returns.

        A body that returns the whole carried record stamps its codomain as `output_contract`
        and leaves `returned_fields` unset. A body that returns a subset stamps
        `returned_fields` (carried order) and a merged `output_contract`: each returned field
        projected from the body codomain, each omitted field resolved from its seed binding:
        the seed's declared type when the binding is typed, else through
        `ctx.binding_contract_reader` (opaque when the reader is absent or the seed is
        unproven).
        The deep TYPE-level half compares every statically proven returned field with its
        statically proven carried seed in the loader pass `check_loop_type_contract`, which
        needs the assembled parent producers/flow-input contracts.
        `ctx.registry` reads an inline body's declared outputs and resolves the type names of
        `isinstance` checks in the predicate; the inline-CODE anchors on `ctx` are read only by
        the CODE build.

        Args:
            ctx (`BuildContext`):
                The immutable build inputs; `resolver` is required.

        Returns:
            `BuildResult`:
                The built `LoopNode` with its carried-record seed wiring.

        Raises:
            `LoadError`:
                The body, the field-name contract, or the termination configuration is
                illegal, or a `while:`/`until:` loop without `max:` can never end because its
                condition reads only fields the body never returns.
        """
        resolver = ctx.resolver
        child = _resolve_child(
            self.id,
            self.call,
            resolver,
            self.child_label,
            diagnostics=ctx.diagnostics,
            range=self.field_ranges.get(("field", "call")) or self.range,
        )
        sig = child_signature(child)
        child_display = _callable_display(self.call, self.child_label)
        # The child's asserts ride its compiled boundary nodes (lowered at the child's own
        # build); `clone_child` deep-copies the whole child subgraph per iteration, so those
        # lowered asserts survive the splice.
        node = LoopNode(
            self.id,
            flow_id=self.call,
            flow_version=None,
            child=child.compiled,
            child_inputs=sig.inputs,
            title=self.node_name,
        )
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        node.output_contract = sig.output

        errors: list[str] = []
        # Exactly one of `while:`/`until:`/`times:` selects the loop's
        # driver. `while`/`until` are PREDICATE loops (record-scoped boolean, optional `max:`
        # limit); `times: N` is a fixed COUNT (no predicate, `max:` redundant since the count
        # bounds it). Bake `predicate_kind`/`predicate`/`times`/`max_iters` per the present kind.
        present = [
            k
            for k, v in (
                ("while", self.while_),
                ("until", self.until_),
                ("times", self.times),
            )
            if v is not None
        ]
        if len(present) != 1:
            errors.append(
                f"loop node {self.id!r}: exactly one of while:/until:/times: is required "
                f"(got {sorted(present)})"
            )
        elif present[0] in ("while", "until"):
            kind = present[0]
            node.predicate_kind = kind
            node.predicate = self.while_ if kind == "while" else self.until_
            node.max_iters = self.max
            if self.max is not None:
                if not isinstance(self.max, int) or isinstance(self.max, bool):
                    # `max:` bounds the iteration COUNT, so it must be a plain integer. The
                    # descriptor is a bare dataclass (its `Optional[int]` annotation isn't
                    # enforced), so a YAML string/float passes through and would blow up the
                    # `< 1` compare or the runtime guard, and a YAML bool (an int subclass in
                    # Python) would silently read as 0/1. Reject any non-int type.
                    errors.append(
                        f"loop node {self.id!r}: `max:` must be an integer (got {self.max!r})"
                    )
                elif self.max < 1:
                    # A limit below 1 is nonsensical: `max:` bounds the iteration COUNT, so it
                    # must permit at least one body run. Reject at the boundary rather than
                    # letting `max: 0` run one iteration and then fail with a confusing
                    # "exceeded max (0)".
                    errors.append(
                        f"loop node {self.id!r}: `max:` must be >= 1 (got {self.max})"
                    )
        elif present[0] == "times":
            # `times: N` is a fixed count: exactly N runs, no predicate. `max_iters` doubles as
            # the count N so the runtime guard needs no `times`-special case. Same int/bool/`>= 1`
            # guard as `max:` — the count must be a plain integer permitting at least one run.
            node.predicate_kind = "times"
            node.predicate = None
            node.times = self.times
            node.max_iters = self.times
            if not isinstance(self.times, int) or isinstance(self.times, bool):
                errors.append(
                    f"loop node {self.id!r}: `times:` must be an integer (got {self.times!r})"
                )
            elif self.times < 1:
                errors.append(
                    f"loop node {self.id!r}: `times:` must be >= 1 (got {self.times})"
                )
            if self.max is not None:
                errors.append(
                    f"loop node {self.id!r}: max: is redundant with times: (times bounds the count)"
                )

        # The `'a -> 'a` FIELD-NAME contract (types are the loader pass). The carried record's
        # keys are the seed input names. The body returns a record: a record body
        # (`output: {...}`) returns its declared fields, and a value body (`output: ${x}`)
        # returns the fields of the record it references. Those field NAMES must be a subset
        # of the carried keys (an omitted field carries over); the body INPUT NAMES must be a
        # subset too (it reads a subset of the carried record). A body must return at least
        # one carried field, so a body returning no record or an empty record is rejected.
        carried = set(self.inputs or {})
        out_names = _contract_field_names(sig.output) or None
        if out_names is None:
            errors.append(
                f"loop node {self.id!r}: body {child_display} must return a record with at "
                f"least one carried field (a subset of the carried record {sorted(carried)}); "
                "write `output: {field: ...}` or a whole reference to a node with a record output"
            )
        elif out_names - carried:
            errors.append(
                f"loop node {self.id!r}: body output field(s) {sorted(out_names - carried)} "
                f"are not carried record fields {sorted(carried)} "
                "(a body returns a subset of the carried record)"
            )
        elif out_names != carried:
            node.returned_fields = tuple(n for n in self.inputs if n in out_names)
            node.output_contract = combine_output_contracts(
                {
                    name: (
                        self._returned_field_contract(sig.output, name)
                        if name in out_names
                        else self._seed_contract(ctx, name)
                    )
                    for name in self.inputs
                }
            )
        body_inputs = {d.name for d in sig.inputs}
        extra = body_inputs - carried
        if extra:
            errors.append(
                f"loop node {self.id!r}: body input(s) {sorted(extra)} are not carried record "
                f"fields {sorted(carried)} (the body reads a subset of the carried record)"
            )
        # `while:`/`until:` predicate scope: the predicate is evaluated against the carried record
        # (record-scoped `evaluate_when_record`), where the ONLY resolvable names are the carried
        # keys — an undeclared name or field silently resolves falsy, so `while:` would run zero
        # times and `until:` without a `max:` would spin forever. Reject a predicate whose ref
        # head isn't a carried field, or whose field read the carried record's type can't have,
        # at load, so a typo is loud not a runaway.
        # `times:` has no predicate, so skip it.
        predicate_src = self.while_ if self.while_ is not None else self.until_
        if predicate_src is not None:
            pred_kw = "while" if self.while_ is not None else "until"
            # The unified condition ref-walk: works for either spelling (bare `score >= 0.8`,
            # whole `${score >= 0.8}`), unlike a flat `${...}` regex
            # which read a whole-span predicate as one bogus head and a bare one as no heads.
            try:
                tree = parse_condition(predicate_src)
            except ExpressionError as exc:
                message = (
                    f"loop node {self.id!r}: `{pred_kw}:` is not a valid boolean expression: {exc}"
                )
                if not errors:
                    # Alone, the error belongs to the field; beside other errors it joins
                    # them at the node.
                    raise ctx.diagnostics.error(
                        message,
                        range=self.field_ranges.get(("field", pred_kw)) or self.range,
                    ) from exc
                errors.append(message)
                tree = None
            refs = expr_refs(tree) if tree is not None else []
            heads = {r.split(".", 1)[0] for r in refs}
            unknown = sorted(heads - carried)
            if unknown:
                errors.append(
                    f"loop node {self.id!r}: `{pred_kw}:` references unknown name(s) {unknown} — "
                    f"the predicate reads the carried record {sorted(carried)}"
                )
            if tree is not None and not errors and node.output_contract is not None:
                # With the field names valid, `output_contract` is the carried record's
                # contract (the body's for a total loop, merged with typed seeds otherwise).
                base_type_of = _carried_base_type(node.output_contract)
                if pred_kw == "while":
                    # `while:` is first evaluated on the seeds, which the type contract
                    # does not always force to fit the body (a literal, an untyped
                    # producer), so its checks must hold on a seed's type too.
                    seeds = {head: self._seed_contract(ctx, head) for head in sorted(heads)}
                    base_type_of = _seeded_base_type(base_type_of, seeds)
                for message in _predicate_errors(
                    node.output_contract, tree, base_type_of, ctx.registry
                ):
                    errors.append(f"loop node {self.id!r}: `{pred_kw}:` {message}")
            # A body that returns none of the fields the predicate reads can never change it,
            # so without `max:` the loop never ends. `returned_fields` is set only when the
            # body returns a known strict subset of the carried record. Skip when a driver,
            # parse, or unknown-name error already explains the loop.
            if (
                len(present) == 1
                and self.max is None
                and heads
                and not unknown
                and node.returned_fields is not None
                and heads.isdisjoint(node.returned_fields)
            ):
                stuck = sorted(heads)
                reads = ", ".join(f"`{n}`" for n in stuck)
                fix = " or ".join(f"`{n}`" for n in stuck)
                errors.append(
                    f"loop node {self.id!r}: this loop can never end: `{pred_kw}: "
                    f"{predicate_src}` reads {reads}, which the body never updates; "
                    f"return {fix} from the body, or add `max:`"
                )
        if errors:
            raise ctx.diagnostics.error("\n  ".join(errors), range=self.range)
        return self._finish(node, wiring)

    @staticmethod
    def _returned_field_contract(
        body_output: OutputContract | None,
        name: str,
    ) -> OutputContract | None:
        """One returned field's contract projected from the body codomain, else `None`."""
        if body_output is None:
            return None
        return project_output_contract(body_output, [name])

    def _seed_contract(self, ctx: BuildContext, name: str) -> OutputContract | None:
        """One omitted field's contract, resolved from its seed binding, else `None`.

        A typed seed (`name: !Type value`) carries its declared type, which the driver
        checks when it binds the seed; an untyped seed's contract is read from its value.

        Args:
            ctx (`BuildContext`):
                The immutable build inputs; supplies `binding_contract_reader`.
            name (`str`):
                The omitted carried field.

        Returns:
            `OutputContract | None`:
                The seed's declared or inferred contract, or `None` when unproven.
        """
        declared = self.input_types.get(name)
        if declared is not None:
            return declared
        if ctx.binding_contract_reader is None:
            return None
        return ctx.binding_contract_reader(
            f"{self.id}.input.{name}",
            self.inputs[name],
            self.field_ranges.get(("input", name))
            or self.field_ranges.get(("field", "inputs")),
        )


def _contract_field_names(
    contract: OutputContract | None,
) -> Optional[set[str]]:
    """The field names of a closed record contract, else `None`."""
    if type(contract) is PartialRecordContract:
        return set(contract.names)
    if contract is None or contract.type.fields is None:
        return None
    return set(contract.type.fields)


def _carried_base_type(contract: OutputContract) -> BaseTypeOf:
    """The declared type of a carried-record path, or `None` when it is unproven."""

    def base_type_of(base: str) -> Type | None:
        projected = project_output_contract(contract, base.split("."))
        return projected.type if type(projected) is ValueContract else None

    return base_type_of


def _seeded_base_type(
    carried: BaseTypeOf, seeds: dict[str, OutputContract | None]
) -> BaseTypeOf:
    """The type a carried-record path has on every evaluation of a `while:`, seed included.

    That is the carried type when the seed's type fits it, the seed's when the carried
    type fits the seed's, and otherwise, or when either is unproven, `None`.
    """

    def base_type_of(base: str) -> Type | None:
        carried_type = carried(base)
        head, *rest = base.split(".")
        seed = seeds.get(head)
        if not isinstance(carried_type, Type) or seed is None:
            return None
        projected = project_output_contract(seed, rest)
        if type(projected) is not ValueContract:
            return None
        return wider_type(carried_type, projected.type)

    return base_type_of


def _predicate_errors(
    contract: OutputContract,
    tree: Tree | Token,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
) -> list[str]:
    """Return the predicate's impossible checks and field-read errors, each message once.

    Checks and narrowed reads take each path's type from `base_type_of`; a read nothing
    narrows is walked on `contract`, the carried record. A read is checked under each
    distinct guard it is read with, since the same path can be read both narrowed and not.
    """
    messages = [
        impossible_check_message(check)
        for check in impossible_checks(tree, base_type_of, registry)
    ]
    for ref, guard in dict.fromkeys(checked_refs(tree, base_type_of, registry)):
        if guard is None:
            # A read after a reported check is reached by no value.
            continue
        message = _predicate_field_error(contract, ref, guard, base_type_of, registry)
        if message is not None:
            messages.append(message)
    return list(dict.fromkeys(messages))


def _predicate_field_error(
    contract: OutputContract,
    ref: str,
    guard: Guard,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
) -> str | None:
    """Return why a predicate ref reads a field the carried record can't have, else `None`.

    `and`, `or` and `not` short-circuit, so the checks before a read (`isinstance` and
    union tag comparisons) narrow the values it can see (`guarded_read_types`): the read
    is fine when any narrowed type allows it, and otherwise the first one's error is
    reported, naming the narrowed type when it lacks the field. With nothing narrowed the
    read is walked on the carried record. A read guarded by a check whose type name
    doesn't resolve is not checked, so the type-table check reports the unknown name.
    `guard` comes from `checked_refs`.
    """
    if unresolved_type(guard, registry):
        # The type-table pass reports the unknown name; an error here would fail the build
        # first and hide it behind a hint to add the check the author wrote.
        return None
    candidates = guarded_read_types(ref, guard, base_type_of, registry)
    if not candidates:
        return output_contract_field_error(contract, ref.split("."), ref, isinstance_hint=True)
    first: str | None = None
    for candidate in candidates:
        typ, fields, _ = candidate
        error = output_contract_field_error(ValueContract(typ), fields, ref, isinstance_hint=True)
        if error is None:
            return None
        if first is None:
            first = narrowed_read_error(ref, candidate, error, registry)
    return first


def while_non_null_paths(
    kind: str | None, predicate: str | None
) -> frozenset[tuple[str, ...]]:
    """Return the carried paths a loop's condition proves not null whenever the body starts.

    Only `while:` is checked before each iteration, the first included; `until:` is
    checked after the body and `times:` has no condition, so they prove nothing. A path
    comes from a `<path> != null` check the condition proves when true: each part of an
    `and` adds its paths, and an `or` keeps only the paths every part proves.

    Args:
        kind (`str | None`):
            The loop's driver: `"while"`, `"until"` or `"times"`.
        predicate (`str | None`):
            The condition text, bare or whole-span.

    Returns:
        `frozenset[tuple[str, ...]]`:
            Each proven path as field names from the carried record.

    Example:
        `while_non_null_paths("while", "n < 3 and page.next != null")` is
        `frozenset({("page", "next")})`.
    """
    if kind != "while" or predicate is None:
        return frozenset()
    try:
        tree = parse_condition(predicate)
    except ExpressionError:
        return frozenset()  # the build already reported the condition
    return _non_null_paths(proves(tree, True))


def _non_null_paths(guard: Guard) -> frozenset[tuple[str, ...]]:
    """Return the paths `guard` proves not null (see `while_non_null_paths`)."""
    if isinstance(guard, Holds):
        if isinstance(guard.check, NotNullCheck) and guard.value:
            return frozenset({tuple(guard.check.base.split("."))})
        return frozenset()
    parts = [_non_null_paths(part) for part in guard.parts]
    if isinstance(guard, AllOf):
        return frozenset().union(*parts)
    return frozenset.intersection(*parts) if parts else frozenset()


def nullable_at(typ: Type, path: tuple[str, ...]) -> Type:
    """Return `typ` with the value at `path`, and each record above it, made nullable.

    The path is followed through record fields. Where it names an absent field, or stops
    at a union or a non-record before its end, the rest of it is left as is, but the value
    it stopped at is still made nullable. Each field on the path stops being required in
    its record, since a field read through a null or absent record is null.

    Args:
        typ (`Type`):
            The type to change.
        path (`tuple[str, ...]`):
            Field names from `typ` to the value. `()` is `typ` itself.

    Returns:
        `Type`:
            The changed copy; `typ` itself is not changed.

    Example:
        `nullable_at({next: str}, ("next",))` with `next` required is
        `Optional({next: Optional(str)})` with `next` not required.
    """
    if path and typ.variants is None and typ.fields is not None and path[0] in typ.fields:
        inner = nullable_at(typ.fields[path[0]], path[1:])
        required = (typ.required or frozenset()) - {path[0]}
        typ = replace(typ, fields={**typ.fields, path[0]: inner}, required=required)
    if typ.kind is ValueKind.NONE:
        return typ  # the null type is already nullable
    return replace(typ, nullable=True)


def loop_entry_inputs(node: LoopNode) -> list[Parameter]:
    """Return a loop body's inputs as the values that enter the body may fill them.

    A path the loop's `while:` proves not null is never null when the body starts, so
    each input type is read as nullable at such a path and at each record above it. The
    seed check and the returned-field check compare against these, while the node's
    `child_inputs` stay as declared.

    Args:
        node (`LoopNode`):
            The built loop node.

    Returns:
        `list[Parameter]`:
            The body's inputs in order, each relaxed copy keeping its origins.
    """
    paths = while_non_null_paths(node.predicate_kind, node.predicate)
    inputs: list[Parameter] = []
    for parameter in node.child_inputs or ():
        contract = parameter.contract
        if contract is not None:
            typ = contract.type
            for path in sorted(paths):
                if path[0] == parameter.name:
                    typ = nullable_at(typ, path[1:])
            if typ is not contract.type:
                parameter = replace(parameter, contract=replace(contract, type=typ))
        inputs.append(parameter)
    return inputs


def carried_field_types(contract: OutputContract | None) -> dict[str, Type]:
    """Return the proven field types of a loop's carried-record contract.

    An exact record gives its fields' types and a partially typed record its proven
    field contracts' types; anything else proves no field.

    Args:
        contract (`OutputContract | None`):
            The loop node's `output_contract`: the body's own contract for a total loop,
            or the merged returned-plus-seeded contract for a partial one.

    Returns:
        `dict[str, Type]`:
            Each proven carried field's type, keyed by field name.
    """
    if type(contract) is ValueContract and contract.type.fields is not None:
        return dict(contract.type.fields)
    if type(contract) is PartialRecordContract:
        return {
            field_name: field_contract.type
            for field_name, field_contract in contract.fields
            if field_contract is not None
        }
    return {}
