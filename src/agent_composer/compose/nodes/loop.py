"""kind=loop — the `('a -> 'a) -> 'a -> 'a` driver's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `loop` node's frozen
`LoopDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding the resolve-and-bake `build` (predicate-slice legality +
the `'a -> 'a` field-name contract). Registered into `KINDS`/`BINDING_FIELDS` at import so the
shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.expr import ExpressionError, condition_refs
from agent_composer.nodes.loop import LoopNode
from agent_composer.typesys.values import Type
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.compose.nodes.resolver import (
    _callable_display,
    _resolve_child,
    child_signature,
)


@dataclass(frozen=True, kw_only=True)
class LoopDescriptor(NodeDescriptor):
    """kind=loop — a `('a -> 'a) -> 'a -> 'a` driver over a carried record.

    `call` names the body subflow (defs-first, else external — same resolution as
    `call`/`map`). `inputs` seeds the carried record at turn 0. Exactly one of
    `while_`/`until_`/`times` carries the predicate; `max` is the runaway guard (required for
    while/until). `while`/`until` are Python keywords, so they land here as `while_`/`until_`.

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) and the template `parse` from `NodeDescriptor` (no
    custom parse hook — loop's field legality is the default allowed/required key check + the
    asserts-list check; all its deeper legality lives in `build`). Only the loop-specific fields
    are declared here; `inputs` comes from the base, and a loop never populates `outputs`.
    `child_label` is compiler-owned diagnostic metadata for an anonymous inline child. It is
    deliberately absent from `allowed`, never authorable, and excluded from descriptor equality.

    Class attrs: `kind="loop"`; `allowed` adds `call`/`inputs`/`while`/`until`/`times`/`max`/
    `asserts` to the commons; `required` is `{"call"}` (the body subflow); `binding_fields=()` —
    a loop's `inputs:` are NOT inline-call-desugared today, so the kind-blind walk skips it.
    `keyword_remap` is loop-owned: it maps the surface Python keywords `while:`/`until:` onto the
    `while_`/`until_` descriptor fields so the inherited template `parse` reads them correctly.
    """

    kind = "loop"
    allowed = NodeDescriptor.allow("call", "inputs", "while", "until", "times", "max", "asserts")
    required = NodeDescriptor.require("call")
    binding_fields = ()
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

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `LoopNode` (+ wiring) for this `LoopDescriptor` — a
        `('a -> 'a) -> 'a -> 'a` driver — resolve-and-bake, then stamp `.range`/`.field_ranges`
        via `_finish`.

        `loop` is a higher-order driver: the body subflow (`call:`) reads a subset of the
        carried record (`inputs:`) and returns the WHOLE next carried record. Returns a
        `BuildResult` (the split): the node carries `params`; the flow owns the carried-record
        seed sources in `wiring`. Resolves the callable through `ctx.resolver` (required — a loop
        is higher-order), derives its `ChildSignature`, bakes the compiled child + parameters,
        stamps `output_type` (the body codomain, which IS the carried record type), and enforces
        the FIELD-NAME half of the contract + slice legality (`while:` present, `max:` present).
        The deep TYPE-level half (body output Type == carried record Type) is the loader pass
        `check_loop_type_contract`, which needs the assembled parent producers/flow-input types.
        `ctx.registry` is unused (loop resolves its output type from the body signature); the
        inline-CODE anchors on `ctx` are read only by the CODE build.
        """
        resolver = ctx.resolver
        child = _resolve_child(
            self.id,
            self.call,
            resolver,
            self.child_label,
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
            child_source=child.source,        # render-only; the CLI nested-error traceback
            title=self.node_name,
        )
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        node.output_type = sig.output          # the body codomain == the carried record type

        errors: list[str] = []
        # Predicate slice legality: exactly one of `while:`/`until:`/`times:` selects the loop's
        # driver. `while`/`until` are PREDICATE loops (record-scoped boolean, `max:` runaway
        # guard); `times: N` is a fixed COUNT (no predicate, `max:` redundant since the count
        # bounds it). Bake `predicate_kind`/`predicate`/`times`/`max_iters` per the present kind.
        present = [k for k, v in (("while", self.while_), ("until", self.until_), ("times", self.times)) if v is not None]
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
            if self.max is None:
                errors.append(f"loop node {self.id!r}: `max:` is required (runaway guard)")
            elif not isinstance(self.max, int) or isinstance(self.max, bool):
                # `max:` bounds the iteration COUNT, so it must be a plain integer. The descriptor
                # is a bare dataclass (its `Optional[int]` annotation isn't enforced), so a YAML
                # string/float passes through and would blow up the `< 1` compare or the runtime
                # guard, and a YAML bool (an int subclass in Python) would silently read as 0/1.
                # Reject any non-int type.
                errors.append(f"loop node {self.id!r}: `max:` must be an integer (got {self.max!r})")
            elif self.max < 1:
                # A runaway guard below 1 is nonsensical: the guard bounds the iteration COUNT, so
                # it must permit at least one body run. Reject at the boundary rather than letting
                # `max: 0` run one iteration and then fail with a confusing "exceeded max (0)".
                errors.append(f"loop node {self.id!r}: `max:` must be >= 1 (got {self.max})")
        elif present[0] == "times":
            # `times: N` is a fixed count: exactly N runs, no predicate. `max_iters` doubles as
            # the count N so the runtime guard needs no `times`-special case. Same int/bool/`>= 1`
            # guard as `max:` — the count must be a plain integer permitting at least one run.
            node.predicate_kind = "times"
            node.predicate = None
            node.times = self.times
            node.max_iters = self.times
            if not isinstance(self.times, int) or isinstance(self.times, bool):
                errors.append(f"loop node {self.id!r}: `times:` must be an integer (got {self.times!r})")
            elif self.times < 1:
                errors.append(f"loop node {self.id!r}: `times:` must be >= 1 (got {self.times})")
            if self.max is not None:
                errors.append(
                    f"loop node {self.id!r}: max: is redundant with times: (times bounds the count)"
                )

        # The `'a -> 'a` FIELD-NAME contract (types are the loader pass). The carried record's
        # keys are the seed `inputs:` names. The body OUTPUT field NAMES must equal them (the
        # whole next state threads back); the body INPUT NAMES must be a subset (it reads a
        # slice of the carried record). A non-record body output (no fields) can't match a
        # multi-field carried record — flag it too.
        carried = set(self.inputs or {})
        out_names = _type_field_names(sig.output)
        if out_names is None:
            errors.append(
                f"loop node {self.id!r}: body {child_display} must declare a record "
                f"output whose field names equal the carried record {sorted(carried)}"
            )
        elif out_names != carried:
            errors.append(
                f"loop node {self.id!r}: body output fields {sorted(out_names)} must equal the "
                f"carried record fields {sorted(carried)} (the whole next state threads back)"
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
        # keys — an undeclared name silently resolves falsy and spins the loop to `max`. Reject a
        # predicate whose ref head isn't a carried field at load, so a typo is loud not a runaway.
        # `times:` has no predicate, so skip it.
        predicate_src = self.while_ or self.until_
        if predicate_src:
            pred_kw = "while" if self.while_ else "until"
            # The unified condition ref-walk: works for any spelling (bare `score >= 0.8`,
            # mixed `${score} >= 0.8`, whole `${score >= 0.8}`), unlike a flat `${...}` regex
            # which read a whole-span predicate as one bogus head and a bare one as no heads.
            try:
                heads = {r.split(".", 1)[0] for r in condition_refs(predicate_src)}
            except ExpressionError as exc:
                errors.append(
                    f"loop node {self.id!r}: `{pred_kw}:` is not a valid boolean expression: {exc}"
                )
                heads = set()
            unknown = sorted(heads - carried)
            if unknown:
                errors.append(
                    f"loop node {self.id!r}: `{pred_kw}:` references unknown name(s) {unknown} — "
                    f"the predicate reads the carried record {sorted(carried)}"
                )
        if errors:
            raise LoadError("\n  ".join(errors))
        return self._finish(node, wiring)


def _type_field_names(typ: Optional[Type]) -> Optional[set[str]]:
    """The per-field NAMES of a record `Type`, or `None` when `typ` is not a
    closed record (no `fields:` — a scalar/list/opaque codomain). Used for the
    body-output field-NAME-set half of the loop's `'a -> 'a` contract."""
    if typ is None or typ.fields is None:
        return None
    return set(typ.fields.keys())
