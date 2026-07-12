"""kind=loop — the `('a -> 'a) -> 'a -> 'a` driver's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `loop` node's frozen
`LoopDescriptor`, its located `parse(node_id, reader)` (recording each field's `SourceRange`),
and the resolve-and-bake `build` (predicate-slice legality + `'a -> 'a` field-name contract).
Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.expr import ExpressionError, condition_refs
from agent_composer.nodes.loop import LoopNode
from agent_composer.typesys.values import Type
from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.compose.kinds.helpers import (
    _KEYWORD_FIELD_REMAP,
    _sink_params,
    _sink_wiring,
    check_node_keys,
    read_common_fields,
)
from agent_composer.compose.kinds.resolver import (
    _resolve_child,
    child_signature,
)
from agent_composer.events import SourceRange

# Loop's own field surface (was the `loop` row of parser._KIND_SPECS): `call` (the body subflow)
# is the one required flat field; `inputs` seeds the carried record; exactly one of
# `while`/`until`/`times` selects the driver (checked at build), and `max` is the runaway guard.
LOOP_REQUIRED: frozenset[str] = frozenset({"call"})
LOOP_ALLOWED: frozenset[str] = frozenset(
    {"call", "inputs", "while", "until", "times", "max", "asserts"}
)
# A `loop` carries no `${...}` sink bindings the kind-blind walk in `bindings.py` should touch:
# `LoopDescriptor` is absent from the legacy `calls.py:map_binding_strings_in_descriptor` ladder,
# so its `inputs:` are NOT inline-call-desugared today. Empty preserves that behavior.
LOOP_BINDING_FIELDS: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class LoopDescriptor(NodeDescriptor):
    """kind=loop — a `('a -> 'a) -> 'a -> 'a` driver over a carried record.

    `call` names the body subflow (defs-first, else external — same resolution as
    `call`/`map`). `inputs` seeds the carried record at turn 0. Exactly one of
    `while_`/`until_`/`times` carries the predicate; `max` is the runaway guard (required for
    while/until). `while`/`until` are Python keywords, so they land here as `while_`/`until_`
    (the parse remap, like `else`->`else_`).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`; only the loop-specific
    fields are declared here.
    """

    call: str
    inputs: dict[str, Any] = field(default_factory=dict)
    while_: Optional[str] = None
    until_: Optional[str] = None
    times: Optional[int] = None
    max: Optional[int] = None
    asserts: list[str] = field(default_factory=list)


def parse(node_id: str, reader: NodeReader) -> LoopDescriptor:
    """Read one `kind: loop` node body (via a located `NodeReader`) into a `LoopDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's `SourceRange`
    (`desc.field_ranges[field]`). Field legality mirrors the legacy `parser._parse_node` loop
    path. The surface keys `while:`/`until:` (Python keywords) map onto the `while_`/`until_`
    fields (via `helpers._KEYWORD_FIELD_REMAP`).
    """
    check_node_keys(node_id, "loop", reader, LOOP_ALLOWED, LOOP_REQUIRED)
    body = reader.value

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in LOOP_ALLOWED:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        # `while`/`until` -> `while_`/`until_` (the descriptor field names); the rest unchanged.
        kwargs[_KEYWORD_FIELD_REMAP.get(key, key)] = body[key]

    asserts = kwargs.get("asserts")
    if asserts is not None and not (
        isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
    ):
        raise LoadError(
            f"node {node_id!r} (kind=loop): `asserts:` must be a list of strings",
            range=ranges.get("asserts"),
        )

    read_common_fields(node_id, "loop", reader, kwargs, ranges)

    return LoopDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def _type_field_names(typ: Optional[Type]) -> Optional[set[str]]:
    """The per-field NAMES of a record `Type`, or `None` when `typ` is not a
    closed record (no `fields:` — a scalar/list/opaque codomain). Used for the
    body-output field-NAME-set half of the loop's `'a -> 'a` contract."""
    if typ is None or typ.fields is None:
        return None
    return set(typ.fields.keys())


def build(desc: LoopDescriptor, *, registry=None, resolver, **_loader_hints):
    """Construct the runtime `LoopNode` (+ wiring) for one `LoopDescriptor` — a
    `('a -> 'a) -> 'a -> 'a` driver — resolve-and-bake, then stamp `.range`/`.field_ranges`.

    `loop` is a higher-order driver: the body subflow (`call:`) reads a subset of the
    carried record (`inputs:`) and returns the WHOLE next carried record. Returns
    `(node, wiring)` (the split): the node carries `params`; the flow owns the carried-record
    seed sources in `wiring`. Resolves the callable, derives its `ChildSignature`, bakes the
    compiled child + input decls, stamps `output_type` (the body codomain, which IS the carried
    record type), and enforces the FIELD-NAME half of the contract + slice legality (`while:`
    present, `max:` present). The deep TYPE-level half (body output Type == carried record Type)
    is the loader pass `check_loop_type_contract`, which needs the assembled parent
    producers/flow-input types. `registry` is unused (loop resolves its output type from the
    body signature); `_loader_hints` absorbs the inline-CODE anchors only CODE reads.
    """
    child = _resolve_child(desc.id, desc.call, resolver)
    sig = child_signature(child)
    # Stamp the child's AssertSet onto its compiled flow (the Grow subgraph body), exactly as
    # the call/map build — clone_child reads child_asserts off the cloned body child.
    child.compiled.child_asserts = child.asserts
    node = LoopNode(
        desc.id,
        flow_id=desc.call,
        flow_version=None,
        child=child.compiled,
        child_inputs=child.input,
        child_asserts=child.asserts,
        child_source=child.source,        # render-only; the CLI nested-error traceback
        title=desc.node_name,
    )
    node.params = _sink_params(desc.inputs)
    wiring = _sink_wiring(desc.inputs)
    node.output_type = sig.output          # the body codomain == the carried record type

    errors: list[str] = []
    # Predicate slice legality: exactly one of `while:`/`until:`/`times:` selects the loop's
    # driver. `while`/`until` are PREDICATE loops (record-scoped boolean, `max:` runaway guard);
    # `times: N` is a fixed COUNT (no predicate, `max:` redundant since the count bounds it). Bake
    # `predicate_kind`/`predicate`/`times`/`max_iters` per the present kind below.
    present = [k for k, v in (("while", desc.while_), ("until", desc.until_), ("times", desc.times)) if v is not None]
    if len(present) != 1:
        errors.append(
            f"loop node {desc.id!r}: exactly one of while:/until:/times: is required "
            f"(got {sorted(present)})"
        )
    elif present[0] in ("while", "until"):
        kind = present[0]
        node.predicate_kind = kind
        node.predicate = desc.while_ if kind == "while" else desc.until_
        node.max_iters = desc.max
        if desc.max is None:
            errors.append(f"loop node {desc.id!r}: `max:` is required (runaway guard)")
        elif not isinstance(desc.max, int) or isinstance(desc.max, bool):
            # `max:` bounds the iteration COUNT, so it must be a plain integer. The descriptor is a
            # bare dataclass (its `Optional[int]` annotation isn't enforced), so a YAML string/float
            # passes through and would blow up the `< 1` compare or the runtime guard, and a YAML
            # bool (an int subclass in Python) would silently read as 0/1. Reject any non-int type.
            errors.append(f"loop node {desc.id!r}: `max:` must be an integer (got {desc.max!r})")
        elif desc.max < 1:
            # A runaway guard below 1 is nonsensical: the guard bounds the iteration COUNT, so it
            # must permit at least one body run. Reject at the boundary rather than letting `max: 0`
            # run one iteration and then fail with a confusing "exceeded max (0)".
            errors.append(f"loop node {desc.id!r}: `max:` must be >= 1 (got {desc.max})")
    elif present[0] == "times":
        # `times: N` is a fixed count: exactly N runs, no predicate. `max_iters` doubles as the
        # count N so the runtime guard needs no `times`-special case. Same int/bool/`>= 1` guard as
        # `max:` — the count must be a plain integer permitting at least one run.
        node.predicate_kind = "times"
        node.predicate = None
        node.times = desc.times
        node.max_iters = desc.times
        if not isinstance(desc.times, int) or isinstance(desc.times, bool):
            errors.append(f"loop node {desc.id!r}: `times:` must be an integer (got {desc.times!r})")
        elif desc.times < 1:
            errors.append(f"loop node {desc.id!r}: `times:` must be >= 1 (got {desc.times})")
        if desc.max is not None:
            errors.append(
                f"loop node {desc.id!r}: max: is redundant with times: (times bounds the count)"
            )

    # The `'a -> 'a` FIELD-NAME contract (types are the loader pass). The carried record's
    # keys are the seed `inputs:` names. The body OUTPUT field NAMES must equal them (the
    # whole next state threads back); the body INPUT NAMES must be a subset (it reads a
    # slice of the carried record). A non-record body output (no fields) can't match a
    # multi-field carried record — flag it too.
    carried = set(desc.inputs or {})
    out_names = _type_field_names(sig.output)
    if out_names is None:
        errors.append(
            f"loop node {desc.id!r}: body callable {desc.call!r} must declare a record "
            f"output whose field names equal the carried record {sorted(carried)}"
        )
    elif out_names != carried:
        errors.append(
            f"loop node {desc.id!r}: body output fields {sorted(out_names)} must equal the "
            f"carried record fields {sorted(carried)} (the whole next state threads back)"
        )
    body_inputs = {d.name for d in sig.inputs}
    extra = body_inputs - carried
    if extra:
        errors.append(
            f"loop node {desc.id!r}: body input(s) {sorted(extra)} are not carried record "
            f"fields {sorted(carried)} (the body reads a subset of the carried record)"
        )
    # `while:`/`until:` predicate scope: the predicate is evaluated against the carried record
    # (record-scoped `evaluate_when_record`), where the ONLY resolvable names are the carried
    # keys — an undeclared name silently resolves falsy and spins the loop to `max`. Reject a
    # predicate whose ref head isn't a carried field at load, so a typo is loud not a runaway.
    # `times:` has no predicate, so skip it.
    predicate_src = desc.while_ or desc.until_
    if predicate_src:
        pred_kw = "while" if desc.while_ else "until"
        # The unified condition ref-walk: works for any spelling (bare `score >= 0.8`,
        # mixed `${score} >= 0.8`, whole `${score >= 0.8}`), unlike a flat `${...}` regex
        # which read a whole-span predicate as one bogus head and a bare one as no heads.
        try:
            heads = {r.split(".", 1)[0] for r in condition_refs(predicate_src)}
        except ExpressionError as exc:
            errors.append(
                f"loop node {desc.id!r}: `{pred_kw}:` is not a valid boolean expression: {exc}"
            )
            heads = set()
        unknown = sorted(heads - carried)
        if unknown:
            errors.append(
                f"loop node {desc.id!r}: `{pred_kw}:` references unknown name(s) {unknown} — "
                f"the predicate reads the carried record {sorted(carried)}"
            )
    if errors:
        raise LoadError("\n  ".join(errors))
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring
