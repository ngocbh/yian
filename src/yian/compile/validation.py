"""Representation-neutral graph + reference leaf checkers.

These operate on bare `Type`/`Edge`/path-set inputs (never a `FlowSpec`), so the
Compose loader (`yian.compose`) reuses them directly:

- `_reject_cycles` — Kahn's-algorithm cycle check over a bare edge list + node-id set.
- `_classify_path` — head-resolution for one dotted `${...}` ref + the dotted
  record-field walk, dispatched through `_walk_record_fields`.
- `_walk_record_fields` — the dotted-field-into-a-record check.

`FlowValidationError` is the shared structural-error type the compose loader catches
and re-raises as a located `LoadError`.
"""


from dataclasses import replace

from yian.compile.model import END_ID, START_ID
from yian.nodes.contracts import ValueContract, output_contract_field_error
from yian.typesys.values import ValueKind, Type

_CLOSED_HEADS = ()              # `outputs` is no longer a head; node-first head dispatch
_SYSTEM_AMBIENTS = ("today", "now", "run_id")  # the ONLY valid ${system.X} ambients
#   today=date / now=datetime — the host clock seam; run_id=str — the run-scoped id
#   (a host-injected ambient, like the clock; seeded once at the boundary, child-inherited)


class FlowValidationError(ValueError):
    """
    A flow graph is structurally invalid and cannot be compiled or run.

    Raised by the graph/reference checkers and caught by the compose loader, which
    re-raises it as a located [`LoadError`][yian.LoadError].

    Attributes:
        errors (`list[str]`):
            All accumulated located problems. The reference-wiring pass collects every
            problem it finds; fail-fast structural checks carry the single message.
    """

    def __init__(self, message: str, errors: "list[str] | None" = None) -> None:
        super().__init__(message)
        self.errors = errors if errors is not None else [message]


def _reject_cycles(edges: "list", node_ids: set[str]) -> None:
    """Raise if the real-node graph has a directed cycle (Kahn's algorithm).

    Edges touching `__start__`/`__end__` can't be part of a cycle (the sentinels
    have no incoming/outgoing respectively), so only real-node edges count.

    Takes the bare `edges` list (any object with `.from_`/`.to`) + the node-id set,
    so the loader can reuse it on its synthesized `Edge`s.
    """
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    for edge in edges:
        if edge.from_ in (START_ID, END_ID) or edge.to in (START_ID, END_ID):
            continue
        adjacency[edge.from_].append(edge.to)
        in_degree[edge.to] += 1

    queue = [nid for nid in node_ids if in_degree[nid] == 0]
    visited = 0
    while queue:
        nid = queue.pop()
        visited += 1
        for nxt in adjacency[nid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if visited != len(node_ids):
        stuck = sorted(nid for nid in node_ids if in_degree[nid] > 0)
        raise FlowValidationError(f"flow has a cycle involving {stuck}; flows must be acyclic")


def types_compatible(source: Type, sink: Type) -> bool:
    """Return structural compatibility of a source value type with a sink slot.

    Same `kind` (with int->number widening, mirroring `_coerce_scalar`); the singleton
    `None` source may fill any nullable sink; records match structurally (every field the
    sink REQUIRES is present + compatible); enums by tag-subset (source tags within the
    sink's accepted tags); list elements recurse, treating a bare specialized list as its
    non-null scalar element contract; a nullable source cannot fill a non-nullable sink.
    A bounded numeric sink requires a source interval contained within its bounds.
    A source union fits a sink union when each of its members fits some sink member; a
    record fits a union when it fits some member; a union fits `object` but never a
    record, and `object` never fits a union.

    Args:
        source (`Type`): Type guaranteed by the producer.
        sink (`Type`): Type required by the consumer slot.

    Returns:
        `bool`: Whether every value admitted by ``source`` is admitted by ``sink``.
    """
    if source.kind is ValueKind.NONE:
        return sink.kind is ValueKind.NONE or sink.nullable
    if source.nullable and not sink.nullable:
        return False
    source_is_list = source.kind.is_list()
    sink_is_list = sink.kind.is_list()
    if source_is_list or sink_is_list:
        if not (source_is_list and sink_is_list):
            return False
        source_element = source.element
        if source_element is None and source.kind.element_kind is not None:
            source_element = Type.scalar(source.kind.element_kind)
        sink_element = sink.element
        if sink_element is None and sink.kind.element_kind is not None:
            sink_element = Type.scalar(sink.kind.element_kind)
        if sink_element is None:
            return True
        return source_element is not None and types_compatible(
            source_element,
            sink_element,
        )
    if source.kind != sink.kind:
        if not (sink.kind == ValueKind.NUMBER and source.kind == ValueKind.INTEGER):
            return False
    if sink.bounds is not None and (
        source.bounds is None
        or source.bounds[0] < sink.bounds[0]
        or source.bounds[1] > sink.bounds[1]
    ):
        return False
    if sink.variants is not None:
        if source.variants is not None:
            members = source.variants
        elif source.fields is not None:
            # Members are non-nullable; the union's own nullability was checked above.
            members = (replace(source, nullable=False),)
        else:
            return False
        return all(
            any(types_compatible(member, option) for option in sink.variants)
            for member in members
        )
    if sink.fields is not None:
        # A union may hold a member other than this record, and load can't tell which.
        if source.variants is not None:
            return False
        if source.fields is None:
            return False
        for fname in (sink.required or frozenset()):
            if fname not in source.fields:
                return False
        for fname, ftyp in sink.fields.items():
            if fname in source.fields and not types_compatible(source.fields[fname], ftyp):
                return False
    if sink.tags is not None and source.tags is not None:
        if not source.tags <= sink.tags:
            return False
    return True


def _walk_record_fields(typ: "Type | None", fields: list, path: str) -> "str | None":
    """Walk a dotted field path into a producer's record `Type`.

    Only a CHECKED record (`typ.fields` is not None) is walked; a scalar / opaque /
    unresolved producer stays lenient (dotted access allowed, unchecked). An absent
    field on a checked record is a located compile error. A union is checked by the
    shared output-contract walk from there on."""
    for index, f in enumerate(fields):
        if typ is not None and typ.variants is not None:
            return output_contract_field_error(ValueContract(typ), fields[index:], path)
        if typ is None or typ.fields is None:
            return None  # opaque / not a checked record -> lenient
        if f not in typ.fields:
            return f"reference ${{{path}}} reads unknown field {f!r} on record"
        typ = typ.fields[f]
    return None


def _classify_path(
    path: str,
    valid_targets: set[str],
    flow_inputs: set[str],
    extra_heads: tuple = (),
    producers: "dict[str, Type] | None" = None,
) -> "str | None":
    """Return an error message for a bad reference path, or None if acceptable.

    `path` is a single dotted ref the binding actually reads (`expr.template`'s
    `expr_refs_of` already collected the per-span reference paths). Empty path
    segments are rejected at parse, so the head is always present here.

    - `${input.X}` closed-checks against the flow's declared inputs.
    - `${<node>.output[.X]}` (node-first) checks against real node ids; the second
      segment MUST be the literal `output` (the syntactic discriminator).
    - `${system.X}` is the strict clock ambient namespace.
    - `extra_heads` is a per-call set of body-local scopes (e.g. MAP's `${item}`).
    - Legacy `${inputs.X}` / `${outputs.X}` return the retired-plural-head typo hint
      ("did you mean `input` / `<node>.output`?").
    """
    parts = path.split(".")
    head = parts[0]
    # Singular `input` head.
    if head == "input":
        if len(parts) < 2:
            return f"reference ${{{path}}} must be input.<name>"
        if parts[1] not in flow_inputs:
            return f"reference ${{{path}}} points to unknown flow input {parts[1]!r}"
        return None
    if head in ("inputs", "outputs"):  # transitional typo hint for the retired plural heads
        suggestion = "input" if head == "inputs" else "<node>.output"
        return (
            f"reference ${{{path}}} uses the retired plural head {head!r} "
            f"(did you mean `{suggestion}`?)"
        )
    if head in extra_heads:
        return None  # body-local scope (e.g. ${item}) — lenient, dotted access allowed
    if head == "system":
        # strict ambient namespace: only the declared ambients are valid
        if len(parts) < 2 or parts[1] not in _SYSTEM_AMBIENTS:
            bad = parts[1] if len(parts) >= 2 else ""
            return f"reference ${{{path}}} uses unknown system ambient {bad!r} (only today/now/run_id)"
        return None
    # node-id head — `${<node>.output[.X]}`.
    if head in valid_targets:
        # the 1-segment form `${<node>}` is an error (must say `<node>.output[.path]`).
        if len(parts) < 2:
            return f"reference ${{{path}}} must be {head}.output[.path]"
        # the second segment MUST be the literal `output` (the discriminator).
        if parts[1] != "output":
            return (
                f"reference ${{{path}}} on node {head!r} must be {head}.output[.field] "
                f"(`.output` is the node-value selector)"
            )
        # dotted-field walk into the producer's record Type, starting at parts[2:].
        return _walk_record_fields((producers or {}).get(head), parts[2:], path)
    return f"reference ${{{path}}} uses unknown namespace {head!r}"
