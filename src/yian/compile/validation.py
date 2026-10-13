"""Representation-neutral graph + reference leaf checkers.

These operate on bare `Type`/`Edge`/path-set inputs (never a `FlowSpec`), so the
Compose loader (`yian.compose`) reuses them directly:

- `_reject_cycles` — Kahn's-algorithm cycle check over a bare edge list + node-id set.
- `_classify_path` — head-resolution for one dotted `${...}` ref. Field reads are
  checked by the shared output-contract walk in `nodes/contracts.py`.
- `types_compatible` — whether every value of a source type fits a sink type.
- `type_relation` — how an inferred type relates to a declared one: it fits, is
  narrower (only some values fit), or is disjoint (no value fits).

`FlowValidationError` is the shared structural-error type the compose loader catches
and re-raises as a located `LoadError`.
"""

import math
from typing import Literal

from dataclasses import replace

from yian.compile.model import END_ID, START_ID
from yian.typesys.values import TypeCheckError, ValueKind, Type, build_value_as

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
        source_element = _list_element(source)
        sink_element = _list_element(sink)
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


TypeRelation = Literal["fits", "narrower", "disjoint"]

_RELATION_RANK = {"disjoint": 0, "narrower": 1, "fits": 2}

# Date and datetime values are stored as ISO strings, so each fits a `str` slot.
_STRING_STORED = (ValueKind.DATE, ValueKind.DATETIME)


def type_relation(inferred: Type, declared: Type) -> TypeRelation:
    """Classify how an inferred value type relates to a declared type.

    - ``"fits"``: every value of ``inferred`` is a value of ``declared``; no runtime
      check is needed.
    - ``"narrower"``: some values of ``inferred`` are values of ``declared`` and some
      may not be; the binding is accepted and checked at runtime.
    - ``"disjoint"``: no non-null value of ``inferred`` is a value of ``declared``;
      the binding can never succeed. Sharing only `None` does not count as overlap,
      so ``Optional[str]`` against ``Optional[int]`` is disjoint. This null-only
      rule applies to the whole value: a record whose differing field both sides
      may omit (or may both hold `None`) still shares non-null record values, so it
      is narrower, not disjoint.

    This mirrors `types_compatible` for ``"fits"`` but is stricter where that check
    is lenient for wiring: a tag-less string into a `Literal` is only narrower, and
    records are treated as closed, as the write-boundary check treats them (unknown
    fields are rejected), so a record fits only when its fields are a subset of the
    declared fields at every level. It is also more permissive in one place: `date`
    and `datetime` fit `str`, since they are stored as strings. A `Literal` meets
    `date` or `datetime` only through tags that are valid ISO strings of that kind.
    Unions are compared member-wise: each inferred member takes its best relation
    against the declared members, so a union member over the union is narrower and
    a record matching no member is disjoint.

    Args:
        inferred (`Type`): Type the producing expression is inferred to have.
        declared (`Type`): Type the binding declares.

    Returns:
        `TypeRelation`: ``"fits"``, ``"narrower"``, or ``"disjoint"``.
    """
    if inferred.kind is ValueKind.NONE:
        return "fits" if _accepts_none(declared) else "disjoint"
    if declared.kind is ValueKind.NONE:
        return "disjoint"
    relation = _relate_non_null(inferred, declared)
    if relation == "fits" and inferred.nullable and not declared.nullable:
        return "narrower"
    return relation


def _accepts_none(typ: Type) -> bool:
    """Return whether ``typ`` admits the `None` value."""
    return typ.kind is ValueKind.NONE or typ.nullable


def _weakest(*relations: TypeRelation) -> TypeRelation:
    """Return the least permissive of ``relations``."""
    return min(relations, key=_RELATION_RANK.__getitem__)


def _list_element(typ: Type) -> "Type | None":
    """Return a list type's element contract, or `None` for an unrefined list."""
    if typ.element is not None:
        return typ.element
    if typ.kind.element_kind is not None:
        return Type.scalar(typ.kind.element_kind)
    return None


def _relate_non_null(inferred: Type, declared: Type) -> TypeRelation:
    """Classify the non-null values of ``inferred`` against ``declared``."""
    if inferred.variants is not None or declared.variants is not None:
        return _relate_union(inferred, declared)
    inferred_is_list = inferred.kind.is_list()
    declared_is_list = declared.kind.is_list()
    if inferred_is_list or declared_is_list:
        if not (inferred_is_list and declared_is_list):
            return "disjoint"
        declared_element = _list_element(declared)
        if declared_element is None:
            return "fits"
        inferred_element = _list_element(inferred)
        if inferred_element is None:
            return "narrower"
        # The empty list is shared by every pair of list types; it is not overlap.
        return type_relation(inferred_element, declared_element)

    relation = _relate_kinds(inferred.kind, declared.kind)
    if relation == "disjoint":
        return relation
    return _weakest(
        relation,
        _relate_bounds(inferred, declared),
        _relate_tags(inferred, declared),
        _relate_fields(inferred, declared),
    )


def _relate_union(inferred: Type, declared: Type) -> TypeRelation:
    """Classify the non-null values of ``inferred`` against ``declared`` member-wise.

    Either side may be a union; a non-union side counts as a one-member union.
    Each inferred member takes its best relation against the declared members,
    since a value fits the union when it fits any member. The results then
    combine as alternatives: fits when every member fits, disjoint when every
    member is disjoint, and narrower otherwise.

    Args:
        inferred (`Type`): Inferred type; a union or any other type.
        declared (`Type`): Declared type; a union or any other type.

    Returns:
        `TypeRelation`: ``"fits"``, ``"narrower"``, or ``"disjoint"``.
    """
    inferred_members = inferred.variants or (inferred,)
    declared_members = declared.variants or (declared,)
    relations = {
        max(
            (_relate_non_null(member, option) for option in declared_members),
            key=_RELATION_RANK.__getitem__,
        )
        for member in inferred_members
    }
    if len(relations) == 1:
        return relations.pop()
    return "narrower"


def _relate_kinds(inferred: ValueKind, declared: ValueKind) -> TypeRelation:
    """Classify two non-list storage kinds before refinements are compared."""
    if inferred is declared:
        return "fits"
    if inferred is ValueKind.INTEGER and declared is ValueKind.NUMBER:
        return "fits"
    if inferred in _STRING_STORED and declared is ValueKind.STRING:
        return "fits"
    if inferred is ValueKind.STRING and declared in _STRING_STORED:
        return "narrower"
    return "disjoint"


def _relate_bounds(inferred: Type, declared: Type) -> TypeRelation:
    """Compare numeric intervals; an absent bound is unbounded."""
    if declared.bounds is None:
        return "fits"
    low, high = inferred.bounds if inferred.bounds is not None else (-math.inf, math.inf)
    if declared.bounds[0] <= low and high <= declared.bounds[1]:
        return "fits"
    shared_low = max(low, declared.bounds[0])
    shared_high = min(high, declared.bounds[1])
    if inferred.kind is ValueKind.INTEGER:
        # An int interval overlaps only if a whole number lies in the shared range.
        shared_low, shared_high = math.ceil(shared_low), math.floor(shared_high)
    return "narrower" if shared_low <= shared_high else "disjoint"


def _relate_tags(inferred: Type, declared: Type) -> TypeRelation:
    """Compare `Literal` tag sets; a tag-less source into a `Literal` is narrower.

    A `Literal` against `date` or `datetime` (either way round) overlaps only
    when some tag is a valid ISO string of that kind.
    """
    if inferred.tags is not None and declared.kind in _STRING_STORED:
        return _tags_meet_kind(inferred.tags, declared.kind)
    if declared.tags is None:
        return "fits"
    if inferred.kind in _STRING_STORED:
        return _tags_meet_kind(declared.tags, inferred.kind)
    if inferred.tags is None:
        return "narrower"
    if inferred.tags <= declared.tags:
        return "fits"
    return "narrower" if inferred.tags & declared.tags else "disjoint"


def _tags_meet_kind(tags: frozenset[str], kind: ValueKind) -> TypeRelation:
    """Return narrower when some tag is a valid ``kind`` string, else disjoint."""
    for tag in tags:
        try:
            build_value_as(kind, tag)
        except TypeCheckError:
            continue
        return "narrower"
    return "disjoint"


def _relate_fields(inferred: Type, declared: Type) -> TypeRelation:
    """Compare record fields, treating both records as closed."""
    if declared.fields is None:
        return "fits"
    if inferred.fields is None:
        return "narrower"
    inferred_required = inferred.required or frozenset()
    declared_required = declared.required or frozenset()
    if not inferred_required <= declared.fields.keys():
        return "disjoint"
    if not declared_required <= inferred.fields.keys():
        return "disjoint"
    relation: TypeRelation = "fits"
    if not inferred.fields.keys() <= declared.fields.keys():
        relation = "narrower"
    if not declared_required <= inferred_required:
        relation = "narrower"
    for name, inferred_field in inferred.fields.items():
        declared_field = declared.fields.get(name)
        if declared_field is None:
            continue
        field_relation = type_relation(inferred_field, declared_field)
        if field_relation == "disjoint":
            # Two records still share values when both may omit the field or both
            # may hold `None` there.
            both_omit = name not in inferred_required and name not in declared_required
            both_none = _accepts_none(inferred_field) and _accepts_none(declared_field)
            if not (both_omit or both_none):
                return "disjoint"
            field_relation = "narrower"
        relation = _weakest(relation, field_relation)
    return relation


def _classify_path(
    path: str,
    valid_targets: set[str],
    flow_inputs: set[str],
    extra_heads: tuple = (),
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
        return None
    return f"reference ${{{path}}} uses unknown namespace {head!r}"
