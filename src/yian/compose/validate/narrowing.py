"""Union narrowing: what each node may assume about union values, from the CASEs that route it.

A narrowing fact says which part of a union a value at a base path can be. A base path is
`<node>.output[.field...]` or `input.<name>[.field...]`. Facts come from CASE branches: an
`on:` CASE over a tag field of a union gives each `when:` edge the member with that tag and the
`else:` edge the unlisted members.

Facts flow along the graph in topological order. A node's control edges join by union (it ran
because one of them fired), each required data group joins its producers by union, and every
contribution (control, required data groups, `depends_on`) meets by intersection. Each node
then gets a view of the producer and flow-input contracts with its narrowed bases rebuilt, so
the reference and type checks can read member fields inside a branch.

The pass reads built, desugared CASE descriptors and their edges, so it runs after
`desugar_case_call_targets` and the node builds.

Knows about: descriptors, compiled edges, output contracts, and `typesys` unions.
Never imports: runtime.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence, TypeAlias

from yian.compile.model import Edge
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.output_contracts import (
    combine_output_contracts,
    direct_reference_parts,
    project_output_contract,
    record_field_origins,
)
from yian.compose.nodes.base import NodeDescriptor
from yian.compose.nodes.case import CaseDescriptor
from yian.expr import ExpressionError
from yian.expr.template import Span, scan_template
from yian.nodes.case import DEFAULT_HANDLE
from yian.nodes.contracts import (
    OutputContract,
    PartialRecordContract,
    ValueContract,
)
from yian.typesys.values import (
    Type,
    ValueKind,
    canonical_union_order,
    union_tag_fields,
)


@dataclass(frozen=True)
class Members:
    """The value is one of `members`, all members of the base's union, or `None` if nullable.

    Members are deduplicated by `==` and kept in canonical union order. An empty set
    that is not nullable means no value satisfies the facts, so the node never runs.

    Args:
        members (`tuple[Type, ...]`):
            Non-nullable record members of the base's union.
        nullable (`bool`, *optional*, defaults to `False`):
            Whether the value may also be `None`.
    """

    members: tuple[Type, ...]
    nullable: bool = False

    def __post_init__(self) -> None:
        """Deduplicate the members and put them in canonical order."""
        unique: list[Type] = []
        for member in self.members:
            if not any(member == seen for seen in unique):
                unique.append(member)
        object.__setattr__(self, "members", canonical_union_order(unique))


@dataclass(frozen=True)
class Exact:
    """The value satisfies `type`.

    Args:
        type (`Type`):
            The type the value is known to satisfy.
    """

    type: Type


Fact: TypeAlias = Members | Exact
"""One narrowing fact for a base path."""

Env: TypeAlias = dict[str, Fact]
"""Facts keyed by dotted base path; an absent base is unconstrained."""


def _contains(members: Sequence[Type], typ: Type) -> bool:
    """Whether `typ` equals one of `members`."""
    return any(member == typ for member in members)


def union_facts(a: Optional[Fact], b: Optional[Fact]) -> Optional[Fact]:
    """Return what holds when either fact holds, or `None` for no fact.

    Args:
        a (`Fact | None`):
            One fact; `None` is unconstrained.
        b (`Fact | None`):
            The other fact; `None` is unconstrained.

    Returns:
        `Fact | None`:
            The set union of two `Members`, an `Exact` both sides share, or `None`.

    Example:
        `union_facts(Members((Buy,)), Members((Sell,)))` is `Members((Buy, Sell))`;
        `union_facts(Members((Buy,)), None)` is `None`.
    """
    if isinstance(a, Members) and isinstance(b, Members):
        return Members(a.members + b.members, a.nullable or b.nullable)
    if isinstance(a, Exact) and isinstance(b, Exact) and a.type == b.type:
        return a
    return None


def intersect_facts(a: Optional[Fact], b: Optional[Fact]) -> Optional[Fact]:
    """Return what holds when both facts hold, or `None` for no fact.

    Losing information is always sound, so a pair that cannot be combined gives `None`.

    Args:
        a (`Fact | None`):
            One fact; `None` is unconstrained.
        b (`Fact | None`):
            The other fact; `None` is unconstrained.

    Returns:
        `Fact | None`:
            The set intersection of two `Members`; `Members` of the one member an
            `Exact` names, else that `Exact`; an `Exact` both sides share; the other
            fact when one side is `None`; otherwise `None`.

    Example:
        `intersect_facts(Members((Buy, Sell)), Members((Sell, Hold)))` is
        `Members((Sell,))`; `intersect_facts(Exact(Buy), Members((Buy, Sell)))` is
        `Members((Buy,))`.
    """
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, Members) and isinstance(b, Members):
        kept = tuple(member for member in a.members if _contains(b.members, member))
        return Members(kept, a.nullable and b.nullable)
    if isinstance(a, Exact) and isinstance(b, Exact):
        return a if a.type == b.type else None
    exact, members = (a, b) if isinstance(a, Exact) else (b, a)
    if _contains(members.members, exact.type):
        return Members((exact.type,))
    return exact


def isinstance_fact(base_type: Type, typ: Type) -> Fact:
    """Return the fact that a value of `base_type` satisfies `typ`.

    Args:
        base_type (`Type`):
            The type of the value being tested.
        typ (`Type`):
            The type it satisfies. A nullable `typ` also admits `None`.

    Returns:
        `Fact`:
            `Members` when `typ` (ignoring nullability) equals a member of the base's
            union, or is a union of its members; otherwise `Exact(typ)`.

    Example:
        For `Union[Buy, Hold, Sell]`, `Buy` gives `Members((Buy,))` and an unrelated
        record gives `Exact` of that record.
    """
    strict = replace(typ, nullable=False)
    members = base_type.variants
    if members is not None:
        if _contains(members, strict):
            return Members((strict,), typ.nullable)
        if strict.variants is not None and all(
            _contains(members, member) for member in strict.variants
        ):
            return Members(strict.variants, typ.nullable)
    return Exact(typ)


def _union_envs(envs: Sequence[Env]) -> Env:
    """Join environments by per-base union; a base missing from any is dropped."""
    if not envs:
        return {}
    joined: Env = {}
    for base, fact in envs[0].items():
        for env in envs[1:]:
            fact = union_facts(fact, env.get(base))
            if fact is None:
                break
        if fact is not None:
            joined[base] = fact
    return joined


def _intersect_envs(envs: Sequence[Env]) -> Env:
    """Meet environments by per-base intersection."""
    met: Env = {}
    for env in envs:
        for base, fact in env.items():
            combined = intersect_facts(met.get(base), fact)
            if combined is None:
                met.pop(base, None)
            else:
                met[base] = combined
    return met


# --------------------------------------------------------------------------- #
# Contract views
# --------------------------------------------------------------------------- #


def _narrow_type(typ: Type, path: tuple[str, ...], narrowed: Type) -> Optional[Type]:
    """Replace the type at `path` through record fields, or `None` if unreachable."""
    if not path:
        return narrowed
    # A union is never entered: rewriting its members would break member identity.
    if typ.fields is None or typ.variants is not None or path[0] not in typ.fields:
        return None
    inner = _narrow_type(typ.fields[path[0]], path[1:], narrowed)
    if inner is None:
        return None
    return replace(typ, fields={**typ.fields, path[0]: inner})


def narrow_contract_at(
    contract: OutputContract,
    path: tuple[str, ...],
    typ: Type,
) -> Optional[OutputContract]:
    """Return `contract` with the type at `path` replaced by `typ`, keeping origins.

    The path is followed through record fields only. An exact contract keeps its
    origins unchanged, since narrowing does not move any path. Replacing a whole
    partial record gives an exact contract with each proven field's origins rebased
    under its name; narrowing a partial record's last opaque field gives the exact
    record.

    Args:
        contract (`OutputContract`):
            An exact or partial-record contract.
        path (`tuple[str, ...]`):
            Field names from the contract's root to the narrowed value. `()` is the root.
        typ (`Type`):
            The narrowed type.

    Returns:
        `OutputContract | None`:
            The narrowed contract, or `None` when `path` names an absent field, crosses a
            union or a non-record, or goes below an opaque field.

    Raises:
        `TypeError`:
            `contract` is neither a `ValueContract` nor a `PartialRecordContract`.

    Example:
        Narrowing `Union[Buy, Sell]` at `()` to `Buy` gives a `Buy` contract with the
        same origins.
    """
    if type(contract) is ValueContract:
        narrowed = _narrow_type(contract.type, path, typ)
        if narrowed is None:
            return None
        return ValueContract(narrowed, contract.origins)
    if type(contract) is not PartialRecordContract:
        raise TypeError(
            "output contract must be an exact ValueContract or "
            f"PartialRecordContract, got {type(contract).__name__}"
        )
    if not path:
        return ValueContract(typ, record_field_origins(contract.fields))
    fields = dict(contract.fields)
    name, rest = path[0], path[1:]
    if name not in fields:
        return None
    field_contract = fields[name]
    if field_contract is None:
        if rest:
            return None
        fields[name] = ValueContract(typ)
        return combine_output_contracts(fields)
    narrowed_field = narrow_contract_at(field_contract, rest, typ)
    if narrowed_field is None:
        return None
    fields[name] = narrowed_field
    return PartialRecordContract(tuple(fields.items()))


def _members_type(members: Sequence[Type], nullable: bool) -> Optional[Type]:
    """The type of a value that is one of `members`, or `None` if it may be `None`.

    Returns `None` for an empty set that is not nullable: no value fits.
    """
    if not members:
        return Type.scalar(ValueKind.NONE) if nullable else None
    if len(members) == 1:
        return replace(members[0], nullable=nullable)
    return Type(
        kind=ValueKind.OBJECT,
        variants=canonical_union_order(members),
        nullable=nullable,
    )


def _fact_type(fact: Fact) -> Optional[Type]:
    """The type a node sees for a base under `fact`, or `None` for no view.

    An empty member set on a nullable base can only be `None`; on a non-nullable base
    the node never runs, so it gets no view.
    """
    if isinstance(fact, Exact):
        return fact.type
    return _members_type(fact.members, fact.nullable)


def _split_base(base: str) -> Optional[tuple[str, str, tuple[str, ...]]]:
    """Split a base path into `("producer"|"input", root name, field path)`."""
    parts = tuple(base.split("."))
    if len(parts) >= 2 and parts[0] == "input":
        return "input", parts[1], parts[2:]
    if len(parts) >= 2 and parts[1] == "output":
        return "producer", parts[0], parts[2:]
    return None


def _base_contract(
    base: str,
    producers: Mapping[str, OutputContract],
    inputs: Mapping[str, ValueContract],
) -> Optional[OutputContract]:
    """The contract at a base path, or `None` when it is unknown or unproven."""
    split = _split_base(base)
    if split is None:
        return None
    space, name, path = split
    root = (producers if space == "producer" else inputs).get(name)
    if root is None:
        return None
    return project_output_contract(root, list(path))


def _build_views(
    env: Env,
    producers: dict[str, OutputContract],
    inputs: dict[str, ValueContract],
) -> tuple[dict[str, OutputContract], dict[str, ValueContract]]:
    """Rebuild each narrowed base in the contract maps, outer bases first."""
    producer_view = producers
    input_view = inputs
    split_bases = [(base, _split_base(base)) for base in env]
    ordered = sorted(
        ((base, split) for base, split in split_bases if split is not None),
        key=lambda item: (len(item[1][2]), item[0]),
    )
    for base, (space, name, path) in ordered:
        target = _fact_type(env[base])
        if target is None:
            continue
        view = producer_view if space == "producer" else input_view
        root = view.get(name)
        if root is None:
            continue
        current = project_output_contract(root, list(path))
        if type(current) is ValueContract and current.type == target:
            continue
        narrowed = narrow_contract_at(root, path, target)
        if narrowed is None:
            continue
        if space == "producer":
            producer_view = {**producer_view, name: narrowed}
        else:
            input_view = {**input_view, name: narrowed}
    return producer_view, input_view


# --------------------------------------------------------------------------- #
# CASE facts
# --------------------------------------------------------------------------- #


def _direct_ref(text: object) -> Optional[tuple[str, ...]]:
    """The dotted path of a template that is one plain `${a.b.c}` reference."""
    if not isinstance(text, str):
        return None
    try:
        segments = scan_template(text)
    except ExpressionError:
        return None
    if len(segments) != 1 or not isinstance(segments[0], Span):
        return None
    return direct_reference_parts(segments[0].tree)


def _union_type(contract: Optional[OutputContract]) -> Optional[Type]:
    """The union type of an exact contract, else `None`."""
    if type(contract) is ValueContract and contract.type.variants is not None:
        return contract.type
    return None


def _tags(union: Type, tag_field: str) -> list[str]:
    """The sorted tag values of a tagged union's tag field."""
    return sorted(tag for member in union.variants for tag in member.fields[tag_field].tags)


def _members_fact(
    reference: Optional[Type],
    members: Sequence[Type],
    nullable: bool,
) -> Fact:
    """The fact that a value is one of `members`, relative to the base's own union."""
    if not members:
        return Members((), nullable)
    typ = _members_type(members, nullable)
    if reference is None:
        # The base has no union of its own outside this branch, so the set is only an exact type.
        return Exact(typ)
    return isinstance_fact(reference, typ)


def _on_case_facts(
    desc: CaseDescriptor,
    producers_view: Mapping[str, OutputContract],
    inputs_view: Mapping[str, ValueContract],
    producers: Mapping[str, OutputContract],
    inputs: Mapping[str, ValueContract],
    diagnostics: LoadDiagnostics,
) -> dict[str, Env]:
    """The facts each handle of an `on:` CASE over a union tag field carries.

    Raises:
        `LoadError`:
            A `when:` value is not a tag of the union.
    """
    parts = _direct_ref(desc.on)
    if parts is None or len(parts) < 3:
        return {}
    base, tag_field = ".".join(parts[:-1]), parts[-1]
    if _split_base(base) is None:
        return {}
    # Members are taken from the base's own union, so facts from different CASEs on the same
    # base compare by member; the routing CASE's environment is met with them on each edge.
    reference = _union_type(_base_contract(base, producers, inputs))
    if reference is not None and tag_field not in union_tag_fields(reference):
        reference = None
    union = reference or _union_type(_base_contract(base, producers_view, inputs_view))
    if union is None or tag_field not in union_tag_fields(union):
        return {}

    tags = _tags(union, tag_field)
    for index, case in enumerate(desc.cases):
        value = case.get("when")
        if value is not None and str(value) not in tags:
            when_range = desc.when_ranges[index] if index < len(desc.when_ranges) else None
            raise diagnostics.error(
                f"when: {value} is not a tag of the union at "
                f"${{{'.'.join(parts)}}} (tags: {', '.join(tags)})",
                range=when_range or desc.range,
            )

    by_handle: dict[str, list[Type]] = {}
    listed: set[str] = set()
    for case in desc.cases:
        value, then = case.get("when"), case.get("then")
        if value is None or then is None:
            continue
        listed.add(str(value))
        by_handle.setdefault(then, []).extend(
            member for member in union.variants if str(value) in member.fields[tag_field].tags
        )
    facts = {
        handle: {base: _members_fact(reference, members, False)}
        for handle, members in by_handle.items()
    }
    if any(case.get("then") == DEFAULT_HANDLE for case in desc.cases):
        # A `then:` node named like the else handle shares its edges: routing to it, or
        # matching nothing, fires every edge with that handle, so it carries no fact.
        facts.pop(DEFAULT_HANDLE, None)
        return facts
    if desc.else_ is not None:
        unlisted = [
            member for member in union.variants if not member.fields[tag_field].tags & listed
        ]
        facts[DEFAULT_HANDLE] = {base: _members_fact(reference, unlisted, union.nullable)}
    return facts


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Narrowing:
    """Per-node narrowing results for one flow.

    Args:
        producers (`dict[str, OutputContract]`):
            The unnarrowed producer contracts.
        inputs (`dict[str, ValueContract]`):
            The unnarrowed flow-input contracts.
        environments (`dict[str, Env]`):
            Each ordered node's facts, keyed by base path.
        views (`dict[str, tuple[dict[str, OutputContract], dict[str, ValueContract]]]`):
            The narrowed contract maps of nodes that narrow something.
        unordered (`frozenset[str]`):
            Nodes a topological order could not reach, on or after a cycle. They
            get no facts.
    """

    producers: dict[str, OutputContract]
    inputs: dict[str, ValueContract]
    environments: dict[str, Env] = field(default_factory=dict)
    views: dict[
        str, tuple[dict[str, OutputContract], dict[str, ValueContract]]
    ] = field(default_factory=dict)
    unordered: frozenset[str] = frozenset()

    def for_node(
        self, node_id: str
    ) -> tuple[dict[str, OutputContract], dict[str, ValueContract]]:
        """Return the producer and flow-input contracts node `node_id` sees.

        Args:
            node_id (`str`):
                A node id. An unknown or unnarrowed node gets the unchanged maps.

        Returns:
            `tuple[dict[str, OutputContract], dict[str, ValueContract]]`:
                The producer contracts and the flow-input contracts.
        """
        return self.views.get(node_id, (self.producers, self.inputs))

    def environment(self, node_id: str) -> Env:
        """Return node `node_id`'s facts by base path; `{}` when it has none."""
        return self.environments.get(node_id, {})


def _known_edges(
    descriptors: Mapping[str, NodeDescriptor],
    edges: Sequence[Edge],
) -> list[Edge]:
    """The given edges plus each descriptor's ordering edges, between known nodes only."""
    known = list(edges)
    for node_id, desc in descriptors.items():
        for source in desc.depends_on:
            known.append(
                Edge(id="", from_=source, to=node_id, ordering=True, optional=False)
            )
        for source in desc.runs_after:
            known.append(
                Edge(id="", from_=source, to=node_id, ordering=True, optional=True)
            )
    return [
        edge for edge in known if edge.from_ in descriptors and edge.to in descriptors
    ]


def _topological_order(
    descriptors: Mapping[str, NodeDescriptor],
    edges: Sequence[Edge],
) -> list[str]:
    """Kahn's order over `edges`, in descriptor order among ready nodes."""
    indegree = {node_id: 0 for node_id in descriptors}
    successors: dict[str, list[str]] = {node_id: [] for node_id in descriptors}
    for edge in edges:
        indegree[edge.to] += 1
        successors[edge.from_].append(edge.to)
    ready = deque(node_id for node_id, count in indegree.items() if count == 0)
    order: list[str] = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        for successor in successors[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    return order


def compute_narrowing(
    descriptors: Mapping[str, NodeDescriptor],
    edges: Sequence[Edge],
    producers: dict[str, OutputContract],
    inputs: dict[str, ValueContract],
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> Narrowing:
    """Compute each node's narrowing facts and contract views.

    A node's environment is the intersection of its contributions:

    - its control edges, joined by union, each giving the routing CASE's environment
      met with that edge's facts (an edge without a fact for a base leaves it
      unconstrained);
    - each required data group, giving the union of its producers' environments;
      an optional group gives nothing;
    - each required ordering edge (`depends_on`), giving its source's environment.

    Edges and dependencies that name unknown nodes are ignored. Nodes on or after a
    cycle get no facts and are reported in `unordered`.

    Args:
        descriptors (`Mapping[str, NodeDescriptor]`):
            The flow's node descriptors by id. A `CaseDescriptor` with `on:` produces facts;
            `depends_on`/`runs_after` give ordering edges.
        edges (`Sequence[Edge]`):
            The data edges plus each CASE's own data and control edges. Control edges
            carry the branch target, or `default` for `else:`, as `source_handle`.
        producers (`dict[str, OutputContract]`):
            Each node's output contract by id.
        inputs (`dict[str, ValueContract]`):
            The flow-input contracts by name.
        diagnostics (`LoadDiagnostics`, *optional*, defaults to `None`):
            Locates load errors. `None` uses an unlocated default.

    Returns:
        `Narrowing`:
            The environments, the views of narrowed nodes, and the unordered nodes.

    Raises:
        `LoadError`:
            An `on:` CASE over a union tag field lists a `when:` value that is not a tag.

    Example:
        With `d` a `Union[Buy, Sell]` and `c` an `on: ${d.output.action}` CASE routing
        `buy` to `b`, `compute_narrowing(...).for_node("b")[0]["d"]` is a `Buy` contract.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    known = _known_edges(descriptors, edges)
    order = _topological_order(descriptors, known)
    incoming: dict[str, list[Edge]] = {node_id: [] for node_id in descriptors}
    for edge in known:
        incoming[edge.to].append(edge)

    environments: dict[str, Env] = {}
    views: dict[str, tuple[dict[str, OutputContract], dict[str, ValueContract]]] = {}
    handle_facts: dict[str, dict[str, Env]] = {}
    for node_id in order:
        control: list[Env] = []
        groups: dict[Optional[str], tuple[bool, list[Env]]] = {}
        contributions: list[Env] = []
        for edge in incoming[node_id]:
            source_env = environments.get(edge.from_, {})
            if edge.source_handle is not None:
                edge_facts = handle_facts.get(edge.from_, {}).get(edge.source_handle, {})
                control.append(_intersect_envs([source_env, edge_facts]))
            elif edge.ordering:
                if not edge.optional:
                    contributions.append(source_env)
            else:
                # Like the runtime join, a group's first edge decides whether it is optional.
                optional, members = groups.setdefault(edge.input_group, (edge.optional, []))
                members.append(source_env)
        if control:
            contributions.append(_union_envs(control))
        contributions.extend(
            _union_envs(members) for optional, members in groups.values() if not optional
        )
        env = _intersect_envs(contributions)
        if env:
            environments[node_id] = env
            view = _build_views(env, producers, inputs)
            if view[0] is not producers or view[1] is not inputs:
                views[node_id] = view
        desc = descriptors[node_id]
        if isinstance(desc, CaseDescriptor) and desc.on is not None:
            producers_view, inputs_view = views.get(node_id, (producers, inputs))
            handle_facts[node_id] = _on_case_facts(
                desc,
                producers_view,
                inputs_view,
                producers,
                inputs,
                diagnostics,
            )

    return Narrowing(
        producers=producers,
        inputs=inputs,
        environments=environments,
        views=views,
        unordered=frozenset(descriptors) - frozenset(order),
    )


__all__ = [
    "Env",
    "Exact",
    "Fact",
    "Members",
    "Narrowing",
    "compute_narrowing",
    "intersect_facts",
    "isinstance_fact",
    "narrow_contract_at",
    "union_facts",
]
