"""Union narrowing: what each node may assume about union values, from the CASEs that route it.

A narrowing fact says which part of a union a value at a base path can be. A base path is
`<node>.output[.field...]` or `input.<name>[.field...]`. Facts come from CASE branches: an
`on:` CASE over a tag field of a union gives each `when:` edge the member with that tag and the
`else:` edge the unlisted members. A searched CASE gives each case's edge the facts of its
top-level `isinstance(<base>, T)` conjuncts; its `else:` edge carries none.

Facts flow along the graph in topological order. A node's control edges join by union (it ran
because one of them fired), each required data group joins its producers by union, and every
contribution (control, required data groups, `depends_on`) meets by intersection. Each node
then gets a view of the producer and flow-input contracts with its narrowed bases rebuilt, so
the reference and type checks can read member fields inside a branch. A searched CASE's own
param whose every read follows an `isinstance` conjunct in the same `and` chain gets a view
narrowed by those checks too.

The pass reads built, desugared CASE descriptors and their edges, so it runs after
`desugar_case_call_targets` and the node builds.

The fact algebra (`Members`, `Exact`, their union and intersection) lives in
`helpers.narrowing_facts`; this module re-exports it.

Knows about: descriptors, compiled edges, output contracts, narrowing facts, and `typesys`
unions.
Never imports: runtime.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence, TypeAlias

from yian.compile.model import Edge
from yian.compose.helpers.narrowing_facts import (
    Env,
    Exact,
    Fact,
    Members,
    fact_type,
    intersect_envs,
    intersect_facts,
    isinstance_fact,
    members_type,
    union_envs,
    union_facts,
)
from yian.compose.helpers.output_contracts import (
    combine_output_contracts,
    direct_reference_parts,
    project_output_contract,
    record_field_origins,
)
from yian.compose.nodes.base import NodeDescriptor
from yian.diagnostics import SourceRange
from yian.compose.helpers.isinstance_types import resolve_isinstance_type
from yian.compose.nodes.case import (
    CaseDescriptor,
    IsinstanceCheck,
    isinstance_conjuncts,
    on_match_value,
    searched_param_guards,
)
from yian.expr import ExpressionError
from yian.expr.template import Span, scan_template
from yian.nodes.case import ELSE_HANDLE
from yian.nodes.contracts import (
    OutputContract,
    PartialRecordContract,
    ValueContract,
)
from yian.typesys import TypeCheckError
from yian.typesys.types import TypeRegistry
from yian.typesys.values import Type, union_tag_fields


Views: TypeAlias = tuple[Mapping[str, OutputContract], Mapping[str, ValueContract]]
"""The producer contracts and the flow-input contracts one node or param reads against."""


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
    producers: Mapping[str, OutputContract],
    inputs: Mapping[str, ValueContract],
) -> Views:
    """Rebuild each narrowed base in the contract maps, outer bases first."""
    producer_view = producers
    input_view = inputs
    split_bases = [(base, _split_base(base)) for base in env]
    ordered = sorted(
        ((base, split) for base, split in split_bases if split is not None),
        key=lambda item: (len(item[1][2]), item[0]),
    )
    for base, (space, name, path) in ordered:
        target = fact_type(env[base])
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
    if reference is None:
        # The base has no union of its own outside this branch, so the set is only an exact type.
        return Exact(members_type(members, nullable))
    return Members(tuple(members), nullable)


def _on_case_facts(
    desc: CaseDescriptor,
    producers_view: Mapping[str, OutputContract],
    inputs_view: Mapping[str, ValueContract],
    producers: Mapping[str, OutputContract],
    inputs: Mapping[str, ValueContract],
) -> dict[str, Env]:
    """The facts each handle of an `on:` CASE over a union tag field carries.

    A `when:` value that is not a tag gives its handle an empty member set;
    `check_case_when_values` reports it.
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
    by_handle: dict[str, list[Type]] = {}
    listed: set[str] = set()
    for case in desc.cases:
        # Match what the lowering compares: an unquoted date is its ISO text.
        value, then = on_match_value(case.get("when")), case.get("then")
        if value is None or then is None:
            continue
        members = by_handle.setdefault(then, [])
        if isinstance(value, str) and value in tags:
            listed.add(value)
            members.extend(m for m in union.variants if value in m.fields[tag_field].tags)
    facts = {
        handle: {base: _members_fact(reference, members, False)}
        for handle, members in by_handle.items()
    }
    if desc.else_ is not None:
        unlisted = [
            member for member in union.variants if not member.fields[tag_field].tags & listed
        ]
        facts[ELSE_HANDLE] = {base: _members_fact(reference, unlisted, union.nullable)}
    return facts


def _checks_env(
    checks: Sequence[IsinstanceCheck],
    producers: Mapping[str, OutputContract],
    inputs: Mapping[str, ValueContract],
    registry: TypeRegistry,
) -> Env:
    """The facts that hold when every `isinstance` check in `checks` passes.

    Each fact is taken against the base's unnarrowed type, so facts from different
    CASEs on the same base compare by member. A check whose base is not a node output or
    flow input, or whose type name does not resolve, gives no fact; the type-table pass
    reports an unknown name.
    """
    facts: list[Env] = []
    for base, type_name in checks:
        if _split_base(base) is None:
            continue
        try:
            typ = resolve_isinstance_type(type_name, registry)
        except TypeCheckError:
            continue
        contract = _base_contract(base, producers, inputs)
        if type(contract) is not ValueContract:
            facts.append({base: Exact(typ)})
            continue
        fact = isinstance_fact(contract.type, typ)
        if fact is not None:
            facts.append({base: fact})
    return intersect_envs(facts)


def _searched_case_facts(
    desc: CaseDescriptor,
    producers: Mapping[str, OutputContract],
    inputs: Mapping[str, ValueContract],
    registry: TypeRegistry,
) -> dict[str, Env]:
    """The facts each handle of a searched CASE carries.

    Cases routing to one target join by union. The `else:` handle carries no fact.
    """
    by_handle: dict[str, list[Env]] = {}
    for case in desc.cases:
        then = case.get("then")
        if then is None:
            continue
        checks = isinstance_conjuncts(case.get("when"))
        by_handle.setdefault(then, []).append(
            _checks_env(checks, producers, inputs, registry)
        )
    facts = {handle: union_envs(envs) for handle, envs in by_handle.items()}
    return {handle: env for handle, env in facts.items() if env}


def _guarded_param_views(
    desc: CaseDescriptor,
    env: Env,
    producers: Mapping[str, OutputContract],
    inputs: Mapping[str, ValueContract],
    registry: TypeRegistry,
) -> dict[str, Views]:
    """The views of a searched CASE's params that every read guards with `isinstance`.

    A param read under different checks gets the union of their facts, met with the
    CASE's own environment.
    """
    views: dict[str, Views] = {}
    for param, occurrences in searched_param_guards(desc).items():
        if not occurrences or not all(checks for _, checks in occurrences):
            continue
        guard = union_envs(
            [_checks_env(checks, producers, inputs, registry) for _, checks in occurrences]
        )
        if not guard:
            continue
        view = _build_views(intersect_envs([env, guard]), producers, inputs)
        if view[0] is not producers or view[1] is not inputs:
            views[param] = view
    return views


def _unguarded_read_sites(
    desc: CaseDescriptor,
) -> dict[str, tuple[str, Optional[SourceRange]]]:
    """Where to report a searched CASE param whose first read is guarded and a later one isn't.

    The guarded read is fine, so an error on the param belongs to its first unguarded
    read: the label and range of that read's `when:`.
    """
    sites: dict[str, tuple[str, Optional[SourceRange]]] = {}
    for param, occurrences in searched_param_guards(desc).items():
        if not occurrences or not occurrences[0][1]:
            continue
        unguarded = next((index for index, checks in occurrences if not checks), None)
        if unguarded is None:
            continue
        when_range = desc.when_ranges[unguarded] if unguarded < len(desc.when_ranges) else None
        sites[param] = (f"cases[{unguarded}].when", when_range)
    return sites


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Narrowing:
    """Per-node narrowing results for one flow.

    Args:
        producers (`Mapping[str, OutputContract]`):
            The unnarrowed producer contracts.
        inputs (`Mapping[str, ValueContract]`):
            The unnarrowed flow-input contracts.
        environments (`dict[str, Env]`):
            Each ordered node's facts, keyed by base path.
        views (`dict[str, Views]`):
            The narrowed contract maps of nodes that narrow something.
        param_views (`dict[str, dict[str, Views]]`):
            Per searched CASE, the narrowed contract maps of its params whose every read
            follows an `isinstance` conjunct.
        param_sites (`dict[str, dict[str, tuple[str, SourceRange | None]]]`):
            Per searched CASE, the label and range to report a param at when its first
            read is guarded and a later one isn't: the first unguarded read's `when:`.
        unordered (`frozenset[str]`):
            Nodes a topological order could not reach, on or after a cycle. They
            get no facts.
    """

    producers: Mapping[str, OutputContract]
    inputs: Mapping[str, ValueContract]
    environments: dict[str, Env] = field(default_factory=dict)
    views: dict[str, Views] = field(default_factory=dict)
    param_views: dict[str, dict[str, Views]] = field(default_factory=dict)
    param_sites: dict[str, dict[str, tuple[str, Optional[SourceRange]]]] = field(
        default_factory=dict
    )
    unordered: frozenset[str] = frozenset()

    def for_node(self, node_id: str) -> Views:
        """Return the producer and flow-input contracts node `node_id` sees.

        Args:
            node_id (`str`):
                A node id. An unknown or unnarrowed node gets the unchanged maps.

        Returns:
            `Views`:
                The producer contracts and the flow-input contracts.
        """
        return self.views.get(node_id, (self.producers, self.inputs))

    def for_param(self, node_id: str, param: str) -> Views:
        """Return the producer and flow-input contracts param `param` of `node_id` reads against.

        Args:
            node_id (`str`):
                A node id.
            param (`str`):
                One of the node's params. A searched CASE's `isinstance`-guarded param
                gets its guarded view; any other param gets the node's view.

        Returns:
            `Views`:
                The producer contracts and the flow-input contracts.
        """
        guarded = self.param_views.get(node_id, {}).get(param)
        return guarded if guarded is not None else self.for_node(node_id)

    def param_site(
        self, node_id: str, param: str
    ) -> Optional[tuple[str, Optional[SourceRange]]]:
        """Return the label and range to report param `param` of `node_id` at, if moved.

        Args:
            node_id (`str`):
                A node id.
            param (`str`):
                One of the node's params.

        Returns:
            `tuple[str, SourceRange | None] | None`:
                The `cases[N].when` label and range of the param's first unguarded read
                when its first read is guarded and a later one isn't; otherwise `None`,
                and the param keeps the site of its first read.
        """
        return self.param_sites.get(node_id, {}).get(param)

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
    producers: Mapping[str, OutputContract],
    inputs: Mapping[str, ValueContract],
    *,
    registry: TypeRegistry | None = None,
) -> Narrowing:
    """Compute each node's narrowing facts and contract views.

    A node's environment is the intersection of its contributions:

    - its control edges, joined by union, each giving the routing CASE's environment
      met with that edge's facts (an edge without a fact for a base leaves it
      unconstrained);
    - each required data group, giving the union of its producers' environments;
      an optional group gives nothing;
    - each required ordering edge (`depends_on`), giving its source's environment.

    A searched CASE's edge carries the facts of its case's top-level `isinstance`
    conjuncts, and its `else:` edge none.

    Edges and dependencies that name unknown nodes are ignored. Nodes on or after a
    cycle get no facts and are reported in `unordered`.

    Args:
        descriptors (`Mapping[str, NodeDescriptor]`):
            The flow's node descriptors by id. A `CaseDescriptor` with `on:` produces facts;
            `depends_on`/`runs_after` give ordering edges.
        edges (`Sequence[Edge]`):
            The data edges plus each CASE's own data and control edges. Control edges
            carry the branch target, or the else handle for `else:`, as `source_handle`.
        producers (`Mapping[str, OutputContract]`):
            Each node's output contract by id.
        inputs (`Mapping[str, ValueContract]`):
            The flow-input contracts by name.
        registry (`TypeRegistry`, *optional*, defaults to `None`):
            The flow's `typedefs:` registry, for `isinstance` type names. `None` knows
            only the builtin names.

    Returns:
        `Narrowing`:
            The environments, the views of narrowed nodes, and the unordered nodes.

    Example:
        With `d` a `Union[Buy, Sell]` and `c` an `on: ${d.output.action}` CASE routing
        `buy` to `b`, `compute_narrowing(...).for_node("b")[0]["d"]` is a `Buy` contract.
    """
    registry = registry or {}
    known = _known_edges(descriptors, edges)
    order = _topological_order(descriptors, known)
    incoming: dict[str, list[Edge]] = {node_id: [] for node_id in descriptors}
    for edge in known:
        incoming[edge.to].append(edge)

    environments: dict[str, Env] = {}
    views: dict[str, Views] = {}
    param_views: dict[str, dict[str, Views]] = {}
    param_sites: dict[str, dict[str, tuple[str, Optional[SourceRange]]]] = {}
    handle_facts: dict[str, dict[str, Env]] = {}
    for node_id in order:
        control: list[Env] = []
        groups: dict[Optional[str], tuple[bool, list[Env]]] = {}
        contributions: list[Env] = []
        for edge in incoming[node_id]:
            source_env = environments.get(edge.from_, {})
            if edge.source_handle is not None:
                edge_facts = handle_facts.get(edge.from_, {}).get(edge.source_handle, {})
                control.append(intersect_envs([source_env, edge_facts]))
            elif edge.ordering:
                if not edge.optional:
                    contributions.append(source_env)
            else:
                # Like the runtime join, a group's first edge decides whether it is optional.
                optional, members = groups.setdefault(edge.input_group, (edge.optional, []))
                members.append(source_env)
        if control:
            contributions.append(union_envs(control))
        contributions.extend(
            union_envs(members) for optional, members in groups.values() if not optional
        )
        env = intersect_envs(contributions)
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
            )
        elif isinstance(desc, CaseDescriptor):
            handle_facts[node_id] = _searched_case_facts(desc, producers, inputs, registry)
            guarded = _guarded_param_views(desc, env, producers, inputs, registry)
            if guarded:
                param_views[node_id] = guarded
            sites = _unguarded_read_sites(desc)
            if sites:
                param_sites[node_id] = sites

    return Narrowing(
        producers=producers,
        inputs=inputs,
        environments=environments,
        views=views,
        param_views=param_views,
        param_sites=param_sites,
        unordered=frozenset(descriptors) - frozenset(order),
    )


__all__ = [
    "Env",
    "Exact",
    "Fact",
    "Members",
    "Narrowing",
    "Views",
    "compute_narrowing",
    "intersect_facts",
    "isinstance_fact",
    "narrow_contract_at",
    "union_facts",
]
