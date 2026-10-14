"""Graph-stage: a lifted node takes its host's CASE routes, ordering, and data gates.

A desugar pass that lifts a node out of an authored node records that node as the lifted
node's `host` (an inline `call(...)` in a node's binding, an adaptive gate's composer, the
inner calls of a nested branch `call(...)`). The lifted node runs only on its host's routes and
after its host's ordering: it gets a copy of every CASE route edge into the host (same
`source_handle`, the else handle included), and the ordering edges the host's `depends_on:`
(co-skip) and `runs_after:` (order only) declare. A host that is itself lifted passes on what it
inherits, so chains of lifts are followed.

The lifted node also waits for the host's other data sources and is skipped exactly when the
host's co-skip rule would skip the host. For each data input group of the host it gets *data
gates*: ordinary data edges from that group's producers, all in one input group named
`<node>.<group>` (the node whose input it is, then that input's group). Grouped like the host's
input, the runtime join applies the host's rule to them: a required group is dead once all its
producers are skipped, a defaulted one (`${x.output :- d}`) only waits. They are data-shaped
because a coalesce (`${a.output | b.output}`, skipped only when every producer is) cannot be
written as ordering edges, which co-skip one by one. The lifted node binds its inputs from
wiring, so a gate never feeds a value.

A producer is *family* when it is the lifted node, one of its hosts, or a node lifted out of one
of those hosts. A group whose only producer is family gives no gate: siblings lifted from the
same host don't wait on each other. A required group read from one family node is instead looked
through: the lifted node gets that node's own data gates (unless the lifted node itself requires
that node as an input, so its inputs already gate the lifted node). Each ancestor host is
visited the same way, so transitive lifts gate on every host up the chain.

A gate group is dropped as redundant when the lifted node already depends on all its producers
and either the gate is optional or one of the lifted node's required groups lies inside it, so
those producers already skip the lifted node.

One case is not an exact match: a family producer in a defaulted or multi-producer group, which
only happens when an author names a lifted node's id (`${__call_0.output | t.output}`). Its other
producers then only make the lifted node wait, so it may run while its host is skipped; it never
skips when the host would run.

An unknown `depends_on`/`runs_after` source is skipped here; `infer_ordering_edges` reports it
at the host, which is the authored node. An unknown data producer gives no gate either.

Imports flow down/peer only: `compile.model` (Edge), `compose.parse.file` (NodeDescriptor).
Only the loader pipeline imports this back.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Optional

from yian.compile.model import Edge
from yian.compose.parse.file import NodeDescriptor


class _Gate(NamedTuple):
    """One inherited edge before it gets a target and an id."""

    source: str
    handle: Optional[str]
    ordering: bool
    optional: bool
    group: Optional[str] = None


def host_control_edges(
    descriptors: Mapping[str, NodeDescriptor],
    edges: Sequence[Edge],
) -> list[Edge]:
    """Return the route, ordering, and data-gate edges each lifted node inherits from its host.

    For each node with a `host`, in descriptor order, emits a copy of every edge into the host
    that has a `source_handle` (a CASE route), then one ordering edge per host `depends_on`
    source (`optional=False`) and per `runs_after` source (`optional=True`), then whatever the
    host inherits from its own host, then the node's data gates: one data edge per producer of
    each host input group it must wait on, in input group `<node>.<group>` with that group's
    `optional`, visiting the host, then its hosts and any looked-through family node, in that
    order. A source that is not a node, or is the lifted node itself, is skipped. Route copies
    and data gates are named `<source>-><node>#<i>` and ordering edges `<source>~><node>#<i>`,
    with `i` the first index no edge in ``edges`` or already emitted uses.

    Args:
        descriptors (`Mapping[str, NodeDescriptor]`):
            The flow level's desugared descriptors by node id.
        edges (`Sequence[Edge]`):
            The edges built so far (data edges and each CASE's own edges). Route and data
            edges are read from it; every id in it is taken.

    Returns:
        `list[Edge]`:
            The inherited edges, new ones only.

    Raises:
        `ValueError`:
            Raised when a host is not a node or hosts form a cycle, or when a lifted node has
            a route edge, `depends_on`, or `runs_after` of its own. Desugar never builds
            these, so each is a compiler bug.

    Example:
        With a CASE `c` routing `then: publish` and `publish` reading `call(side)` lifted as
        `__call_0` (host `publish`) and `${maybe.output}` as `notes`, the result holds
        `Edge(id="c->__call_0#0", from_="c", to="__call_0", source_handle="publish")` and
        `Edge(id="maybe->__call_0#0", from_="maybe", to="__call_0", input_group="publish.notes")`.
    """
    routes_into: dict[str, list[Edge]] = {}
    data_into: dict[str, list[Edge]] = {}
    for edge in edges:
        if edge.source_handle is not None:
            routes_into.setdefault(edge.to, []).append(edge)
        elif not edge.ordering:
            data_into.setdefault(edge.to, []).append(edge)

    memo: dict[str, list[_Gate]] = {}

    def checked_host(node_id: str, host: str) -> str:
        """``host``, checked to be a node of this level."""
        if host not in descriptors:
            raise ValueError(
                f"lifted node {node_id!r} names host {host!r}, which is not a node"
            )
        return host

    def gating(node_id: str, chain: tuple[str, ...]) -> list[_Gate]:
        """The route and ordering gates of ``node_id``, inherited ones included."""
        if node_id in memo:
            return memo[node_id]
        if node_id in chain:
            raise ValueError(f"lifted-node hosts form a cycle: {' -> '.join((*chain, node_id))}")
        desc = descriptors[node_id]
        gates = [
            _Gate(edge.from_, edge.source_handle, False, edge.optional)
            for edge in routes_into.get(node_id, [])
        ]
        gates += [_Gate(src, None, True, False) for src in desc.depends_on if src in descriptors]
        gates += [_Gate(src, None, True, True) for src in desc.runs_after if src in descriptors]
        if desc.host is not None:
            gates += gating(checked_host(node_id, desc.host), (*chain, node_id))
        memo[node_id] = gates
        return gates

    def input_groups(node_id: str) -> list[list[Edge]]:
        """``node_id``'s incoming data edges grouped by input group, in edge order."""
        groups: dict[Optional[str], list[Edge]] = {}
        for edge in data_into.get(node_id, []):
            groups.setdefault(edge.input_group, []).append(edge)
        return list(groups.values())

    chains: dict[str, tuple[str, ...]] = {}

    def ancestors(node_id: str) -> tuple[str, ...]:
        """``node_id``'s hosts, nearest first."""
        if node_id not in chains:
            chain: list[str] = []
            current = node_id
            while (host := descriptors[current].host) is not None:
                host = checked_host(current, host)
                if host == node_id or host in chain:
                    raise ValueError(
                        f"lifted-node hosts form a cycle: {' -> '.join((node_id, *chain, host))}"
                    )
                chain.append(host)
                current = host
            chains[node_id] = tuple(chain)
        return chains[node_id]

    def data_gates(node_id: str) -> list[_Gate]:
        """The data gates of lifted node ``node_id``, in visiting order."""
        chain = ancestors(node_id)
        hosts = set(chain)
        seen = {node_id, *chain}
        queue = deque(chain)
        gates: list[_Gate] = []
        sources: set[str] = set()
        required: list[frozenset[str]] = []
        for group in input_groups(node_id):
            sources.update(edge.from_ for edge in group)
            if not group[0].optional:
                required.append(frozenset(edge.from_ for edge in group))

        def family(source: str) -> bool:
            """Whether ``source`` is this node, a host of it, or lifted out of one of those."""
            if source in seen:
                return True
            desc = descriptors.get(source)
            return desc is not None and desc.host is not None and not hosts.isdisjoint(
                ancestors(source)
            )

        def add(producers: list[str], group: str, optional: bool) -> None:
            """Gate on ``producers`` as one input group, unless that is redundant."""
            members = frozenset(producers)
            if not members:
                return
            # The lifted node already waits on every producer, and either this gate only
            # waits or a required group of its own already skips it when this one would.
            if members <= sources and (
                optional or any(own <= members for own in required)
            ):
                return
            sources.update(members)
            if not optional:
                required.append(members)
            gates.extend(_Gate(source, None, False, optional, group) for source in producers)

        while queue:
            current = queue.popleft()
            for group in input_groups(current):
                name = f"{current}.{group[0].input_group}"
                optional = group[0].optional
                known = [edge.from_ for edge in group if edge.from_ in descriptors]
                kin = [source for source in known if family(source)]
                if not kin:
                    add(known, name, optional)
                elif len(group) == 1 and not optional:
                    source = group[0].from_
                    if source not in seen:
                        seen.add(source)
                        if frozenset({source}) not in required:
                            queue.append(source)
                else:
                    add([source for source in known if source not in kin], name, True)
        return gates

    taken = {edge.id for edge in edges}
    inherited: list[Edge] = []
    for node_id, desc in descriptors.items():
        if desc.host is None:
            continue
        # Control edges join by OR, so an own route would loosen the inherited gate, and own
        # ordering fields would get ids from `infer_ordering_edges` that may collide with these.
        if routes_into.get(node_id) or desc.depends_on or desc.runs_after:
            raise ValueError(
                f"lifted node {node_id!r} has its own route or ordering edges; "
                "it may only inherit its host's"
            )
        gates = gating(checked_host(node_id, desc.host), (node_id,)) + data_gates(node_id)
        for gate in gates:
            if gate.source == node_id:
                continue
            arrow = "~>" if gate.ordering else "->"
            index = 0
            while f"{gate.source}{arrow}{node_id}#{index}" in taken:
                index += 1
            edge_id = f"{gate.source}{arrow}{node_id}#{index}"
            taken.add(edge_id)
            inherited.append(
                Edge(
                    id=edge_id,
                    from_=gate.source,
                    to=node_id,
                    source_handle=gate.handle,
                    ordering=gate.ordering,
                    optional=gate.optional,
                    input_group=gate.group,
                )
            )
    return inherited
