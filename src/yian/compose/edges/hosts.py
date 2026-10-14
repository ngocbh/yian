"""Graph-stage: a lifted node takes its host's CASE routes and ordering edges.

A desugar pass that lifts a node out of an authored node records that node as the lifted
node's `host` (an inline `call(...)` in a node's binding, an adaptive gate's composer, the
inner calls of a nested branch `call(...)`). The lifted node runs only on its host's routes and
after its host's ordering: it gets a copy of every CASE route edge into the host (same
`source_handle`, the else handle included), and the ordering edges the host's `depends_on:`
(co-skip) and `runs_after:` (order only) declare. A host that is itself lifted passes on what it
inherits, so chains of lifts are followed.

The host's data inputs are not copied: a lifted node does not wait on its host's other inputs.

An unknown `depends_on`/`runs_after` source is skipped here; `infer_ordering_edges` reports it
at the host, which is the authored node.

Imports flow down/peer only: `compile.model` (Edge), `compose.parse.file` (NodeDescriptor).
Only the loader pipeline imports this back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from yian.compile.model import Edge
from yian.compose.parse.file import NodeDescriptor

# One inherited edge before it gets a target and an id: (source, handle, ordering, optional).
_Template = tuple[str, "str | None", bool, bool]


def host_control_edges(
    descriptors: Mapping[str, NodeDescriptor],
    edges: Sequence[Edge],
) -> list[Edge]:
    """Return the route and ordering edges each lifted node inherits from its host.

    For each node with a `host`, in descriptor order, emits a copy of every edge into the host
    that has a `source_handle` (a CASE route), then one ordering edge per host `depends_on`
    source (`optional=False`) and per `runs_after` source (`optional=True`), then whatever the
    host inherits from its own host. A source that is not a node, or is the lifted node itself,
    is skipped. Route copies are named `<source>-><node>#<i>` and ordering edges
    `<source>~><node>#<i>`, with `i` the first index no edge in ``edges`` or already emitted
    uses.

    Args:
        descriptors (`Mapping[str, NodeDescriptor]`):
            The flow level's desugared descriptors by node id.
        edges (`Sequence[Edge]`):
            The edges built so far (data edges and each CASE's own edges). Route edges are
            read from it; every id in it is taken.

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
        `__call_0` (host `publish`), the result holds
        `Edge(id="c->__call_0#0", from_="c", to="__call_0", source_handle="publish")`.
    """
    routes_into: dict[str, list[Edge]] = {}
    for edge in edges:
        if edge.source_handle is not None:
            routes_into.setdefault(edge.to, []).append(edge)

    memo: dict[str, list[_Template]] = {}

    def host_of(node_id: str, desc: NodeDescriptor) -> str:
        """The host id of a lifted node, checked to be a node of this level."""
        assert desc.host is not None
        if desc.host not in descriptors:
            raise ValueError(f"lifted node {node_id!r} names host {desc.host!r}, which is not a node")
        return desc.host

    def gating(node_id: str, chain: tuple[str, ...]) -> list[_Template]:
        """The route and ordering templates that gate ``node_id``, inherited ones included."""
        if node_id in memo:
            return memo[node_id]
        if node_id in chain:
            raise ValueError(f"lifted-node hosts form a cycle: {' -> '.join((*chain, node_id))}")
        desc = descriptors[node_id]
        templates: list[_Template] = [
            (edge.from_, edge.source_handle, False, edge.optional)
            for edge in routes_into.get(node_id, [])
        ]
        templates += [(src, None, True, False) for src in desc.depends_on if src in descriptors]
        templates += [(src, None, True, True) for src in desc.runs_after if src in descriptors]
        if desc.host is not None:
            templates += gating(host_of(node_id, desc), (*chain, node_id))
        memo[node_id] = templates
        return templates

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
        for source, handle, ordering, optional in gating(host_of(node_id, desc), (node_id,)):
            if source == node_id:
                continue
            arrow = "~>" if ordering else "->"
            index = 0
            while f"{source}{arrow}{node_id}#{index}" in taken:
                index += 1
            edge_id = f"{source}{arrow}{node_id}#{index}"
            taken.add(edge_id)
            inherited.append(
                Edge(
                    id=edge_id,
                    from_=source,
                    to=node_id,
                    source_handle=handle,
                    ordering=ordering,
                    optional=optional,
                )
            )
    return inherited
