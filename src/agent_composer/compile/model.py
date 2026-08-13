"""The compiled flow graph: nodes, edges, adjacency, and the node-state vocabulary.

`CompiledFlow` is immutable topology — the nodes and how they connect. All mutable run
state (which nodes/edges were taken or skipped) lives in the runtime's state manager, so one
`CompiledFlow` can back many independent runs. `NodeState` is the per-node state vocabulary
the runtime tracks against that topology.

`START_ID` / `END_ID` are the reserved boundary-node ids. Every flow begins with a synthesized
START node (which binds the run inputs) and ends with a synthesized END node (whose value is the
flow's result). They are owned canonically by the node classes (`StartNode.ID` / `EndNode.ID`)
and re-exported here so runtime consumers can import them alongside the graph types.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from agent_composer.nodes.base import Node
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode

# The synthesized boundary-node ids. Canonical source is the node class; re-exported here.
START_ID = StartNode.ID
END_ID = EndNode.ID

__all__ = ["START_ID", "END_ID", "NodeState", "Edge", "FlowOutput", "Flow", "CompiledFlow"]


class NodeState(str, Enum):
    """Per-node run state, tracked by the runtime's state manager (never on the immutable graph).

    UNKNOWN  — not yet decided; the initial state of every node.
    TAKEN    — the node ran and its outputs are committed to the pool.
    SKIPPED  — the node will not run (a branch not taken, or skip-flooded from a skipped input).
    EXPANDED — a spawner node (CALL/MAP/AGENT/LOOP) that ran and returned a `Grow`: it grew the
               graph, and its own value is committed by the subgraph terminal via `commit_as`.
    """

    UNKNOWN = "unknown"
    TAKEN = "taken"
    SKIPPED = "skipped"
    EXPANDED = "expanded"


@dataclass(frozen=True)
class Edge:
    """A directed edge `from_ -> to` in the compiled graph.

    Most edges carry data (one producer output feeding one consumer input group); some carry
    only branch-control or run-ordering and no data.
    """

    id: str                               # unique edge id within the flow
    from_: str                            # producer node id
    to: str                               # consumer node id
    source_handle: Optional[str] = None   # branch label on an edge leaving an CASE node; None otherwise
    input_group: Optional[str] = None     # the consumer input this data edge feeds; None for pure control/ordering edges
    optional: bool = False                # if True, a skipped producer does NOT co-skip the consumer
                                          # (a data group with a literal / `:?` default, or an ordering edge)
    ordering: bool = False                # if True, carries no data — a `depends_on` / `runs_after` ordering edge


@dataclass(frozen=True)
class FlowOutput:
    """One declared flow output: its `name` and where its value comes from (`from_`).

    `from_` is either a `${...}` binding string (e.g. `${node.key}`) or a literal value.
    """

    name: str
    from_: Any


class Flow:
    """The shared topology core every flow carries: nodes, edges, wiring, and its boundary ids.

    This is the one type that both the top-level compiled flow (`CompiledFlow`) and every spliced
    subgraph share. A flow *is a function*: it has a single entry (`start_id`) and a single result
    node (`end_id`), and its body is the `nodes`/`edges`/`wiring` triple. Spliced subgraphs pass
    their namespaced boundary ids; the top-level case defaults to the reserved `START_ID`/`END_ID`.

    Topology only — a `Flow` holds no run state, so one instance can back many independent runs.

    Attributes:
        nodes (`dict[str, Node]`):
            Map of node id to its `Node`. Includes the synthesized `START`/`END` boundary
            nodes plus every authored node.
        edges (`list[Edge]`):
            All directed edges. Carries data edges (a producer output feeding a consumer
            input group) and pure control/ordering edges alike.
        wiring (`dict[str, dict[str, Any]]`):
            Authoritative input wiring `wiring[node_id][param] -> source`, where source is a
            `${...}` binding string or a literal. `edges` is its derived data-flow projection.
        start_id (`str`):
            The flow's single entry node id — the node the engine seeds and advances at start.
            Defaults to `START_ID` (the top-level synthesized START); spliced flows namespace it.
        end_id (`str`):
            The flow's result node id — the node whose value is the flow's output. Defaults to
            `END_ID` (the top-level synthesized END); spliced flows namespace it.
    """

    def __init__(self, nodes: dict[str, Node], edges: list[Edge],
                 wiring: Optional[dict[str, dict[str, Any]]] = None,
                 start_id: str = START_ID, end_id: str = END_ID) -> None:
        self.nodes = nodes                              # node id -> Node
        self.edges = edges                              # all directed edges
        # Authoritative input wiring: wiring[node_id][param_name] -> source, where source is a
        # `${...}` binding string or a literal. `edges` is the derived data-flow projection of this.
        self.wiring: dict[str, dict[str, Any]] = dict(wiring or {})
        self.start_id = start_id                        # single entry node id
        self.end_id = end_id                            # result node id


class CompiledFlow(Flow):
    """
    The immutable compiled graph: nodes, edges, adjacency, and flow-owned input wiring.

    Topology only — it holds no run state, so one instance can back many independent runs.
    All mutable per-run state (which nodes/edges were taken or skipped) lives in the
    runtime's state manager. The sole mutation is [`add_subgraph`][agent_composer.CompiledFlow.add_subgraph],
    the runtime's append-only overlay used when a spawner node expands the graph at run time.

    Extends the internal `Flow` topology carrier with the runtime-only concerns below;
    the shared topology core (`nodes`/`edges`/`wiring`/`start_id`/`end_id`) lives on the base.

    Attributes:
        outputs (`list[FlowOutput]`):
            The declared `outputs:`; empty means the flow falls back to the raw terminal
            node value.
        flow_llm_config (`dict[str, Any]`):
            The flow's authored `llm_config:` model-selection defaults (the cascade's flow
            layer); `{}` when absent. `resolve_llm_cascade` gap-fills each agent's effective
            config from this, the parent chain, the CLI layer, and explicit process defaults
            captured once at run initialization. It returns the complete static-path map.
    """

    def __init__(self, nodes: dict[str, Node], edges: list[Edge],
                 outputs: Optional[list[FlowOutput]] = None,
                 wiring: Optional[dict[str, dict[str, Any]]] = None,
                 flow_llm_config: Optional[dict] = None) -> None:
        # The top-level compiled flow always roots at the synthesized START and returns END.
        super().__init__(nodes=nodes, edges=edges, wiring=wiring,
                         start_id=START_ID, end_id=END_ID)
        self.outputs = outputs or []                    # declared outputs; empty -> fall back to the raw terminal value
        # The flow's authored llm_config: defaults — the cascade's flow layer; {} when absent.
        self.flow_llm_config: dict = flow_llm_config or {}
        # Adjacency indexes, built once from `edges`.
        self._incoming: dict[str, list[Edge]] = {}      # node id -> edges ending at it
        self._outgoing: dict[str, list[Edge]] = {}      # node id -> edges leaving it
        for edge in edges:
            self._outgoing.setdefault(edge.from_, []).append(edge)
            self._incoming.setdefault(edge.to, []).append(edge)

    def incoming(self, node_id: str) -> list[Edge]:
        """Edges ending at `node_id` (its inputs); empty list if none."""
        return self._incoming.get(node_id, [])

    def outgoing(self, node_id: str) -> list[Edge]:
        """Edges leaving `node_id` (its outputs); empty list if none."""
        return self._outgoing.get(node_id, [])

    @property
    def terminal_id(self) -> Optional[str]:
        """The return-boundary node id (END_ID) when the synthesized END node exists — which is
        every loaded or built flow. None only for a hand-built graph that omits an END node."""
        return END_ID if END_ID in self.nodes else None

    @classmethod
    def from_parts(cls, nodes: dict[str, Node], edges: list[Edge],
                   outputs: Optional[list[FlowOutput]] = None,
                   wiring: Optional[dict[str, dict[str, Any]]] = None,
                   flow_llm_config: Optional[dict] = None) -> "CompiledFlow":
        """Build a CompiledFlow. The entry is always the synthesized START node (StartNode.ID),
        added implicitly by the loader / graph builder — never authored by the user; the engine
        seeds and advances it at run start."""
        return cls(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring,
                   flow_llm_config=flow_llm_config)

    def add_subgraph(self, nodes: dict[str, Node], edges: list[Edge],
                     wiring: dict[str, dict[str, Any]]) -> None:
        """Append a (deep-namespaced) subgraph onto the live topology — the one allowed mutation.

        Used by the runtime to grow the graph when a spawner node (CALL/MAP) expands: it adds the
        child's nodes, edges, and wiring and updates adjacency. The new subgraph's entry nodes are
        scheduled by the dispatcher, not here."""
        self.nodes.update(nodes)
        self.edges.extend(edges)
        for nid, w in wiring.items():
            self.wiring[nid] = dict(w)
        for edge in edges:
            self._outgoing.setdefault(edge.from_, []).append(edge)
            self._incoming.setdefault(edge.to, []).append(edge)

    def remove_subgraph(self, node_ids: set[str]) -> None:
        """Drop a finished (deep-namespaced) sub-namespace from the live topology — the inverse
        of add_subgraph, used ONLY to prune a fully-committed loop iteration whose ids are never
        referenced again. Removes the nodes, their incident edges, wiring, and adjacency.

        Leaves the CompiledFlow internally consistent: no surviving edge or adjacency entry keys
        on or points at a removed id, matching what the graph would be had those nodes never been
        added."""
        for nid in node_ids:
            self.nodes.pop(nid, None)
            self.wiring.pop(nid, None)
        # Drop every edge incident to a removed node (as producer or consumer).
        self.edges = [e for e in self.edges
                      if e.from_ not in node_ids and e.to not in node_ids]
        # Rebuild adjacency: drop keys for removed ids, and prune surviving lists of any edge
        # that touches a removed id.
        self._outgoing = {k: [e for e in v if e.to not in node_ids]
                          for k, v in self._outgoing.items() if k not in node_ids}
        self._incoming = {k: [e for e in v if e.from_ not in node_ids]
                          for k, v in self._incoming.items() if k not in node_ids}
