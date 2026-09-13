"""The compiled flow graph: nodes, edges, adjacency, and growth validation.

`CompiledFlow` carries authored topology — the nodes and how they connect. Mutable
dispositions live in ``RunState``, and the run boundary deep-copies the graph before a
growth reducer splices or prunes dynamic topology, so one authored flow can back many
independent runs.

Growth fragments also carry source-neutral seed connections and actual child
boundary pairs. Their graph validation lives here so manifests and accepted-state
preparation share one check without depending on the kind-aware growth builders.

`START_ID` / `END_ID` are the reserved boundary-node ids. Every flow begins with a synthesized
START node (which binds the run inputs) and ends with a synthesized END node (whose value is the
flow's result). They are owned canonically by the node classes (`StartNode.ID` / `EndNode.ID`)
and re-exported here so runtime consumers can import them alongside the graph types.
"""

from dataclasses import dataclass
from typing import Any, Optional

from agent_composer.memory.contracts import MemoryPolicySpec
from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import (
    Binding,
    ChildFlowBoundary,
    ExecutionScope,
    Parameter,
    SeedConnection,
    as_binding,
)
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode

# The synthesized boundary-node ids. Canonical source is the node class; re-exported here.
START_ID = StartNode.ID
END_ID = EndNode.ID

__all__ = [
    "START_ID",
    "END_ID",
    "GrowthRejected",
    "Edge",
    "FlowOutput",
    "Flow",
    "CompiledFlow",
]


class GrowthRejected(Exception):
    """Mark expected node-produced growth data rejected before durability.

    Compile and runtime raise this contract only around explicit growth-shape or
    growth-policy rejection. Arbitrary builder, compiler, and host exceptions stay
    outside it so the dispatcher can preserve their identity.

    Attributes:
        original (`Exception`): The exact typed rejection used for diagnostic data.
    """

    def __init__(self, original: Exception) -> None:
        """Retain the exact growth rejection for semantic failure construction."""
        if not isinstance(original, Exception):
            raise TypeError("GrowthRejected original must be an Exception")
        super().__init__(str(original))
        self.original = original


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
    """One declared result name paired with its canonical semantic binding.

    Raw `${...}` sources and literals are accepted at construction and normalized
    to an origin-free `Binding`. The temporary read-only `from_` view exposes the
    source for callers migrating to the carrier-owned representation.

    Attributes:
        name (`str`): Declared result name.
        binding (`Binding`): Source or literal plus its optional authored origin.
    """

    name: str
    binding: Binding

    def __post_init__(self) -> None:
        """Normalize a raw second positional value without replacing a carrier."""
        object.__setattr__(self, "binding", as_binding(self.binding))

    @property
    def from_(self) -> Any:
        """Return the source-only compatibility view of this declared output."""
        return self.binding.source


def _normalize_wiring(
    wiring: Optional[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Binding]]:
    """Copy flow-owned wiring into its canonical carrier representation.

    Args:
        wiring (`dict[str, dict[str, Any]] | None`):
            Raw or already-normalized sources keyed first by node id and then
            parameter name. `None` represents empty wiring.

    Returns:
        `dict[str, dict[str, Binding]]`:
            A fresh two-level mapping whose direct values are bindings.

    Raises:
        `TypeError`:
            Raised when a node id or parameter name is not a string.
    """
    if wiring is None:
        return {}

    normalized: dict[str, dict[str, Binding]] = {}
    for node_id, node_wiring in wiring.items():
        if not isinstance(node_id, str):
            raise TypeError(
                "flow wiring node id must be a string, "
                f"got {type(node_id).__name__}"
            )
        normalized_node: dict[str, Binding] = {}
        for parameter, source in node_wiring.items():
            if not isinstance(parameter, str):
                raise TypeError(
                    f"flow wiring parameter name for node {node_id!r} must be a string, "
                    f"got {type(parameter).__name__}"
                )
            normalized_node[parameter] = as_binding(source)
        normalized[node_id] = normalized_node
    return normalized


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
        wiring (`dict[str, dict[str, Binding]]`):
            Authoritative input wiring `wiring[node_id][param] -> Binding`. Each
            carrier owns a `${...}` source or literal plus its optional origin;
            `edges` is the derived data-flow projection.
        start_id (`str`):
            The flow's single entry node id — the node the engine seeds and advances at start.
            Defaults to `START_ID` (the top-level synthesized START); spliced flows namespace it.
        end_id (`str`):
            The flow's result node id — the node whose value is the flow's output. Defaults to
            `END_ID` (the top-level synthesized END); spliced flows namespace it.
        input_connections (`tuple[SeedConnection, ...]`):
            Derived child START bindings selected from a parent's materialized seed routes.
        child_boundaries (`tuple[ChildFlowBoundary, ...]`):
            Actual authored child flow boundaries, excluding generated fan nodes.
    """

    def __init__(self, nodes: dict[str, Node], edges: list[Edge],
                 wiring: Optional[dict[str, dict[str, Any]]] = None,
                 start_id: str = START_ID, end_id: str = END_ID, *,
                 input_connections: tuple[SeedConnection, ...] = (),
                 child_boundaries: tuple[ChildFlowBoundary, ...] = ()) -> None:
        self.nodes = nodes                              # node id -> Node
        self.edges = edges                              # all directed edges
        # Normalize once so every consumer below this construction boundary reads one shape.
        self.wiring: dict[str, dict[str, Binding]] = _normalize_wiring(wiring)
        self.start_id = start_id                        # single entry node id
        self.end_id = end_id                            # result node id
        self.input_connections = input_connections      # source-neutral materialized seed recipes
        self.child_boundaries = child_boundaries        # actual authored child boundaries


def validate_growth_metadata(flow: Flow) -> None:
    """Validate immutable growth recipes against their declared fragment targets.

    Both fragment acceptance and topology identity use this source-neutral graph
    check. Execution scopes may be absent on direct graphs; present scopes must
    identify the corresponding actual boundary and containing flow occurrence.
    """
    for name, carrier in (("input_connections", SeedConnection),
                          ("child_boundaries", ChildFlowBoundary)):
        values = getattr(flow, name)
        if type(values) is not tuple or any(type(value) is not carrier for value in values):
            raise TypeError(f"{name} must be a tuple of exact {carrier.__name__} values")
    starts: set[str] = set()
    endpoints: set[str] = set()
    for boundary in flow.child_boundaries:
        ChildFlowBoundary(boundary.start_id, boundary.end_id)
        scopes = []
        for node_id, role in ((boundary.start_id, "start"), (boundary.end_id, "end")):
            if node_id not in flow.nodes or node_id in endpoints:
                raise ValueError("child boundaries must name distinct fragment nodes")
            endpoints.add(node_id)
            node = flow.nodes[node_id]
            if not isinstance(node, Node):
                raise TypeError("child boundaries must name Node values")
            scope = node.execution_scope
            if scope is not None:
                if type(scope) is not ExecutionScope:
                    raise TypeError("child boundary scope must be an ExecutionScope")
                ExecutionScope(scope.static_path, scope.flow_path, scope.occurrence_path, scope.boundary)
                if scope.boundary != role:
                    raise ValueError("child boundary role disagrees with its execution scope")
                if not scope.static_path or scope.static_path[:-1] != scope.flow_path:
                    raise ValueError("child boundary location must belong to its containing flow")
            scopes.append(scope)
        if all(scope is not None for scope in scopes) and (
            scopes[0].flow_path != scopes[1].flow_path
            or scopes[0].occurrence_path != scopes[1].occurrence_path
        ):
            raise ValueError("child boundaries must belong to the same flow occurrence")
        starts.add(boundary.start_id)
    targets: set[tuple[str, str]] = set()
    for connection in flow.input_connections:
        SeedConnection(connection.child_start_id, connection.input_name, connection.seed_path)
        target = (connection.child_start_id, connection.input_name)
        if connection.child_start_id not in starts or target in targets:
            raise ValueError("seed connections must name unique declared child START inputs")
        targets.add(target)
        params = flow.nodes[connection.child_start_id].params
        if params is None:
            params = []
        if type(params) is not list or any(type(param) is not Parameter for param in params):
            raise TypeError("seed connection target parameters must be Parameter values")
        if connection.input_name not in {param.name for param in params}:
            raise ValueError("seed connection target input must be declared")
        if connection.input_name not in flow.wiring.get(connection.child_start_id, {}):
            raise ValueError("seed connection target input must be wired")


class CompiledFlow(Flow):
    """
    The compiled graph: nodes, edges, adjacency, and flow-owned input wiring.

    It holds no dispositions or variable values. The run boundary copies it for each
    execution; only that private copy is mutated when the reducer applies a prepared
    expansion or prunes a retired loop iteration. The authored instance therefore remains
    reusable across independent runs.

    Extends the internal `Flow` topology carrier with the runtime-only concerns below;
    the shared topology core (`nodes`/`edges`/`wiring`/`start_id`/`end_id`) lives on the base.

    Attributes:
        outputs (`list[FlowOutput]`):
            The declared flow `output:`; empty means the flow falls back to the raw terminal
            node value.
        flow_llm_config (`dict[str, Any]`):
            The flow's authored `llm_config:` model-selection defaults (the cascade's flow
            layer); `{}` when absent. `resolve_llm_cascade` gap-fills each AGENT node's
            effective config from this, the parent chain, the CLI layer, and process defaults
            captured once at run initialization. It returns the complete static-path map.
        flow_id (`str | None`):
            Stable authored semantic identity. Direct memoryless Python graphs may omit it.
        memory_policy (`MemoryPolicySpec | None`):
            Optional flow-owned named policy declaration. Runtime layout is derived later.
    """

    def __init__(self, nodes: dict[str, Node], edges: list[Edge],
                 outputs: Optional[list[FlowOutput]] = None,
                 wiring: Optional[dict[str, dict[str, Any]]] = None,
                 flow_llm_config: Optional[dict] = None,
                 flow_id: str | None = None,
                 memory_policy: MemoryPolicySpec | None = None) -> None:
        # The top-level compiled flow always roots at the synthesized START and returns END.
        super().__init__(nodes=nodes, edges=edges, wiring=wiring,
                         start_id=START_ID, end_id=END_ID)
        self.outputs = outputs or []                    # declared outputs; empty -> fall back to the raw terminal value
        # The flow's authored llm_config: defaults — the cascade's flow layer; {} when absent.
        self.flow_llm_config: dict = flow_llm_config or {}
        if flow_id is not None:
            if type(flow_id) is not str:
                raise TypeError("flow_id must be an exact string or None")
            if not flow_id:
                raise ValueError("flow_id must be non-empty")
        if memory_policy is not None and type(memory_policy) is not MemoryPolicySpec:
            raise TypeError(
                "memory_policy must be an exact MemoryPolicySpec or None"
            )
        self.flow_id = flow_id
        self.memory_policy = memory_policy
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
                   flow_llm_config: Optional[dict] = None,
                   flow_id: str | None = None,
                   memory_policy: MemoryPolicySpec | None = None) -> "CompiledFlow":
        """Build a CompiledFlow. The entry is always the synthesized START node (StartNode.ID),
        added implicitly by the loader / graph builder — never authored by the user; the engine
        seeds and advances it at run start."""
        return cls(nodes=nodes, edges=edges, outputs=outputs, wiring=wiring,
                   flow_llm_config=flow_llm_config, flow_id=flow_id,
                   memory_policy=memory_policy)

    def add_subgraph(self, nodes: dict[str, Node], edges: list[Edge],
                     wiring: dict[str, dict[str, Any]]) -> None:
        """Splice a deep-namespaced subgraph into one run's private topology.

        Called while the reducer applies a prepared expansion from CALL, MAP, or LOOP.
        Adds the fragment's nodes, edges, and wiring and updates adjacency; the same
        reducer plan updates dispositions and the ready frontier."""
        normalized_wiring = _normalize_wiring(wiring)
        self.nodes.update(nodes)
        self.edges.extend(edges)
        self.wiring.update(normalized_wiring)
        for edge in edges:
            self._outgoing.setdefault(edge.from_, []).append(edge)
            self._incoming.setdefault(edge.to, []).append(edge)

    def remove_subgraph(self, node_ids: set[str]) -> None:
        """Prune a finished deep-namespaced fragment from one run's private topology.

        The growth reducer uses this inverse of ``add_subgraph`` for a committed loop
        iteration whose ids are no longer referenced. It removes nodes, incident edges,
        wiring, and adjacency.

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
