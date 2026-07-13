"""MAP — the `List.map` driver (`kind: map` + `over:`), internal-only build target.

`MapNode` is the mapped-call node, distinct from REF's `CallNode`. The two are different typed
drivers: a `CallNode` applies a callable ONCE (`'a flow -> 'b`); a `MapNode` maps it over a
list (`list['a] -> list['b]`). The discriminator is the KIND (`NodeKind.MAP`) — `MapNode` carries
NO `over` attribute and no `${...}` source on the node. The `over` SOURCE binding rides
`flow.wiring[id]["over"]` (mirroring WaitNode's timed `until`), pre-resolved into `inputs["over"]`
by the engine's `eval_node` before `run`.

`run` returns one `Grow(Flow)` — self-describing expansion: it builds the whole MAP fan-in
subgraph (`map_subgraph` — N per-element child clones PLUS a synthesized `EndNode.list_` fan-in over
the child ENDs) and the engine's generic `_apply_grow` splices it into the live graph, with a
per-kind MAP residual (depth/refdepth/finish-mark). The `Grow.seed` is the raw list of per-element
records (the durable builder input); an empty `over` -> a subgraph whose sole node is the list-`END`
(it emits `[]`). The per-element call-args go RAW: the spliced child START_ID owns omitted-input
defaulting (its params carry default/required), so the driver no longer pre-defaults.
`child`/`child_inputs` are baked at load by `compose.nodes.map.build`; `child_inputs` is read by
compile validation (`check_ref_map_types`). The child's asserts ride its own compiled boundary
nodes and survive the `clone_child` deep-copy per element. `parallel` is inert (concurrency is the
engine's `num_workers`); it is carried for the over case.
"""

from typing import Any, Callable, ClassVar, Optional

from agent_composer.compile.expand import map_subgraph
from agent_composer.expr import eval_binding, resolve_reference
from agent_composer.nodes.base import Grow, Node, NodeKind


class MapNode(Node):
    """
    The `List.map` driver (`kind: map` + `over:`) — map a callable flow over a list (`list['a] -> list['b]`).

    The MAP half of the REF/MAP pair (the REF half is
    [`CallNode`][agent_composer.nodes.call.node.CallNode]). The `over` source is pre-resolved into
    `inputs["over"]` by the engine bind seam; `run` returns one `Grow(Flow)` — it builds the
    whole MAP fan-in (N per-element child clones + a synthesized list END) via `map_subgraph`, and
    the engine's generic `_apply_grow` splices it and grows the live graph. The spliced child START
    owns omitted-input defaulting, so per-element call-args go raw.

    Args:
        node_id (`str`):
            The node's unique id.
        flow_id (`str`):
            The id of the child flow to map.
        parallel (`bool`, *optional*, defaults to `False`):
            Inert — concurrency is the engine's `num_workers`; carried for the `over` case.
        flow_version (`int`, *optional*, defaults to `None`):
            A pinned child flow version, if any.
        child (`Any`, *optional*, defaults to `None`):
            The baked child flow (stamped at load); a `None` child raises at run time.
        child_inputs (`list`, *optional*, defaults to `None`):
            The child's input decls, read by compile validation.
        child_source (`Any`, *optional*, defaults to `None`):
            The child flow's render-only `SourceFrame` (label + node-line maps). Carried
            for the CLI's nested-error traceback only; `run` ignores it entirely.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.MAP
    is_spawner: ClassVar[bool] = True  # grows the graph: run() returns a Grow(Flow)
    binds_per_item: ClassVar[bool] = True  # binds call-args PER ELEMENT via `bind_item`, not up front
    grow_depth_delta: ClassVar[int] = 1  # each child is one REF level deeper (bounded by MAX_REF_DEPTH)

    def __init__(self, node_id: str, *, flow_id: str, parallel: bool = False,
                 flow_version: Optional[int] = None, child: Any = None,
                 child_inputs: Optional[list] = None,
                 child_source: Any = None, title: Optional[str] = None) -> None:
        super().__init__(node_id, title=title)
        self.flow_id = flow_id
        self.flow_version = flow_version
        self.parallel = parallel    # inert (the engine's num_workers)
        self.child = child
        self.child_inputs = child_inputs or []
        # Render-only: the child's SourceFrame for the CLI error traceback. `run` never reads it.
        self.child_source = child_source

    def bind_reserved(self, node_wiring: dict, pool) -> dict:
        """Pre-resolve the MAP `over` source into the list `run` maps over.

        Returns `{"over": <list>}`: the `over` source rides `node_wiring["over"]`, evaluated
        against `pool`. A source that resolves to `None`/non-list is a loud `RuntimeError`
        (surfaced as a NodeFailed by the read seam)."""
        over_src = node_wiring["over"]
        items = eval_binding(over_src, lambda p: resolve_reference(p, pool))
        if items is None or not isinstance(items, list):
            raise RuntimeError(
                f"MAP node {self.id!r}: `over` ({over_src}) did not resolve to a list"
            )
        return {"over": items}

    def reserved_wiring_keys(self) -> set[str]:
        """A MAP reserves `over` (its iteration-source author-wiring key)."""
        return {"over"}

    def run(self, inputs: dict, *, bind_item: Optional[Callable[[Any], dict]] = None):
        if self.child is None:
            raise RuntimeError(f"MAP node {self.id!r}: child flow {self.flow_id!r} not baked")
        # Per-element call-args RAW (no driver pre-default): the spliced child START_ID fills omitted
        # inputs from its params' declared defaults. `run` is self-describing — it builds the whole
        # MAP fan-in subgraph and returns a `Grow` for the engine to splice generically;
        # `seed=records` is the durable builder input (a resumed run rebuilds the same subgraph via
        # `map_subgraph(child, self.id, records)`).
        records = [dict(bind_item(el)) for el in inputs["over"]]
        return Grow(map_subgraph(self.child, spawner_id=self.id, records=records), seed=records)

    def replay_grow(self, seed: Any):
        """Durable-replay inverse of `run`: rebuild the whole MAP fan-in subgraph from the persisted
        per-element records (`seed`) via the SAME `map_subgraph` builder — no body re-run. The
        resolved `over:` list is captured on the seed, so replay needs no re-binding."""
        return map_subgraph(self.child, spawner_id=self.id, records=[dict(r) for r in seed])
