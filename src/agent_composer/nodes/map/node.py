"""MAP — the authorable `List.map` driver (`kind: map` + `over:`).

`MapNode` is the mapped-call node, distinct from REF's `CallNode`. The two are different typed
drivers: a `CallNode` applies a callable ONCE (`'a flow -> 'b`); a `MapNode` maps it over a
list (`list['a] -> list['b]`). The discriminator is the KIND (`NodeKind.MAP`) — `MapNode` carries
NO `over` attribute and no `${...}` source on the node. The `over` SOURCE binding rides
`flow.wiring[id]["over"]` (mirroring WaitNode's absolute `until`), pre-resolved into `inputs["over"]`
by the engine's `eval_node` before `run`.

`run` returns one seed-first `Grow` carrying the raw list of per-element records. The compile-owned
injected builder constructs the MAP fan-in, and the generic growth reducer splices it. An
empty `over` builds a synthetic start wired directly to the list END (which emits `[]`). The
per-element call-args go RAW: the spliced child START_ID owns omitted-input
defaulting (its params carry default/required), so the driver no longer pre-defaults.
`child`/`child_inputs` are baked at load by `compose.nodes.map.build`; `child_inputs` is read by
compile validation (`check_ref_map_types`). The child's asserts ride its own compiled boundary
nodes and survive the `clone_child` deep-copy per element. `parallel` is inert (concurrency is the
engine's `num_workers`); it is carried for the over case.
"""

from typing import Any, Callable, ClassVar, Optional

from agent_composer.nodes.base import Grow, Node, NodeKind
from agent_composer.nodes.binding import _resolve_source
from agent_composer.nodes.contracts import Parameter, binding_source


class MapNode(Node):
    """
    The `List.map` driver (`kind: map` + `over:`) — map a callable flow over a list (`list['a] -> list['b]`).

    The MAP half of the REF/MAP pair (the REF half is
    [`CallNode`][agent_composer.nodes.call.node.CallNode]). The `over` source is pre-resolved into
    `inputs["over"]` by the engine bind seam; `run` returns one seed-first `Grow`. The injected
    compile builder constructs the MAP fan-in and the engine splices it. The spliced child START
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
        child_inputs (`list[Parameter]`, *optional*, defaults to `None`):
            The child's canonical parameters, read by compile validation.
        child_source (`Any`, *optional*, defaults to `None`):
            The child flow's render-only `SourceFrame` (label + node-line maps). Carried
            for the CLI's nested-error traceback only; `run` ignores it entirely.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.MAP
    is_spawner: ClassVar[bool] = True  # grows the graph: run() returns a seed-first Grow
    binds_per_item: ClassVar[bool] = True  # binds call-args PER ELEMENT via `bind_item`, not up front
    grow_depth_delta: ClassVar[int] = 1  # each child is one REF level deeper (bounded by MAX_REF_DEPTH)

    def __init__(self, node_id: str, *, flow_id: str, parallel: bool = False,
                 flow_version: Optional[int] = None, child: Any = None,
                 child_inputs: Optional[list[Parameter]] = None,
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
        items = _resolve_source(over_src, pool)
        if items is None or not isinstance(items, list):
            display_src = binding_source(over_src)
            raise RuntimeError(
                f"MAP node {self.id!r}: `over` ({display_src}) did not resolve to a list"
            )
        return {"over": items}

    def reserved_wiring_keys(self) -> set[str]:
        """A MAP reserves `over` (its iteration-source author-wiring key)."""
        return {"over"}

    def _run_impl(self, inputs: dict, *, bind_item: Optional[Callable[[Any], dict]] = None):
        if self.child is None:
            raise RuntimeError(f"MAP node {self.id!r}: child flow {self.flow_id!r} not baked")
        # Per-element call-args RAW (no driver pre-default): the spliced child START_ID fills omitted
        # inputs from its params' declared defaults. The node describes growth with only the
        # per-element records; the compile-owned injected builder constructs the MAP fan-in.
        records = [dict(bind_item(el)) for el in inputs["over"]]
        return Grow(seed=records)
