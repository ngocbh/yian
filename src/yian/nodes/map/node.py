"""MAP — the authorable `List.map` driver (`kind: map` + `over:`).

`MapNode` is the mapped-call node, distinct from the single-call `CallNode`. The two are different typed
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
compile validation (`check_call_map_types`). The child's asserts ride its own compiled boundary
nodes and survive the `clone_child` deep-copy per element. `parallel` is inert (concurrency is the
engine's `num_workers`); it is carried for the over case.
"""

from typing import Any, Callable, ClassVar, Iterator, Optional

from yian.diagnostics import OriginRef
from yian.nodes.base import Grow, Node, NodeKind
from yian.nodes.binding import (
    BindingError, ReservedBinder, _reject_invalid_binding, _resolve_source, _source_origin,
)
from yian.nodes.contracts import (
    Parameter,
    _iter_origin_carrier,
    _rewrite_origin_carrier,
    binding_source,
)
from yian.typesys.values import value_repr


class MapNode(Node):
    """
    The `List.map` driver (`kind: map` + `over:`) — map a callable flow over a list (`list['a] -> list['b]`).

    The mapped counterpart to [`CallNode`][yian.nodes.call.node.CallNode].
    The `over` source is pre-resolved into
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
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.MAP
    is_spawner: ClassVar[bool] = True  # grows the graph: run() returns a seed-first Grow
    binds_per_item: ClassVar[bool] = True  # binds call-args PER ELEMENT via `bind_item`, not up front
    item_source_key: ClassVar[str] = "over"
    grow_depth_delta: ClassVar[int] = 1  # each child is one call level deeper

    def __init__(self, node_id: str, *, flow_id: str, parallel: bool = False,
                 flow_version: Optional[int] = None, child: Any = None,
                 child_inputs: Optional[list[Parameter]] = None,
                 title: Optional[str] = None) -> None:
        super().__init__(node_id, title=title)
        self.flow_id = flow_id
        self.flow_version = flow_version
        self.parallel = parallel    # inert (the engine's num_workers)
        self.child = child
        self.child_inputs = child_inputs or []

    def iter_local_origins(
        self,
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> Iterator[OriginRef]:
        """Yield common origins followed by the cached child signature."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        yield from super().iter_local_origins(_seen_carriers=seen)
        for parameter in self.child_inputs:
            yield from _iter_origin_carrier(parameter, seen)

    def rewrite_local_origins(
        self,
        mapper: Callable[[OriginRef], OriginRef],
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> None:
        """Rewrite common and cached-signature origins on a copied MAP."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        super().rewrite_local_origins(mapper, _seen_carriers=seen)
        for parameter in self.child_inputs:
            _rewrite_origin_carrier(parameter, mapper, seen)

    def origin_children(self) -> tuple[Any, ...]:
        """Return the baked child flow, or no child for an unbaked MAP."""
        return () if self.child is None else (self.child,)

    def validate_execution_setup(self) -> None:
        """Require the child flow that Compose must bake before execution."""
        super().validate_execution_setup()
        if self.child is None:
            raise RuntimeError(f"MAP node {self.id!r}: child flow {self.flow_id!r} not baked")

    def bind_reserved(
        self, node_wiring: dict, pool, *, bind: ReservedBinder | None = None,
    ) -> dict:
        """Pre-resolve the MAP `over` source into the list `run` maps over.

        Returns `{"over": <list>}`: the `over` source rides `node_wiring["over"]`, evaluated
        against `pool`. A source that resolves to `None`/non-list raises a binder-owned
        ``BindingError`` classified as ``contract.input`` by the read seam; a list nesting
        deeper than `MAX_VALUE_NESTING` levels or holding a NaN, infinity, or complex
        number at any depth raises one classified as ``binding.evaluate``."""
        over_src = node_wiring["over"]

        def validate(items):
            source = binding_source(over_src)
            # An authored literal may hold an int too long for Python to write as text.
            display_src = source if isinstance(source, str) else value_repr(source)
            if items is None or not isinstance(items, list):
                raise BindingError(
                    f"MAP node {self.id!r}: `over` ({display_src}) did not resolve to a list",
                    failure_code="contract.input",
                    origin=_source_origin(over_src),
                )
            _reject_invalid_binding(
                items, f"MAP node {self.id!r}: `over` ({display_src})", _source_origin(over_src)
            )
            return items

        items = (
            validate(_resolve_source(over_src, pool)) if bind is None
            else bind("over", over_src, validate)
        )
        return {"over": items}

    def reserved_wiring_keys(self) -> set[str]:
        """A MAP reserves `over` (its iteration-source author-wiring key)."""
        return {"over"}

    def forward(self, inputs: dict, *, bind_item: Optional[Callable[[Any, int], dict]] = None):
        # Per-element call-args RAW (no driver pre-default): the spliced child START_ID fills omitted
        # inputs from its params' declared defaults. The node describes growth with only the
        # per-element records; the compile-owned injected builder constructs the MAP fan-in.
        records = [dict(bind_item(el, index)) for index, el in enumerate(inputs["over"])]
        return Grow(seed=records)
