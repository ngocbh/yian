"""CALL — the authorable REF driver (`kind: call`).

`CallNode` applies a callable ONCE (`'a flow -> 'b`). It is the REF half of the REF/MAP pair;
the MAP half is `nodes.map.MapNode` (`kind: map` + `over:`, `list['a] -> list['b]`) — the two are
distinct typed drivers. `CallNode` carries NO `over`/`parallel`.

`run` returns one seed-first `Grow` carrying the raw call-arg record. The compile-owned injected
builder constructs the child subgraph, and the generic growth reducer splices it. The child
START_ID owns omitted-input defaulting (its params carry default/required), so the driver no longer
pre-defaults. `child`/`child_inputs` are baked at load by
`compose.nodes.call.build`; `child_inputs` is read by compile validation
(`check_ref_map_types`). The child's asserts ride its own compiled boundary nodes and
survive the `clone_child` deep-copy of the spliced subgraph.
"""

from typing import Any, Callable, ClassVar, Iterator, Optional

from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.base import Grow, Node, NodeKind
from agent_composer.nodes.contracts import (
    Parameter,
    _iter_origin_carrier,
    _rewrite_origin_carrier,
)


class CallNode(Node):
    """
    The REF driver (`kind: call`) — apply a callable flow once (`'a flow -> 'b`).

    The REF half of the REF/MAP pair (the MAP half is
    [`MapNode`][agent_composer.nodes.map.node.MapNode]). `run` returns one seed-first `Grow`;
    the injected compile builder constructs the child fragment and the engine splices it. The
    spliced
    child START owns omitted-input defaulting, so the driver passes call-args raw.

    Args:
        node_id (`str`):
            The node's unique id.
        flow_id (`str`):
            The id of the child flow to call.
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

    kind = NodeKind.CALL
    is_spawner: ClassVar[bool] = True  # grows the graph: run() returns a seed-first Grow
    grow_depth_delta: ClassVar[int] = 1  # each child is one REF level deeper (bounded by MAX_REF_DEPTH)

    def __init__(self, node_id: str, *, flow_id: str, flow_version: Optional[int] = None,
                 child: Any = None, child_inputs: Optional[list[Parameter]] = None,
                 child_source: Any = None, title: Optional[str] = None) -> None:
        super().__init__(node_id, title=title)
        self.flow_id = flow_id
        self.flow_version = flow_version
        self.child = child
        self.child_inputs = child_inputs or []
        # Render-only: the child's SourceFrame for the CLI error traceback. `run` never reads it.
        self.child_source = child_source

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
        _carrier_memo: Optional[dict[int, object]] = None,
    ) -> None:
        """Rewrite common and cached-signature origins on a copied CALL."""
        memo = _carrier_memo if _carrier_memo is not None else {}
        super().rewrite_local_origins(mapper, _carrier_memo=memo)
        for index, parameter in enumerate(self.child_inputs):
            self.child_inputs[index] = _rewrite_origin_carrier(
                parameter, mapper, memo
            )

    def origin_children(self) -> tuple[Any, ...]:
        """Return the baked child flow, or no child for an unbaked CALL."""
        return () if self.child is None else (self.child,)

    def _run_impl(self, inputs: dict, **caps: Any):
        if self.child is None:
            raise RuntimeError(f"CALL node {self.id!r}: child flow {self.flow_id!r} not baked")
        # Pass the call-args RAW: the spliced child START_ID owns omitted-input defaulting now (its
        # params carry default/required), so the driver no longer pre-defaults. The node describes
        # growth with only the durable builder input; the compile-owned injected builder constructs
        # and lowers post-asserts onto the child END for both live execution and replay.
        record = dict(inputs)
        return Grow(seed=record)
