"""START_ID — the flow's canonical parameter-binding boundary, internal-only. PURE.

The loader synthesizes one StartNode per flow as its single root. The generic
``eval_node -> bind_params -> Node.run`` path owns coercion, validation, required
inputs, and defaults; START's body is the identity over that bound record.

The flow's PRE `asserts:` are lowered onto START's ordinary `post_asserts` (each `${input.X}`
rewritten to `${output.X}`) and fire through the generic wrapper/`eval_node` seam against START's
COERCED output record — START is an ordinary scheduled node, not a bypass. This unifies top-level
and child (subflow) boundary asserts: both flow through the same post-assert seam. NOT START_ID's
job: system/clock/run_id. Hidden from authors: a reserved __-prefixed id, never parseable.
"""

from typing import Optional

from agent_composer.nodes.base import FieldRanges, Node, NodeKind, Output
from agent_composer.nodes.contracts import Parameter


class StartNode(Node):
    """
    The flow's parameter-binding boundary — internal-only, loader-synthesized, pure.

    One per flow (the single root). Its params are the input names; the engine reads each source
    from `flow.wiring[START_ID]` (the run-arg seed at top level, the call-arg edges in a child).
    ``run`` returns the already-bound record as one object keyed by input name.

    Args:
        node_id (`str`):
            The node's id (`ID` at top level; deep-namespaced in a child).
        params (`list[Parameter]`, *optional*, defaults to `None`):
            The flow's canonical semantic input signature.
        field_ranges (`dict`, *optional*, defaults to `None`):
            Render-only named boundary field ranges used to locate input declaration failures.
            Flow assertions use the separate positional range bridge. `None` for a spanless flow.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.START

    #: Reserved id of the synthesized input boundary (the old `__start__` sentinel).
    #: The canonical source of truth; `compile.model` re-exports it as `START_ID`/`START_ID`.
    #: At top level the loader mints one StartNode with this id; a child flow's is
    #: deep-namespaced (`<callsite>/__start__`). Authors can never name a node this
    #: (parser reserved-id ban).
    ID = "__start__"

    def __init__(
        self,
        node_id: str,
        *,
        params: Optional[list[Parameter]] = None,
        field_ranges: Optional[dict] = None,
        title: Optional[str] = None,
    ) -> None:
        super().__init__(node_id, title=title)
        parameters = [] if params is None else params
        if type(parameters) is not list:
            raise TypeError(
                f"START params must be an exact list, got {type(parameters).__name__}"
            )
        for parameter in parameters:
            if type(parameter) is not Parameter:
                raise TypeError(
                    "START params must contain Parameter values, "
                    f"got {type(parameter).__name__}"
                )
        self.params = parameters
        # Render-only per-boundary-field source ranges: the flow's `input:` decls, keyed
        # `("input", name)` like every other named site. The loader stamps these so binding can
        # attach a precise `range` to a boundary type failure — the flow-boundary twin of a body
        # node's per-field ranges. Empty for a spanless / in-memory flow. (Flow-level `asserts:`
        # are NOT here: their text is not a unique identity, so they are located by position via
        # `pre_assert_ranges`/`post_assert_ranges`.) Rewrapped as `FieldRanges` so the bare-key
        # guard survives a caller passing a plain dict.
        if field_ranges:
            self.field_ranges = FieldRanges(field_ranges)
    def _run_impl(self, inputs: dict) -> Output:
        """Return the record already bound by the generic node input seam."""
        return Output(value=inputs)
