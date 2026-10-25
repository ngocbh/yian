"""START_ID — the flow's canonical parameter-binding boundary, internal-only. PURE.

The loader synthesizes one StartNode per flow as its single root. The generic
``eval_node -> bind_params -> Node.run`` path owns validation, required inputs, and
defaults; START's body is the identity over that bound record. Only the top-level START
sees converted text, because ``run_flow`` converts supplied inputs before seeding it.

The flow's PRE `asserts:` are lowered onto START's ordinary `post_asserts` (each `${input.X}`
rewritten to `${output.X}`) and fire through the generic wrapper/`eval_node` seam against START's
BOUND output record — START is an ordinary scheduled node, not a bypass. This unifies top-level
and child-flow boundary asserts: both flow through the same post-assert seam. NOT START_ID's
job: system/clock/run_id. Hidden from authors: a reserved __-prefixed id, never parseable.
"""

from typing import Optional

from yian.expr.lineage import FeedbackValue
from yian.nodes.base import (
    BackwardContext, BackwardResult, Node, NodeKind, Output, _project_structural_feedback,
)
from yian.nodes.binding import _validate_parameters
from yian.nodes.contracts import Parameter


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
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.START
    commits_input_record = True

    #: Reserved id of the synthesized input boundary.
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
        title: Optional[str] = None,
    ) -> None:
        super().__init__(node_id, title=title)
        parameters = [] if params is None else params
        _validate_parameters(parameters, "START")
        self.params = parameters

    def forward(self, inputs: dict) -> Output:
        """Return the record already bound by the generic node input seam."""
        return Output(value=inputs)

    def capture_backward_context(self) -> dict:
        """Capture the named record projection independently of delivered presence.

        Returns:
            `dict`: Each declared input maps to its same-named output field in
                `input_output_paths`; an omitted optional input keeps its path.
        """
        return {"input_output_paths": {name: [name] for name in self.backward_input_names()}}

    def backward(
        self, context: BackwardContext, feedback: tuple[FeedbackValue, ...],
    ) -> BackwardResult:
        """Project feedback to declared inputs present in the captured START record.

        Args:
            context (`BackwardContext`): Recorded bound inputs, output or failure,
                declaration inventory, and the captured named output paths.
            feedback (`tuple[FeedbackValue, ...]`): Independent whole-record text
                or structured selections. Failed START accepts text or `None`.

        Returns:
            `BackwardResult`: Messages for delivered inputs, including defaults;
                omitted inputs and unavailable bindings receive no messages.

        Raises:
            `TypeError`: If a message batch or captured path has an invalid type.
            `ValueError`: If feedback or a captured path does not match the
                recorded output, or the required structural layout is absent.
        """
        if context.failure is None and type(context.output) is not dict:
            return BackwardResult()
        return _project_structural_feedback(context, feedback)
